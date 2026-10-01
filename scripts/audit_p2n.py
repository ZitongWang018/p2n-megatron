"""Check a trained checkpoint against independent Qwen3 and gradient references."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import Qwen3Config, Qwen3ForCausalLM

from megatron.core import parallel_state
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from p2n.model import MODEL_PROFILES, P2NGPTModel, make_transformer_config
from p2n.data import TokenStream


def emit(check, **values):
    print(json.dumps({"check": check, **values}), flush=True)


def reference_model(model, p):
    config = Qwen3Config(
        vocab_size=p["vocab_size"], hidden_size=p["hidden_size"],
        intermediate_size=p["ffn_hidden_size"], num_hidden_layers=p["num_layers"],
        num_attention_heads=p["num_attention_heads"], num_key_value_heads=p["num_query_groups"],
        head_dim=p["kv_channels"], rms_norm_eps=1e-6, rope_theta=1_000_000,
        attention_bias=False, attention_dropout=0, tie_word_embeddings=p["tie_embeddings"],
        sliding_window=None,
    )
    config._attn_implementation = "eager"
    hf = Qwen3ForCausalLM(config).cuda().eval()
    groups, dim = p["num_query_groups"], p["kv_channels"]
    queries_per_group = p["num_attention_heads"] // groups
    with torch.no_grad():
        hf.model.embed_tokens.weight.copy_(model.embedding.word_embeddings.weight)
        hf.lm_head.weight.copy_(
            model.shared_embedding_or_output_weight()
            if p["tie_embeddings"] else model.output_layer.weight
        )
        hf.model.norm.weight.copy_(model.decoder.final_layernorm.weight)
        for src, dst in zip(model.decoder.layers, hf.model.layers):
            qkv = src.self_attention.linear_qkv.weight.view(groups, queries_per_group + 2, dim, -1)
            dst.self_attn.q_proj.weight.copy_(qkv[:, :queries_per_group].reshape(-1, p["hidden_size"]))
            dst.self_attn.k_proj.weight.copy_(qkv[:, queries_per_group].reshape(-1, p["hidden_size"]))
            dst.self_attn.v_proj.weight.copy_(qkv[:, queries_per_group + 1].reshape(-1, p["hidden_size"]))
            dst.self_attn.o_proj.weight.copy_(src.self_attention.linear_proj.weight)
            dst.self_attn.q_norm.weight.copy_(src.self_attention.q_layernorm.weight)
            dst.self_attn.k_norm.weight.copy_(src.self_attention.k_layernorm.weight)
            dst.input_layernorm.weight.copy_(src.input_layernorm.weight)
            dst.post_attention_layernorm.weight.copy_(src.pre_mlp_layernorm.weight)
            gate, up = src.mlp.linear_fc1.weight.chunk(2, dim=0)
            dst.mlp.gate_proj.weight.copy_(gate)
            dst.mlp.up_proj.weight.copy_(up)
            dst.mlp.down_proj.weight.copy_(src.mlp.linear_fc2.weight)
    return hf


def reference_forward(hf, ids, p, k):
    # Build document masks independently from the production helper.
    batch, seq = ids.shape
    positions = torch.arange(seq, device=ids.device)[None].expand(batch, -1)
    allowed = torch.zeros(batch, 1, seq, seq, dtype=torch.bool, device=ids.device)
    for b in range(batch):
        document_start = 0
        for t in range(seq):
            if t and int(ids[b, t - 1]) == 0:
                document_start = t
            allowed[b, 0, t, document_start:t + 1] = True
    mask = torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, float("-inf"))
    hidden = hf.model.embed_tokens(ids)
    rope = hf.model.rotary_emb(hidden, positions)

    def layers(x, begin, end):
        for layer in hf.model.layers[begin:end]:
            x = layer(x, attention_mask=mask, position_ids=positions,
                      position_embeddings=rope, use_cache=False)[0]
        return x

    prefix = layers(hidden, 0, p["core_start"])
    state = layers(prefix, p["core_start"], p["core_end"])
    for _ in range(k):
        previous = torch.zeros_like(state)
        previous[:, 1:] = state[:, :-1] * (ids[:, :-1] != 0).unsqueeze(-1)
        state = layers(prefix + previous, p["core_start"], p["core_end"])
    hidden = layers(state, p["core_end"], p["num_layers"])
    return hf.lm_head(hf.model.norm(hidden))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    torch.manual_seed(42)
    model_parallel_cuda_manual_seed(42)
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        profile = checkpoint["arguments"]["profile"]
        p = MODEL_PROFILES[profile]
        optimizer_steps = sorted({int(s["step"]) for s in checkpoint["optimizer"]["state"].values()})
        assert optimizer_steps == [checkpoint["step"]]
        emit("checkpoint", step=checkpoint["step"], optimizer_steps=optimizer_steps,
             arguments=checkpoint["arguments"])
        model = P2NGPTModel(make_transformer_config(profile=profile), method="p2n", eod_id=0,
                            profile=profile).cuda().eval()
        model.load_state_dict(checkpoint["model"])
        assert len(checkpoint["optimizer"]["state"]) == len(list(model.parameters()))
        stream = TokenStream(checkpoint["arguments"]["train_bin"], vocab_size=p["vocab_size"])
        sample_x, sample_y = stream.batch(first_token=12345, batch_size=2, seq_len=64,
                                          device=torch.device("cpu"))
        train_path = Path(checkpoint["arguments"]["train_bin"])
        raw = (
            np.load(sorted(train_path.glob("input_ids_2048_*.npy"))[0], mmap_mode="r").reshape(-1)
            if train_path.is_dir() else np.memmap(train_path, mode="r", dtype=np.uint16)
        )
        for i in range(2):
            start = 12345 + i * 64
            assert np.array_equal(sample_x[i].numpy(), raw[start:start + 64])
            assert np.array_equal(sample_y[i].numpy(), raw[start + 1:start + 65])
        emit("data_and_next_token_labels", passed=True)
        del checkpoint
        hf = reference_model(model, p)
        ids = torch.randint(1, 40000, (2, 64), device="cuda")
        ids[:, 31] = 0
        with torch.no_grad():
            for k in (0, 2, 3):
                actual = model(ids, jacobi_iterations=k)
                expected = reference_forward(hf, ids, p, k)
                error = float((actual - expected).abs().max())
                emit("huggingface_qwen3_parity", k=k, max_abs_error=error)
                torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            original = model(ids, jacobi_iterations=3)
            changed = ids.clone()
            changed[:, 48:] = 200
            torch.testing.assert_close(original[:, :48], model(changed, jacobi_iterations=3)[:, :48], atol=1e-5, rtol=1e-5)
            changed = ids.clone()
            changed[:, :31] = 300
            torch.testing.assert_close(original[:, 32:], model(changed, jacobi_iterations=3)[:, 32:], atol=1e-5, rtol=1e-5)
            emit("causality_and_document_isolation", passed=True)
        del hf
        model.train()
        labels = torch.roll(ids, shifts=-1, dims=1)
        gradients = None
        for enabled in (False, True):
            model.activation_checkpointing = enabled
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, jacobi_iterations=3)
                loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, p["vocab_size"]), labels.reshape(-1))
            loss.backward()
            current = {n: v.grad.detach().cpu().clone() for n, v in model.named_parameters() if v.grad is not None}
            assert len(current) == len(list(model.parameters())), "missing parameter gradients"
            assert all(torch.isfinite(g).all() for g in current.values())
            if gradients is None:
                gradients = current
            else:
                error = max(float((current[n] - gradients[n]).abs().max()) for n in current)
                for name in current:
                    torch.testing.assert_close(current[name], gradients[name], atol=1e-6, rtol=1e-4)
                emit("checkpoint_gradient_parity", parameters_checked=len(current), max_abs_error=error)
        del gradients, current
        model.activation_checkpointing = False
        model.zero_grad(set_to_none=True)
        logits = model(ids, jacobi_iterations=3)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, p["vocab_size"]), labels.reshape(-1))
        loss.backward()
        full_grads = {n: v.grad.detach().cpu().clone() for n, v in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        for i in range(2):
            logits = model(ids[i:i+1], jacobi_iterations=3)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, p["vocab_size"]), labels[i].reshape(-1))
            (loss / 2).backward()
        error = 0.0
        for name, parameter in model.named_parameters():
            actual = parameter.grad.cpu()
            error = max(error, float((actual - full_grads[name]).abs().max()))
            torch.testing.assert_close(actual, full_grads[name], atol=2e-6, rtol=2e-3)
        emit("gradient_accumulation_parity", max_abs_error=error)
        del full_grads
        model.activation_checkpointing = False
        model.zero_grad(set_to_none=True)
        calls = [0] * p["num_layers"]
        states, handles = [], []
        for index, layer in enumerate(model.decoder.layers):
            def hook(module, inputs, output, i=index):
                calls[i] += 1
                if i == p["core_end"] - 1:
                    output[0].retain_grad()
                    states.append(output[0])
            handles.append(layer.register_forward_hook(hook))
        model(ids, jacobi_iterations=3).square().mean().backward()
        expected_calls = [4 if p["core_start"] <= i < p["core_end"] else 1 for i in range(p["num_layers"])]
        assert calls == expected_calls
        state_grad_norms = [float(s.grad.norm()) for s in states]
        assert len(states) == 4 and all(x > 0 for x in state_grad_norms)
        emit("full_bptt_and_layer_counts", calls=calls, state_grad_norms=state_grad_norms)
        for handle in handles:
            handle.remove()
        emit("complete", passed=True)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
