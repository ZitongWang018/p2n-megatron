"""Parameter-free P2N recurrence over a Megatron Core GPT decoder.

The core is evaluated once for the warm state and K more times for Jacobi
updates. Prefix and suffix are each evaluated exactly once. Gradients flow
through every core evaluation; all evaluations reuse the same layer objects.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal


PAPER_600M = {
    "num_layers": 21,
    "hidden_size": 1280,
    "ffn_hidden_size": 5632,
    "num_attention_heads": 20,
    "num_query_groups": 5,
    "kv_channels": 64,
    "vocab_size": 50304,
    "core_start": 7,  # zero-based, inclusive
    "core_end": 14,  # zero-based, exclusive
    "parameter_count": 604_627_328,
}


def make_transformer_config(*, profile: str = "paper-600m") -> TransformerConfig:
    if profile != "paper-600m":
        raise ValueError(f"unknown model profile: {profile}")
    p = PAPER_600M
    return TransformerConfig(
        num_layers=p["num_layers"],
        hidden_size=p["hidden_size"],
        ffn_hidden_size=p["ffn_hidden_size"],
        num_attention_heads=p["num_attention_heads"],
        num_query_groups=p["num_query_groups"],
        kv_channels=p["kv_channels"],
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        attention_backend=AttnBackend.local,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        qk_layernorm=True,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        add_qkv_bias=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        init_method_std=0.02,
        output_layer_init_method=init_method_normal(0.02),
        use_cpu_initialization=False,
        sequence_parallel=False,
        masked_softmax_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        apply_rope_fusion=False,
        cross_entropy_loss_fusion=False,
    )


def shift_previous(states: torch.Tensor, input_ids: torch.Tensor, eod_id: int) -> torch.Tensor:
    """Shift [seq,batch,hidden] one token right, zeroing sequence boundaries."""
    shifted = torch.cat((torch.zeros_like(states[:1]), states[:-1]), dim=0)
    if eod_id >= 0:
        # EOD belongs to its current document; the next token starts a new one.
        valid = (input_ids[:, :-1] != eod_id).transpose(0, 1).unsqueeze(-1)
        shifted = torch.cat((shifted[:1], shifted[1:] * valid), dim=0)
    return shifted


def causal_document_mask(input_ids: torch.Tensor, eod_id: int) -> torch.Tensor:
    """True means masked, including tokens from earlier packed documents."""
    batch, seq = input_ids.shape
    future = torch.ones((seq, seq), dtype=torch.bool, device=input_ids.device).triu_(1)
    if eod_id < 0:
        return future[None, None].expand(batch, 1, seq, seq)
    starts = torch.zeros_like(input_ids, dtype=torch.long)
    starts[:, 1:] = (input_ids[:, :-1] == eod_id).long()
    segment = starts.cumsum(dim=1)
    other_document = segment[:, :, None] != segment[:, None, :]
    return future[None, None] | other_document[:, None]


class P2NGPTModel(GPTModel):
    def __init__(
        self, config: TransformerConfig, *, method: str, eod_id: int,
        activation_checkpointing: bool = False,
    ):
        if method not in {"p2n", "vanilla"}:
            raise ValueError(f"unknown method: {method}")
        super().__init__(
            config=config,
            transformer_layer_spec=get_gpt_layer_local_spec(
                qk_layernorm=True, normalization="RMSNorm"
            ),
            vocab_size=PAPER_600M["vocab_size"],
            max_sequence_length=2048,
            pre_process=True,
            post_process=True,
            parallel_output=True,
            share_embeddings_and_output_weights=True,
            position_embedding_type="rope",
            rotary_percent=1.0,
            rotary_base=1_000_000,
        )
        self.method = method
        self.eod_id = eod_id
        self.activation_checkpointing = activation_checkpointing
        self.core_start = PAPER_600M["core_start"]
        self.core_end = PAPER_600M["core_end"]
        if len(self.decoder.layers) != PAPER_600M["num_layers"]:
            raise AssertionError("Megatron built an unexpected number of layers")

    def _run_layers(self, hidden, begin, end, mask, rope):
        for layer in self.decoder.layers[begin:end]:
            if self.activation_checkpointing and self.training:
                def run(x, current_layer=layer):
                    result, _ = current_layer(
                        hidden_states=x, attention_mask=mask, rotary_pos_emb=rope
                    )
                    return result

                hidden = checkpoint(run, hidden, use_reentrant=False)
            else:
                hidden, _ = layer(
                    hidden_states=hidden,
                    attention_mask=mask,
                    rotary_pos_emb=rope,
                )
        return hidden

    def forward(self, input_ids: torch.Tensor, *, jacobi_iterations: int = 0):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        if self.method == "p2n" and jacobi_iterations < 1:
            raise ValueError("P2N needs at least one Jacobi iteration")
        if self.method == "vanilla" and jacobi_iterations != 0:
            raise ValueError("Vanilla does not use Jacobi iterations")
        batch, seq = input_ids.shape
        positions = torch.arange(seq, device=input_ids.device)[None, :].expand(batch, -1)
        mask = causal_document_mask(input_ids, self.eod_id)
        hidden, rope, _, _, _ = self._preprocess(input_ids, positions)
        hidden = self._run_layers(hidden, 0, self.core_start, mask, rope)
        core_input = hidden
        state = self._run_layers(core_input, self.core_start, self.core_end, mask, rope)
        if self.method == "p2n":
            for _ in range(jacobi_iterations):
                state = self._run_layers(
                    core_input + shift_previous(state, input_ids, self.eod_id),
                    self.core_start,
                    self.core_end,
                    mask,
                    rope,
                )
        hidden = self._run_layers(state, self.core_end, len(self.decoder.layers), mask, rope)
        if self.decoder.final_layernorm is not None:
            hidden = self.decoder.final_layernorm(hidden)
        weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden, weight=weight)
        return logits.transpose(0, 1).contiguous()
