"""Four-rank data-parallel pretraining with Megatron Core's GPT layers.

Launch through torchrun (usually via scripts/smoke_4gpu.sbatch or
scripts/train_4gpu.sbatch). This project uses Megatron Core for the full model
backbone and torch DistributedDataParallel for data parallelism; TP=PP=1.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from megatron.core import parallel_state
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

from p2n.data import TokenStream, synthetic_batch
from p2n.model import (
    PAPER_600M, P2NGPTModel, causal_document_mask, make_transformer_config, shift_previous,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("p2n", "vanilla"), default="p2n")
    parser.add_argument("--train-bin")
    parser.add_argument("--valid-bin")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--allow-wrap", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=57221)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1.5e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--eod-id", type=int, default=50256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--activation-checkpointing", action="store_true")
    args = parser.parse_args()
    if args.seq_len < 2 or args.seq_len > 2048:
        parser.error("seq-len must be between 2 and 2048")
    if args.steps < 1 or args.micro_batch_size < 1:
        parser.error("steps and micro-batch-size must be positive")
    if args.valid_bin and args.eval_batches < 1:
        parser.error("eval-batches must be positive when valid-bin is set")
    if args.smoke and args.train_bin is None:
        parser.error("smoke requires --train-bin; use a real tokenized dataset")
    if not args.smoke and args.train_bin is None:
        parser.error("training requires --train-bin")
    return args


def main():
    args = arguments()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.global_batch_size % (world * args.micro_batch_size):
        raise ValueError("global batch must divide evenly across ranks and microbatches")
    accumulation = args.global_batch_size // (world * args.micro_batch_size)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    torch.manual_seed(args.seed)
    model_parallel_cuda_manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        model = P2NGPTModel(
            make_transformer_config(), method=args.method, eod_id=args.eod_id,
            activation_checkpointing=args.activation_checkpointing,
        ).to(device)
        parameter_count = sum(p.numel() for p in model.parameters())
        if parameter_count != PAPER_600M["parameter_count"]:
            raise AssertionError(
                f"parameter count mismatch: got {parameter_count:,}; "
                f"paper has {PAPER_600M['parameter_count']:,}"
            )
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
        hook_counts = {0: 0, PAPER_600M["core_start"]: 0, PAPER_600M["num_layers"] - 1: 0}
        hooks = []
        if args.smoke and not args.activation_checkpointing:
            for index in hook_counts:
                def count_forward(_module, _inputs, _output, layer_index=index):
                    hook_counts[layer_index] += 1

                hooks.append(model.module.decoder.layers[index].register_forward_hook(count_forward))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
            eps=1e-8, weight_decay=args.weight_decay, fused=True,
        )
        train = TokenStream(
            args.train_bin, vocab_size=PAPER_600M["vocab_size"], allow_wrap=args.allow_wrap
        )
        valid = (
            TokenStream(args.valid_bin, vocab_size=PAPER_600M["vocab_size"])
            if args.valid_bin else None
        )
        start_step = 0
        if args.resume:
            state = torch.load(args.resume, map_location="cpu", weights_only=False)
            if state["method"] != args.method or state["world_size"] != world:
                raise ValueError("checkpoint method or world size does not match")
            for key in (
                "train_bin", "seq_len", "micro_batch_size", "global_batch_size", "eod_id"
            ):
                if state["arguments"].get(key) != getattr(args, key):
                    raise ValueError(f"checkpoint {key} does not match current run")
            model.module.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_step = int(state["step"])
            del state
        if rank == 0:
            print(json.dumps({
                "event": "start", "method": args.method, "params": parameter_count,
                "world_size": world, "seq_len": args.seq_len,
                "micro_batch_size": args.micro_batch_size,
                "global_batch_size": args.global_batch_size,
                "accumulation": accumulation, "train_tokens": len(train.tokens),
                "start_step": start_step,
            }), flush=True)
        _check_boundaries(device)
        for step in range(start_step, args.steps):
            for index in hook_counts:
                hook_counts[index] = 0
            model.train()
            optimizer.zero_grad(set_to_none=True)
            k = (
                2 + (step % 2) if args.smoke else random.Random(args.seed + step).choice((2, 3))
            ) if args.method == "p2n" else 0
            lr = _learning_rate(args, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss_sum = torch.zeros((), device=device)
            for micro in range(accumulation):
                first_token = (
                    step * args.global_batch_size * args.seq_len
                    + micro * world * args.micro_batch_size * args.seq_len
                    + rank * args.micro_batch_size * args.seq_len
                )
                input_ids, labels = train.batch(
                    first_token=first_token,
                    batch_size=args.micro_batch_size,
                    seq_len=args.seq_len,
                    device=device,
                )
                sync = micro == accumulation - 1
                context = contextlib.nullcontext() if sync else model.no_sync()
                with context:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(input_ids, jacobi_iterations=k)
                        loss = F.cross_entropy(
                            logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1)
                        )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"nonfinite loss on rank {rank}, step {step}")
                    (loss / accumulation).backward()
                loss_sum += loss.detach() / accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"nonfinite gradient norm on rank {rank}, step {step}")
            if args.smoke:
                _check_layer_grads(model.module)
                if not args.activation_checkpointing:
                    expected = {
                        0: accumulation,
                        PAPER_600M["core_start"]: (k + 1) * accumulation,
                        PAPER_600M["num_layers"] - 1: accumulation,
                    }
                    if hook_counts != expected:
                        raise AssertionError(f"layer execution counts {hook_counts} != {expected}")
            optimizer.step()
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            loss_sum /= world
            if rank == 0:
                print(json.dumps({
                    "event": "step", "step": step + 1, "jacobi_k": k,
                    "mean_loss": round(float(loss_sum), 6),
                    "grad_norm": round(float(grad_norm), 6),
                    "learning_rate": lr,
                    "layer_calls": hook_counts if args.smoke and not args.activation_checkpointing else None,
                    "gpu_peak_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
                }), flush=True)
            if args.save_every and (step + 1) % args.save_every == 0:
                _save_checkpoint(args, model.module, optimizer, step + 1, rank, world)
            if valid is not None and args.eval_every and (
                (step + 1) % args.eval_every == 0 or step + 1 == args.steps
            ):
                val_loss = _evaluate(
                    model, valid, args, rank=rank, world=world, device=device
                )
                if rank == 0:
                    print(json.dumps({
                        "event": "validation", "step": step + 1,
                        "jacobi_k": 3 if args.method == "p2n" else 0,
                        "mean_loss": round(val_loss, 6),
                        "perplexity": round(math.exp(val_loss), 6),
                    }), flush=True)
        dist.barrier()
        if rank == 0:
            print(json.dumps({"event": "complete", "steps": args.steps}), flush=True)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def _learning_rate(args, step):
    warmup = max(1, math.ceil(args.steps * args.warmup_ratio))
    if step < warmup:
        return args.learning_rate * (step + 1) / warmup
    progress = min(1.0, (step - warmup + 1) / max(1, args.steps - warmup))
    return args.learning_rate * (
        args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    )


def _check_boundaries(device):
    states = torch.arange(5, device=device).float().view(5, 1, 1)
    ids = torch.tensor([[2, 3, 9, 4, 5]], device=device)
    result = shift_previous(states, ids, eod_id=9).flatten().tolist()
    if result != [0.0, 0.0, 1.0, 0.0, 3.0]:
        raise AssertionError(f"boundary shift failed: {result}")
    mask = causal_document_mask(torch.tensor([[1, 9, 2, 3]], device=device), eod_id=9)
    if not (mask[0, 0, 2, 1] and not mask[0, 0, 2, 2] and mask[0, 0, 0, 1]):
        raise AssertionError("causal or packed-document attention boundary failed")


def _check_layer_grads(model):
    for index in (0, PAPER_600M["core_start"], PAPER_600M["num_layers"] - 1):
        params = list(model.decoder.layers[index].parameters())
        if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in params):
            raise AssertionError(f"no finite gradient in layer {index}")


def _save_checkpoint(args, model, optimizer, step, rank, world):
    checkpoint_dir = Path(args.checkpoint_dir)
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"step_{step:07d}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save({
            "step": step, "method": args.method, "world_size": world,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "arguments": vars(args),
        }, temporary)
        os.replace(temporary, path)
        print(json.dumps({"event": "checkpoint", "path": str(path)}), flush=True)
    dist.barrier()


def _evaluate(model, valid, args, *, rank, world, device):
    model.eval()
    loss_sum = torch.zeros((), device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_index in range(args.eval_batches):
            first_token = (
                batch_index * world * args.micro_batch_size * args.seq_len
                + rank * args.micro_batch_size * args.seq_len
            )
            input_ids, labels = valid.batch(
                first_token=first_token, batch_size=args.micro_batch_size,
                seq_len=args.seq_len, device=device,
            )
            logits = model(input_ids, jacobi_iterations=3 if args.method == "p2n" else 0)
            loss_sum += F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1)
            )
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    return float(loss_sum / (world * args.eval_batches))


if __name__ == "__main__":
    main()
