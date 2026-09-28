"""Evaluate a saved 70M checkpoint over a shared validation subset."""

import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from p2n.data import TokenStream
from p2n.model import P2NGPTModel, make_transformer_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid-bin", required=True)
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--eod-id", type=int, default=0)
    parser.add_argument("--jacobi-ks", default="0,1,2,3,4,5")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    per_batch = world * args.micro_batch_size
    if args.examples % per_batch:
        parser.error("examples must be divisible by world size times micro batch size")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    torch.manual_seed(42)
    model_parallel_cuda_manual_seed(42)
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        method = checkpoint["method"]
        profile = checkpoint["arguments"]["profile"]
        model = P2NGPTModel(
            make_transformer_config(profile=profile), method=method,
            eod_id=args.eod_id, profile=profile,
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        model.eval()
        valid = TokenStream(args.valid_bin, vocab_size=50304)
        ks = [int(x) for x in args.jacobi_ks.split(",")]
        if method == "vanilla" and ks != [0]:
            parser.error("vanilla only supports --jacobi-ks 0")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for k in ks:
                loss_sum = torch.zeros((), device=device)
                for batch in range(args.examples // per_batch):
                    first_token = (batch * per_batch + rank * args.micro_batch_size) * 2048
                    input_ids, labels = valid.batch(
                        first_token=first_token, batch_size=args.micro_batch_size,
                        seq_len=2048, device=device,
                    )
                    logits = model(input_ids, jacobi_iterations=k)
                    loss_sum += F.cross_entropy(
                        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1)
                    )
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                if rank == 0:
                    print(json.dumps({
                        "method": method, "jacobi_k": k, "examples": args.examples,
                        "loss": float(loss_sum / (world * (args.examples // per_batch))),
                    }), flush=True)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
