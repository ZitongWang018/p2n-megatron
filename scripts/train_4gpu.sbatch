#!/usr/bin/env bash
#SBATCH --job-name=p2n-600m
#SBATCH --partition=RTX4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00

set -euo pipefail
source "${P2N_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"
: "${TRAIN_BIN:?Set TRAIN_BIN to the 120B-token Pile Megatron uint16 .bin file}"
cd "$P2N_REPO_ROOT"
extra_args=()
if [[ "${TRAIN_CHECKPOINTING:-1}" == "1" ]]; then
  extra_args+=(--activation-checkpointing)
fi
if [[ -n "${VALID_BIN:-}" ]]; then
  extra_args+=(--valid-bin "$VALID_BIN")
fi
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  extra_args+=(--resume "$RESUME_CHECKPOINT")
fi
exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 pretrain_p2n.py \
  --method "${METHOD:-p2n}" \
  --train-bin "$TRAIN_BIN" \
  --seq-len 2048 \
  --micro-batch-size 1 \
  --global-batch-size 1024 \
  --steps 57221 \
  --eod-id "${EOD_ID:-50256}" \
  --save-every 500 \
  --checkpoint-dir "$P2N_DATA_ROOT/checkpoints/${METHOD:-p2n}" \
  "${extra_args[@]}"
