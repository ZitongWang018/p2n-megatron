#!/usr/bin/env bash
#SBATCH --job-name=p2n-600m-smoke
#SBATCH --partition=RTX4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/data3/ztwang/p2n-megatron/logs/smoke-%j.out

set -euo pipefail
source /home/ztwang/p2n-megatron/scripts/env.sh
cd /home/ztwang/p2n-megatron
resume_args=()
if [[ -n "${SMOKE_RESUME:-}" ]]; then
  resume_args+=(--resume "$SMOKE_RESUME")
fi
if [[ "${SMOKE_CHECKPOINTING:-0}" == "1" ]]; then
  resume_args+=(--activation-checkpointing)
fi
if [[ -n "${SMOKE_VALID_BIN:-}" ]]; then
  resume_args+=(--valid-bin "$SMOKE_VALID_BIN" --eval-every 1 --eval-batches 1)
fi
exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 train.py \
  --method "${SMOKE_METHOD:-p2n}" \
  --smoke \
  --train-bin /data0/hli/mcore_data/uint16smallpile_5btok_train.bin \
  --seq-len "${SMOKE_SEQ_LEN:-64}" \
  --micro-batch-size 1 \
  --global-batch-size "${SMOKE_GLOBAL_BATCH_SIZE:-4}" \
  --steps "${SMOKE_STEPS:-2}" \
  --save-every "${SMOKE_SAVE_EVERY:-2}" \
  --checkpoint-dir /data3/ztwang/p2n-megatron/checkpoints/smoke \
  "${resume_args[@]}"
