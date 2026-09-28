#!/usr/bin/env bash
#SBATCH --job-name=p2n-quickstart
#SBATCH --partition=RTX4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:45:00

set -euo pipefail
source "${P2N_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"
cd "$P2N_REPO_ROOT"
resume_args=()
if [[ -n "${QUICKSTART_RESUME:-}" ]]; then
  resume_args+=(--resume "$QUICKSTART_RESUME")
fi
if [[ "${QUICKSTART_CHECKPOINTING:-0}" == "1" ]]; then
  resume_args+=(--activation-checkpointing)
fi
if [[ -n "${QUICKSTART_VALID_BIN:-}" ]]; then
  resume_args+=(--valid-bin "$QUICKSTART_VALID_BIN" --eval-every 1 --eval-batches 1)
fi
exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 pretrain_p2n.py \
  --method "${QUICKSTART_METHOD:-p2n}" \
  --verify \
  --train-bin "$P2N_TRAIN_BIN" \
  --seq-len "${QUICKSTART_SEQ_LEN:-64}" \
  --micro-batch-size 1 \
  --global-batch-size "${QUICKSTART_GLOBAL_BATCH_SIZE:-4}" \
  --steps "${QUICKSTART_STEPS:-2}" \
  --save-every "${QUICKSTART_SAVE_EVERY:-2}" \
  --checkpoint-dir "$P2N_DATA_ROOT/checkpoints/quickstart" \
  "${resume_args[@]}"
