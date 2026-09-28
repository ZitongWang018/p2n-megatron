#!/usr/bin/env bash
#SBATCH --job-name=p2n-150m
#SBATCH --partition=RTX4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00

set -euo pipefail
repo="${P2N_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$repo/scripts/env.sh"
cd "$repo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

method="${METHOD:?Set METHOD to vanilla or p2n}"
case "$method" in vanilla|p2n) ;; *) echo "Invalid METHOD: $method" >&2; exit 2 ;; esac

swanlab_args=()
if [[ "${SWANLAB_ENABLED:-1}" == "1" ]]; then
  secret_file="$P2N_DATA_ROOT/private/swanlab.env"
  [[ -r "$secret_file" ]] || { echo "SwanLab credential file missing" >&2; exit 1; }
  set -a
  source "$secret_file"
  set +a
  : "${SWANLAB_API_KEY:?SwanLab credential is empty}"
  swanlab_args+=(--swanlab --swanlab-workspace ZitongWang --swanlab-project P2N)
fi

verify_args=()
[[ "${VERIFY:-0}" != "1" ]] || verify_args+=(--verify)
train_bin="${TRAIN_BIN:-/data0/hli/mcore_data/uint16smallpile_5btok_train.bin}"
valid_bin="${VALID_BIN:-/data0/hli/mcore_data/uint16smallpile_5btok_valid.bin}"
[[ -r "$train_bin" && -r "$valid_bin" ]] || { echo "Training or validation data missing" >&2; exit 1; }

exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  pretrain_p2n.py --profile qwen3-150m --method "$method" \
  --train-bin "$train_bin" --valid-bin "$valid_bin" \
  --seq-len 2048 --eod-id 0 \
  --micro-batch-size "${MICRO_BATCH_SIZE:-8}" --global-batch-size 1024 \
  --steps "${TRAIN_STEPS:-1427}" --learning-rate 1.5e-3 --weight-decay 0.1 \
  --warmup-ratio 0.05 --min-lr-ratio 0.1 \
  --eval-every "${EVAL_EVERY:-250}" --eval-batches 4 \
  --save-every "${SAVE_EVERY:-250}" \
  --activation-checkpointing "${swanlab_args[@]}" "${verify_args[@]}" \
  --checkpoint-dir "${CHECKPOINT_DIR:-$P2N_DATA_ROOT/checkpoints/qwen3-150m-${method}-bs1024}"
