#!/usr/bin/env bash
#SBATCH --job-name=p2n-70m
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

method="${METHOD:?Set METHOD to vanilla or p2n}"
case "$method" in vanilla|p2n) ;; *) echo "Invalid METHOD: $method" >&2; exit 2 ;; esac

secret_file="$P2N_DATA_ROOT/private/swanlab.env"
if [[ ! -r "$secret_file" ]]; then
  echo "SwanLab credential file missing: $secret_file" >&2
  exit 1
fi
set -a
source "$secret_file"
set +a
: "${SWANLAB_API_KEY:?SwanLab credential is empty}"
swanlab_args=()
if [[ "${SWANLAB_ENABLED:-1}" == "1" ]]; then
  swanlab_args+=(--swanlab --swanlab-workspace ZitongWang --swanlab-project P2N)
fi
verify_args=()
if [[ "${VERIFY:-0}" == "1" ]]; then
  verify_args+=(--verify)
fi

train_bin="${TRAIN_BIN:-/data0/hli/mcore_data/uint16smallpile_5btok_train.bin}"
valid_bin="${VALID_BIN:-/data0/hli/mcore_data/uint16smallpile_5btok_valid.bin}"
[[ -r "$train_bin" && -r "$valid_bin" ]] || { echo "Training or validation data missing" >&2; exit 1; }

exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  pretrain_p2n.py --profile qwen3-70m --method "$method" \
  --train-bin "$train_bin" --valid-bin "$valid_bin" \
  --seq-len 2048 --eod-id 0 \
  --micro-batch-size 8 --global-batch-size 256 \
  --steps "${TRAIN_STEPS:-2836}" --learning-rate 1.5e-3 --weight-decay 0.01 \
  --warmup-ratio 0.01 --min-lr-ratio 0.1 \
  --eval-every 709 --eval-batches 4 --save-every 709 \
  --activation-checkpointing "${swanlab_args[@]}" "${verify_args[@]}" \
  --checkpoint-dir "$P2N_DATA_ROOT/checkpoints/qwen3-70m-${method}-bs256"
