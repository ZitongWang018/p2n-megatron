#!/usr/bin/env bash
#SBATCH --job-name=p2n-70m-eval
#SBATCH --partition=RTX4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00

set -euo pipefail
repo="${P2N_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$repo/scripts/env.sh"
cd "$repo"
valid=/data0/hli/mcore_data/uint16smallpile_5btok_valid.bin
gpus="${EVAL_GPUS:-8}"
examples="${EVAL_EXAMPLES:-4096}"
for method in vanilla p2n; do
  args=(--checkpoint "$P2N_DATA_ROOT/checkpoints/qwen3-70m-${method}-bs256/step_0002836.pt"
        --valid-bin "$valid" --examples "$examples" --micro-batch-size 8 --eod-id 0)
  if [[ "$method" == vanilla ]]; then
    args+=(--jacobi-ks 0)
  elif [[ -n "${EVAL_KS:-}" ]]; then
    args+=(--jacobi-ks "$EVAL_KS")
  fi
  "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$gpus" \
    scripts/eval_70m.py "${args[@]}"
done
