#!/usr/bin/env bash
#SBATCH --job-name=p2n-audit
#SBATCH --partition=RTX4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/env.sh"
cd "$P2N_REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=1 \
  scripts/audit_p2n.py \
  --checkpoint "${AUDIT_CHECKPOINT:-$P2N_DATA_ROOT/checkpoints/qwen3-70m-p2n-bs256/step_0002836.pt}"
