#!/usr/bin/env bash
#SBATCH --job-name=p2n-pretrain
#SBATCH --partition=RTX4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00

set -euo pipefail
repo="${P2N_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$repo/scripts/run.sh" train
