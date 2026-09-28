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
repo="${P2N_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$repo/scripts/run.sh" quickstart
