#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo/scripts/env.sh"
mode="${1:-quickstart}"

case "$mode" in
  quickstart)
    if [[ -z "${P2N_TRAIN_BIN:-}" ]]; then
      shared=/data0/hli/mcore_data/uint16smallpile_5btok_train.bin
      if [[ -r "$shared" ]]; then
        P2N_TRAIN_BIN="$shared"
      else
        P2N_TRAIN_BIN="$P2N_DATA_ROOT/data/quickstart.bin"
        if [[ ! -s "$P2N_TRAIN_BIN" ]]; then
          "$PYTHON" "$repo/scripts/prepare_example_data.py" "$P2N_TRAIN_BIN"
        fi
      fi
    fi
    export P2N_TRAIN_BIN
    sbatch --output="$P2N_DATA_ROOT/logs/%x-%j.out" \
      "$repo/scripts/quickstart_4gpu.sbatch"
    ;;
  train)
    : "${TRAIN_BIN:?Set TRAIN_BIN to your uint16 .bin training file}"
    sbatch --output="$P2N_DATA_ROOT/logs/%x-%j.out" \
      "$repo/scripts/train_4gpu.sbatch"
    ;;
  *)
    echo "usage: bash scripts/submit.sh [quickstart|train]" >&2
    exit 2
    ;;
esac
