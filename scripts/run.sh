#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo/scripts/env.sh"
cd "$P2N_REPO_ROOT"
mode="${1:-quickstart}"
extra_args=()

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
    if [[ -n "${QUICKSTART_RESUME:-}" ]]; then
      extra_args+=(--resume "$QUICKSTART_RESUME")
    fi
    if [[ "${QUICKSTART_CHECKPOINTING:-0}" == "1" ]]; then
      extra_args+=(--activation-checkpointing)
    fi
    if [[ -n "${QUICKSTART_VALID_BIN:-}" ]]; then
      extra_args+=(--valid-bin "$QUICKSTART_VALID_BIN" --eval-every 1 --eval-batches 1)
    fi
    exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
      pretrain_p2n.py --method "${QUICKSTART_METHOD:-p2n}" --verify \
      --train-bin "$P2N_TRAIN_BIN" --seq-len "${QUICKSTART_SEQ_LEN:-64}" \
      --micro-batch-size 1 --global-batch-size "${QUICKSTART_GLOBAL_BATCH_SIZE:-4}" \
      --steps "${QUICKSTART_STEPS:-2}" --save-every "${QUICKSTART_SAVE_EVERY:-2}" \
      --checkpoint-dir "$P2N_DATA_ROOT/checkpoints/quickstart" "${extra_args[@]}"
    ;;
  train)
    : "${TRAIN_BIN:?Set TRAIN_BIN to your uint16 .bin training file}"
    if [[ "${TRAIN_CHECKPOINTING:-1}" == "1" ]]; then
      extra_args+=(--activation-checkpointing)
    fi
    if [[ -n "${VALID_BIN:-}" ]]; then
      extra_args+=(--valid-bin "$VALID_BIN")
    fi
    if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
      extra_args+=(--resume "$RESUME_CHECKPOINT")
    fi
    exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
      pretrain_p2n.py --method "${METHOD:-p2n}" --train-bin "$TRAIN_BIN" \
      --seq-len 2048 --micro-batch-size 1 --global-batch-size 1024 \
      --steps 57221 --eod-id "${EOD_ID:-50256}" --save-every 500 \
      --checkpoint-dir "$P2N_DATA_ROOT/checkpoints/${METHOD:-p2n}" "${extra_args[@]}"
    ;;
  *)
    echo "usage: bash scripts/run.sh [quickstart|train]" >&2
    exit 2
    ;;
esac
