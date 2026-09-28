#!/usr/bin/env bash
set -euo pipefail

repo=/home/ztwang/p2n-megatron
vendor="$repo/vendor/Megatron-LM"
expected=c550cf6c41c31cd3ec72e05c25ea0c979f2b6631

if [[ ! -d "$vendor/.git" ]]; then
  mkdir -p "$repo/vendor"
  git clone --depth 1 --branch core_v0.13.0 \
    https://github.com/NVIDIA/Megatron-LM.git "$vendor"
fi

actual=$(git -C "$vendor" rev-parse HEAD)
if [[ "$actual" != "$expected" ]]; then
  echo "Megatron commit mismatch: expected $expected, got $actual" >&2
  exit 1
fi

source "$repo/scripts/env.sh"
"$PYTHON" - <<'PY'
import numpy
import torch
from megatron.core.models.gpt.gpt_model import GPTModel

assert torch.__version__.startswith("2.6.0"), torch.__version__
print(f"Megatron ready; torch={torch.__version__}, numpy={numpy.__version__}")
PY

test -r /data0/hli/mcore_data/uint16smallpile_5btok_train.bin || {
  echo "Smoke dataset is missing or unreadable" >&2
  exit 1
}
