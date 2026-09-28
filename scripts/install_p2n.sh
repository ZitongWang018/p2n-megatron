#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${P2N_DATA_ROOT:-/data3/${USER}/p2n-megatron}"
venv="$data_root/venv"
mkdir -p "$data_root"
if [[ -x "$venv/bin/python" ]]; then
  P2N_PYTHON="$venv/bin/python" source "$repo/scripts/env.sh"
  "$PYTHON" -c 'from megatron.core.models.gpt.gpt_model import GPTModel' \
    && { echo "P2N environment ready: $PYTHON"; exit 0; }
fi
legacy_env="/data3/${USER}/agentic-ttt-env"
if [[ -x "$legacy_env/bin/python" && -d "$data_root/shim" ]]; then
  P2N_PYTHON="$legacy_env/bin/python" source "$repo/scripts/env.sh"
  "$PYTHON" -c 'from megatron.core.models.gpt.gpt_model import GPTModel' \
    && { echo "P2N environment ready: $PYTHON"; exit 0; }
fi
python3 -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip wheel 'setuptools<80' pybind11
"$venv/bin/python" -m pip install 'torch==2.6.0' 'numpy<2'
"$venv/bin/python" -m pip install --no-deps -e "$repo"
P2N_PYTHON="$venv/bin/python" source "$repo/scripts/env.sh"
"$venv/bin/python" - <<'PY'
import torch
from megatron.core.models.gpt.gpt_model import GPTModel
print(f"P2N environment ready: PyTorch {torch.__version__}")
PY
