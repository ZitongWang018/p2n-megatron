#!/usr/bin/env bash
set -euo pipefail

export P2N_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export P2N_DATA_ROOT="${P2N_DATA_ROOT:-/data3/${USER}/p2n-megatron}"
export P2N_PYTHON="${P2N_PYTHON:-$P2N_DATA_ROOT/venv/bin/python}"

# Lumia's existing environment is supported while a clean venv is prepared.
legacy_env="/data3/${USER}/agentic-ttt-env"
if [[ ! -x "$P2N_PYTHON" && -x "$legacy_env/bin/python" ]]; then
  P2N_PYTHON="$legacy_env/bin/python"
fi
if [[ ! -x "$P2N_PYTHON" ]]; then
  echo "Python environment missing. Run bash scripts/install_p2n.sh" >&2
  return 1 2>/dev/null || exit 1
fi

export PYTHON="$P2N_PYTHON"
export TMPDIR="$P2N_DATA_ROOT/tmp" TEMP="$P2N_DATA_ROOT/tmp" TMP="$P2N_DATA_ROOT/tmp"
export XDG_CACHE_HOME="$P2N_DATA_ROOT/cache" HF_HOME="$P2N_DATA_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$P2N_DATA_ROOT/cache/huggingface/datasets"
export PIP_CACHE_DIR="$P2N_DATA_ROOT/cache/pip"
export PYTHONPYCACHEPREFIX="$P2N_DATA_ROOT/cache/pycache"
export TORCHINDUCTOR_CACHE_DIR="$P2N_DATA_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$P2N_DATA_ROOT/cache/triton"
export PYTHONPATH="$P2N_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$PYTHON" == "$legacy_env/bin/python" ]]; then
  export PYTHONPATH="$P2N_REPO_ROOT:$P2N_DATA_ROOT/shim:$legacy_env/packages:$legacy_env/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
  cuda_libs=()
  for component in cuda_runtime cublas cudnn cufft curand cusolver cusparse nccl nvjitlink cuda_cupti cuda_nvrtc nvtx; do
    path="$legacy_env/packages/nvidia/$component/lib"
    [[ -d "$path" ]] && cuda_libs+=("$path")
  done
  cuda_libs+=("$legacy_env/packages/cusparselt/lib")
  cuda_path=$(IFS=:; echo "${cuda_libs[*]}")
  export LD_LIBRARY_PATH="$cuda_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" \
  "$PIP_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" "$P2N_DATA_ROOT/logs" "$P2N_DATA_ROOT/checkpoints" \
  "$P2N_DATA_ROOT/data"
