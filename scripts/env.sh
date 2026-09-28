#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ztwang/p2n-megatron
DATA=/data3/ztwang/p2n-megatron
ENVROOT=/data3/ztwang/agentic-ttt-env
export PYTHON="$ENVROOT/bin/python"
export TMPDIR="$DATA/tmp" TEMP="$DATA/tmp" TMP="$DATA/tmp"
export XDG_CACHE_HOME="$DATA/cache" HF_HOME="$DATA/cache/huggingface"
export HF_DATASETS_CACHE="$DATA/cache/huggingface/datasets"
export PIP_CACHE_DIR="$DATA/cache/pip"
export PYTHONPYCACHEPREFIX="$DATA/cache/pycache"
export TORCHINDUCTOR_CACHE_DIR="$DATA/cache/torchinductor"
export TRITON_CACHE_DIR="$DATA/cache/triton"
export PYTHONPATH="$REPO:$REPO/vendor/Megatron-LM:$DATA/shim:$ENVROOT/packages:$ENVROOT/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"

cuda_libs=()
for component in cuda_runtime cublas cudnn cufft curand cusolver cusparse nccl nvjitlink cuda_cupti cuda_nvrtc nvtx; do
  path="$ENVROOT/packages/nvidia/$component/lib"
  [[ -d "$path" ]] && cuda_libs+=("$path")
done
cuda_libs+=("$ENVROOT/packages/cusparselt/lib")
cuda_path=$(IFS=:; echo "${cuda_libs[*]}")
export LD_LIBRARY_PATH="$cuda_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$DATA/logs" "$DATA/checkpoints"
