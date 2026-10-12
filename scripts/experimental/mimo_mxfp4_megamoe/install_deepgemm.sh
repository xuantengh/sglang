#!/usr/bin/env bash
# Experimental SM90 W4A8 dependency, pinned to sgl-project/DeepGEMM#53.
# Run inside a CUDA development container on an allocated Hopper compute node,
# with the target Python environment active (PyTorch, tvm-ffi, libdw/libelf
# development headers, and the toolchain required by DeepGEMM already installed).
# Usage: bash scripts/experimental/mimo_mxfp4_megamoe/install_deepgemm.sh /path/to/new/DeepGEMM
#
# The PR needs BOTH patches: CUDA 13 dependent-template syntax and an async-proxy
# fence between FP4 decode stores and WGMMA reads. Without the fence, the tested
# PR can produce incorrect/nonfinite outputs. Do not install the unpatched PR.
# This creates a fresh checkout and installs into the active Python environment.
# No SGLang dependency version is changed globally while the PR is experimental.
#
# Validated target: XiaomiMiMo/MiMo-V2.6-Flash-RL, 8x H100, TP8/EP8/attention-DP2.
# After installing, the ordinary server entry point selects W4A8 with:
# SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096 \
# python -m sglang.launch_server --model-path /path/to/checkpoint \
#   --tp-size 8 --ep-size 8 --attn-dp-size 2 --mm-enable-dp-encoder \
#   --moe-runner-backend marlin --moe-a2a-backend megamoe \
#   --attention-backend fa3 --chunked-prefill-size 8192 \
#   --cuda-graph-max-bs-decode 128 --trust-remote-code
# The Marlin selector reuses its checkpoint loader; MegaMoE owns execution.
# Hopper decodes MXFP4 to FP8 internally: this is not native FP4 tensor-core MMA.
# W4A4, clamped SwiGLU, and unquantized MTP experts are not supported by this path.
set -euo pipefail

PATCH_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CHECKOUT=${1:?Pass a new directory for the pinned DeepGEMM checkout}
PR_SHA=b5675bf7c812a583d6eadabfa23ad1e00207c39d
if [[ -e "$CHECKOUT" ]]; then
    echo "Refusing to overwrite existing checkout: $CHECKOUT" >&2
    exit 1
fi
python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
    raise SystemExit("Run this build on an allocated SM90 GPU compute node.")
PY

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 CARGO_BUILD_JOBS=1
export FLASHINFER_CUDA_ARCH_LIST=9.0a TORCH_CUDA_ARCH_LIST=9.0a TVM_FFI_CUDA_ARCH_LIST=9.0a
git clone --no-checkout https://github.com/sgl-project/DeepGEMM.git "$CHECKOUT"
git -C "$CHECKOUT" fetch origin refs/pull/53/head
git -C "$CHECKOUT" checkout --detach "$PR_SHA"
git -C "$CHECKOUT" submodule update --init --recursive --jobs 1
git -C "$CHECKOUT" apply "$PATCH_DIR/cuda13-template.patch"
git -C "$CHECKOUT" apply "$PATCH_DIR/decode-proxy-fence.patch"
# The SGLang wheel contains the PR's Python adapters and TVM-FFI _C.so;
# installing the checkout's legacy setup.py builds a different distribution.
bash "$CHECKOUT/build_sgl_deep_gemm.sh"
python -m pip install --no-deps --force-reinstall "$CHECKOUT"/dist/sgl_deep_gemm-*.whl

python - <<'PY'
import hashlib
from pathlib import Path
import deep_gemm

assert callable(deep_gemm.transform_weights_for_mega_moe_sm90_fp4)
header = Path(deep_gemm.__file__).parent / "include/deep_gemm/impls/sm90_fp8_fp4_mega_moe.cuh"
assert hashlib.sha256(header.read_bytes()).hexdigest() == (
    "7d87b558fc4d8b4725cf7e129ace4cb05e004fb2e33143e364e4db546a8df416"
), f"Installed header does not match the validated patches: {header}"
print(f"Installed patched SM90 W4A8 DeepGEMM: {header}")
PY
