// SPDX-License-Identifier: Apache-2.0

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <algorithm>
#include <cstdint>

namespace sglang {

/// MiMo's packed BF16 QKV, with partial NeoX RoPE and asymmetric QK/V dimensions.
struct MiMoRopeScaleParams {
  const bf16_t* __restrict__ qkv;
  const bf16_t* __restrict__ cos_sin;
  const int64_t* __restrict__ positions;
  bf16_t* __restrict__ q;
  bf16_t* __restrict__ k;
  bf16_t* __restrict__ v;
  int64_t num_tokens;
  float v_scale;
};

/// Rotate Q/K, scale V, and emit three contiguous tensors in one launch.
template <int64_t kQHeads, int64_t kKVHeads>
__global__ void mimo_rope_scale_qkv_kernel(const __grid_constant__ MiMoRopeScaleParams p) {
  constexpr int64_t kQSize = kQHeads * 192;
  constexpr int64_t kKSize = kKVHeads * 192;
  constexpr int64_t kVSize = kKVHeads * 128;
  constexpr int64_t kPackedSize = kQSize + kKSize + kVSize;
  constexpr int64_t kHeads = kQHeads + 2 * kKVHeads;
  const uint32_t lane = device::get_lane_id();
  const int64_t first = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const int64_t stride = int64_t(gridDim.x) * blockDim.x / 32;

  for (int64_t work = first; work < p.num_tokens * kHeads; work += stride) {
    const int64_t token = work / kHeads;
    const int64_t head = work % kHeads;
    const auto* packed = p.qkv + token * kPackedSize;
    if (head < kQHeads + kKVHeads) {
      const auto* src = packed + head * 192;
      auto* dst = head < kQHeads ? p.q + token * kQSize + head * 192 : p.k + token * kKSize + (head - kQHeads) * 192;
      const auto* cache = p.cos_sin + p.positions[token] * 64;
      const bf16_t x = src[lane];
      const bf16_t y = src[32 + lane];
      const bf16_t cos = cache[lane];
      const bf16_t sin = cache[32 + lane];
      // Keep the same BF16 expression (including compiler contraction) as pos_enc.cuh.
      // Promoting this expression to FP32 changes the existing MiMo fallback.
      dst[lane] = x * cos - y * sin;
      dst[32 + lane] = y * cos + x * sin;
      using Pair = packed_t<bf16_t>;
#pragma unroll
      for (uint32_t offset = 64; offset < 192; offset += 64) {
        const auto pair = device::load_as<Pair>(src + offset, lane);
        device::store_as<Pair>(dst + offset, pair, lane);
      }
    } else {
      const int64_t v_head = head - kQHeads - kKVHeads;
      const auto* src = packed + kQSize + kKSize + v_head * 128;
      auto* dst = p.v + token * kVSize + v_head * 128;
      using Vec = device::AlignedVector<bf16_t, 4>;
      auto values = device::load_as<Vec>(src, lane);
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        values[i] = static_cast<bf16_t>(static_cast<float>(values[i]) * p.v_scale);
      }
      device::store_as<Vec>(dst, values, lane);
    }
  }
}

/// Validate MiMo's BF16 192/128/64 layout and launch the fused preparation kernel.
template <int64_t kQHeads, int64_t kKVHeads>
void mimo_rope_scale_qkv(
    tvm::ffi::TensorView qkv,
    tvm::ffi::TensorView positions,
    tvm::ffi::TensorView cos_sin,
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    float v_scale) {
  using namespace host;
  static_assert(kQHeads > 0 && kKVHeads > 0);
  auto N = SymbolicSize{"num_tokens"};
  auto P = SymbolicSize{"cache_positions"};
  auto device = SymbolicDevice{};
  constexpr int64_t kQSize = kQHeads * 192;
  constexpr int64_t kKSize = kKVHeads * 192;
  constexpr int64_t kVSize = kKVHeads * 128;
  TensorMatcher({N, kQSize + kKSize + kVSize})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device)
      .ensure_alignment(8)
      .verify(qkv);
  TensorMatcher({N, kQSize}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).ensure_alignment(8).verify(q);
  TensorMatcher({N, kKSize}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).ensure_alignment(8).verify(k);
  TensorMatcher({N, kVSize}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).ensure_alignment(8).verify(v);
  TensorMatcher({N}).with_dtype<int64_t>().with_device<kDLCUDA>(device).verify(positions);
  TensorMatcher({P, 64}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(cos_sin);
  CHECK_HOST(P.unwrap() > 0) << "RoPE cache must not be empty";
  if (N.unwrap() == 0) return;

  const MiMoRopeScaleParams params{
      static_cast<const bf16_t*>(qkv.data_ptr()),
      static_cast<const bf16_t*>(cos_sin.data_ptr()),
      static_cast<const int64_t*>(positions.data_ptr()),
      static_cast<bf16_t*>(q.data_ptr()),
      static_cast<bf16_t*>(k.data_ptr()),
      static_cast<bf16_t*>(v.data_ptr()),
      N.unwrap(),
      v_scale};
  constexpr uint32_t kThreads = 128;
  const int64_t work = N.unwrap() * (kQHeads + 2 * kKVHeads);
  const auto blocks =
      std::min<int64_t>(div_ceil(work, int64_t(kThreads / 32)), runtime::get_sm_count(device.unwrap().device_id) * 8);
  LaunchKernel(static_cast<uint32_t>(blocks), kThreads, device.unwrap())(
      mimo_rope_scale_qkv_kernel<kQHeads, kKVHeads>, params);
}

}  // namespace sglang
