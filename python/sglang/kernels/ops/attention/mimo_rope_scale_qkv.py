from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_mimo_rope_scale_qkv(num_q_heads: int, num_kv_heads: int) -> Module:
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("Q and KV head counts must be positive")
    args = make_cpp_args(num_q_heads, num_kv_heads)
    return load_jit(
        "mimo_rope_scale_qkv",
        *args,
        cuda_files=["attention/mimo_rope_scale_qkv.cuh"],
        cuda_wrappers=[("run", f"mimo_rope_scale_qkv<{args}>")],
    )


@register_custom_op(mutates_args=["q", "k", "v"])
def _mimo_rope_scale_qkv_out(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    v_scale: float,
) -> None:
    _jit_mimo_rope_scale_qkv(num_q_heads, num_kv_heads).run(
        qkv, positions, cos_sin_cache, q, k, v, v_scale
    )


def mimo_rope_scale_qkv(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    v_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare contiguous MiMo Q/K/V without modifying the packed input.

    CUDA BF16 only: QK head dimension 192, V dimension 128, NeoX rotary
    dimension 64. The BF16 cos/sin cache and intermediate rounding match
    MiMo's existing CUDA RoPE fallback. Positions are contiguous int64 and
    must index valid cache rows. KV cache addressing remains with the backend.
    """
    n = qkv.shape[0]
    q = qkv.new_empty((n, num_q_heads * 192))
    k = qkv.new_empty((n, num_kv_heads * 192))
    v = qkv.new_empty((n, num_kv_heads * 128))
    _mimo_rope_scale_qkv_out(
        qkv, positions, cos_sin_cache, q, k, v, num_q_heads, num_kv_heads, v_scale
    )
    return q, k, v
