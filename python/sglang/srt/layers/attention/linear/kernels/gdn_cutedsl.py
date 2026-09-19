"""CuTe DSL kernels for GDN (Gated Delta Network) linear attention.

Decode uses the existing ``cutedsl_fused_sigmoid_gating_delta_rule_update``.
Prefill uses an architecture-specific chunkwise kernel: WGMMA/TMA with an FP32
register-resident state on SM90, and the tcgen05/TMEM path on SM100+.
"""

import logging
from collections.abc import Callable
from typing import Optional

import torch

from sglang.kernels.ops.attention.cutedsl_gdn import (
    cutedsl_fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

logger = logging.getLogger(__name__)


def _cuda_major() -> int:
    """Return the active CUDA device's compute-capability major."""
    if not torch.cuda.is_available():
        return -1
    major, _ = torch.cuda.get_device_capability()
    return major


class CuteDSLGDNKernel(LinearAttnKernelBase):
    """CuTe DSL kernel for GDN.

    Decode: ``cutedsl_fused_sigmoid_gating_delta_rule_update`` (SM90+).
    Extend (prefill): Hopper WGMMA/TMA or Blackwell tcgen05/TMEM chunkwise
    kernels. Both paths currently require ``head_k_dim == head_v_dim == 128``.
    """

    def __init__(self):
        self._sm_major = _cuda_major()
        self.supports_prefill = self._sm_major >= 9
        # The SM90 kernel can emit the state checkpoints consumed by radix
        # tracking.  The older local SM100 split kernel does not expose them.
        self.uses_state_checkpoints = self._sm_major == 9

        # Heavy CuteDSL imports are deferred to extend() so SM90 boxes can
        # still construct the kernel just for decode.
        self._extend_fn: Optional[Callable] = None
        self._prepare_meta_fn: Optional[Callable] = None
        self._prepare_qkv_fn: Optional[Callable] = None

    def _ensure_extend_loaded(self, head_k_dim: int) -> None:
        if self._extend_fn is not None:
            return
        if not self.supports_prefill:
            raise RuntimeError(
                f"CuTe DSL GDN prefill requires SM90+; got SM{self._sm_major}."
            )
        if head_k_dim != 128:
            raise RuntimeError(
                f"CuTe DSL GDN prefill requires head_k_dim=128, got {head_k_dim}."
            )
        from sglang.kernels.ops.attention.fla.l2norm import (
            gdn_prefill_qkv_prepare_fwd,
        )

        self._prepare_qkv_fn = gdn_prefill_qkv_prepare_fwd
        if self._sm_major == 9:
            from sglang.kernels.ops.attention.linear.gdn_hopper import (
                delta_rule_prefill_dsl_sm90,
            )

            self._extend_fn = delta_rule_prefill_dsl_sm90
            logger.info("Using local CuTe DSL GDN prefill (Hopper WGMMA)")
        else:
            from sglang.kernels.ops.attention.linear.gdn_blackwell import (
                chunk_gated_delta_rule_cutedsl,
                prepare_metadata_cutedsl,
            )

            self._extend_fn = chunk_gated_delta_rule_cutedsl
            self._prepare_meta_fn = prepare_metadata_cutedsl
            logger.info("Using local CuTe DSL GDN prefill (Blackwell tcgen05)")

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return cutedsl_fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        head_k_dim = k.shape[-1]
        self._ensure_extend_loaded(head_k_dim)
        if q.shape[-1] != head_k_dim or v.shape[-1] != head_k_dim:
            raise RuntimeError(
                "CuTe DSL GDN prefill requires equal Q/K/V head dimensions, "
                f"got {q.shape[-1]}, {head_k_dim}, {v.shape[-1]}."
            )

        total_seq_len = q.shape[1]

        # Normalize Q/K once and materialize the token-major TMA layout.  The
        # helper fuses Q/K/V materialization for strided projection views.
        q_norm, k_norm, v_in = self._prepare_qkv_fn(q[0], k[0], v[0])

        # Remap padding (-1) to the last (sentinel) state-pool slot.
        ssm_cache_indices = torch.where(
            cache_indices >= 0,
            cache_indices,
            ssm_states.shape[0] - 1,
        )

        if self._sm_major == 9:
            return self._extend_hopper(
                q_norm,
                k_norm,
                v_in,
                g,
                beta,
                ssm_states,
                ssm_cache_indices,
                query_start_loc,
                kwargs,
            )

        # The Blackwell h kernel reads/writes the indexed pool rows directly,
        # avoiding explicit [N, Hv, V, K] gather/scatter intermediates.
        cu_seqlens = query_start_loc.to(torch.int32)
        chunk_indices, chunk_offsets = self._prepare_meta_fn(
            cu_seqlens, total_seq_len, chunk_size=64
        )
        output, _ = self._extend_fn(
            q=q_norm.unsqueeze(0),
            k=k_norm.unsqueeze(0),
            v=v_in.unsqueeze(0),
            g=g[0].to(torch.float32).unsqueeze(0),
            beta=beta[0].to(torch.float32).unsqueeze(0),
            initial_state=ssm_states,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            initial_state_indices=ssm_cache_indices.to(torch.int32),
        )
        return output, None, None

    def _extend_hopper(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        kwargs: dict,
    ) -> tuple:
        """Run the local SM90 kernel and scatter its FP32 state back to the pool."""
        total_seq_len, _, head_v_dim = v.shape
        indices = cache_indices.to(torch.int64)
        initial_state = ssm_states[indices].to(torch.float32).contiguous()
        output_state = torch.empty_like(initial_state)
        cu_seqlens = query_start_loc.to(torch.int64)
        log_gate = g[0].to(torch.float32).contiguous()
        beta_in = beta[0].to(torch.float32).contiguous()

        output = kwargs.get("output")
        if output is None:
            output_3d = torch.empty_like(v)
        else:
            expected = (1, total_seq_len, v.shape[1], head_v_dim)
            if tuple(output.shape) != expected:
                raise ValueError(
                    f"CuTe DSL GDN prefill output shape must be {expected}, "
                    f"got {tuple(output.shape)}"
                )
            if output.dtype != v.dtype or output.device != v.device:
                raise ValueError(
                    "CuTe DSL GDN prefill output must match v dtype/device"
                )
            output_3d = output[0]
            if not output_3d.is_contiguous():
                raise ValueError("CuTe DSL GDN prefill output must be contiguous")

        num_checkpoints = int(kwargs.get("num_state_checkpoints", 0) or 0)
        checkpoint_every = int(kwargs.get("state_checkpoint_every_n_tokens", 0) or 0)
        checkpoint_starts = kwargs.get("state_checkpoint_cu_starts")
        checkpoints = (
            initial_state.new_empty((num_checkpoints, *initial_state.shape[1:]))
            if num_checkpoints > 0
            else None
        )
        if checkpoints is not None:
            if checkpoint_every <= 0 or checkpoint_starts is None:
                raise ValueError(
                    "GDN state checkpoints require a positive cadence and "
                    "state_checkpoint_cu_starts"
                )
            checkpoint_starts = checkpoint_starts.to(torch.int64)
        else:
            # A tracking plan may legitimately contain no full checkpoint
            # interval (for example, every sequence is shorter than 64).
            checkpoint_every = 0
            checkpoint_starts = None

        self._extend_fn(
            output_3d,
            output_state,
            q,
            k,
            v,
            initial_state,
            log_gate,
            beta_in,
            cu_seqlens,
            q.shape[-1] ** -0.5,
            state_checkpoints=checkpoints,
            checkpoint_cu_starts=checkpoint_starts,
            checkpoint_every_n_tokens=checkpoint_every,
        )
        ssm_states.index_copy_(0, indices, output_state.to(ssm_states.dtype))
        h = checkpoints.unsqueeze(0) if checkpoints is not None else None
        return output_3d.unsqueeze(0), None, h

    def target_verify(self, *args, **kwargs):
        raise NotImplementedError("CuteDSLGDNKernel does not support target_verify")
