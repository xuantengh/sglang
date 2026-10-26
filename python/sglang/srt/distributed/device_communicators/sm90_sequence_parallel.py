"""Bounded TP4 push collectives for MiMo's SM90 attention boundaries."""

import logging
from typing import Optional

import torch
import torch.distributed as dist

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class SM90SequenceParallel:
    """Own the workspace for BF16 all-gather and out-of-place reduce-scatter.

    The launch sizes were measured for hidden size 4096 on four H100 GPUs.
    Unsupported shapes return to the caller's ordinary collective path.
    """

    @classmethod
    def create(cls, group: dist.ProcessGroup, device: torch.device):
        if (
            device.type != "cuda"
            or torch.cuda.get_device_capability(device)[0] != 9
            or dist.get_world_size(group) != 4
            or envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
        ):
            return None

        from sglang.kernels.ops.communication import sp_collective
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        comm = CustomAllReduceV2(
            group,
            device,
            max_push_size=8 * 1024 * 1024,
            max_push_blocks=65,
            max_pull_size=0,
            max_pull_blocks=0,
        )
        if (
            comm.disabled
            or not comm.has_multicast
            or comm.max_push_size < 8 * 1024 * 1024
            or comm.config.num_push_blocks < 65
        ):
            comm.close()
            comm.disabled = True
            logger.info("SM90 sequence-parallel workspace unavailable; using NCCL")
            return None
        try:
            sp_collective.register_comm(comm.obj)
        except AssertionError:
            # Another owner already uses the world-size-keyed kernel registry.
            comm.close()
            comm.disabled = True
            logger.info("SM90 sequence-parallel communicator already owned; using NCCL")
            return None
        logger.info("Using SM90 TP4 sequence-parallel collectives for MiMo")
        return cls(comm)

    def __init__(self, comm):
        self.comm = comm

    def _blocks(
        self, tensor: torch.Tensor, rows: int, *, gather: bool
    ) -> Optional[int]:
        if (
            self.comm is None
            or not tensor.is_cuda
            or tensor.device != self.comm.device
            or tensor.dtype != torch.bfloat16
            or tensor.ndim != 2
            or tensor.shape[1] != 4096
            or not tensor.is_contiguous()
            or rows < 4
            or rows >= 4096
            or rows % 4
        ):
            return None
        if gather:
            return 16 if rows < 8 else 32 if rows < 16 else 64
        return 32 if rows < 64 else 64

    def try_all_gather(self, output: torch.Tensor, tensor: torch.Tensor) -> bool:
        if tensor.ndim != 2:
            return False
        rows = tensor.shape[0] * 4
        blocks = self._blocks(tensor, rows, gather=True)
        if (
            blocks is None
            or output.shape != (rows, 4096)
            or output.dtype != tensor.dtype
            or output.device != tensor.device
            or not output.is_contiguous()
            or torch._C._overlaps(output, tensor)
        ):
            return False
        from sglang.kernels.ops.communication import sp_collective

        sp_collective.all_gather(
            4, tensor, output, tuning=sp_collective.Tuning(blocks, 256)
        )
        return True

    def try_reduce_scatter(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if tensor.ndim != 2:
            return None
        blocks = self._blocks(tensor, tensor.shape[0], gather=False)
        if blocks is None:
            return None
        from sglang.kernels.ops.communication import sp_collective

        # An input shard cannot serve as the output: other CTAs may still read it.
        output = torch.empty(
            (tensor.shape[0] // 4, 4096), dtype=tensor.dtype, device=tensor.device
        )
        return sp_collective.reduce_scatter_res(
            4, tensor, output, tuning=sp_collective.Tuning(blocks, 256)
        )

    def close(self) -> None:
        if self.comm is None:
            return
        from sglang.kernels.ops.communication import sp_collective

        comm, self.comm = self.comm, None
        sp_collective.unregister_comm(comm.obj)
        comm.close()
        comm.disabled = True
