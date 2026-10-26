import os
import unittest

import torch
import torch.distributed as dist

from sglang.kernels.ops.communication.mp import multigpu_launch
from sglang.srt.distributed.parallel_state import init_world_group
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-c", runner_config="4-gpu-h100")


class TestSM90SequenceParallel(CustomTestCase):
    @torch.inference_mode()
    def test_collectives_graph_fallback_and_reinitialization(self):
        from sglang.srt.distributed.device_communicators.sm90_sequence_parallel import (
            SM90SequenceParallel,
        )

        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            self.skipTest("Requires four SM90 GPUs with multicast")
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group("gloo")
        coord = init_world_group(list(range(4)), rank, "nccl")
        with get_parallel().override(world_group=coord):
            try:
                coord.init_sm90_sp_comm()
                adapter = coord.sm90_sp_comm
                self.assertIsNotNone(adapter)
                for rows in (4, 8, 16, 64, 256, 1024):
                    torch.manual_seed(42 + rank)
                    x = torch.randn(rows, 4096, device=device, dtype=torch.bfloat16)
                    original = x.clone()
                    full = torch.empty_like(x)
                    shard = adapter.try_reduce_scatter(x)
                    self.assertIsNotNone(shard)
                    self.assertFalse(torch._C._overlaps(shard, x))
                    self.assertTrue(adapter.try_all_gather(full, shard))
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(x, original))
                    self.check_result(coord, x, shard, full)

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        shard = adapter.try_reduce_scatter(x)
                        self.assertTrue(adapter.try_all_gather(full, shard))
                    for seed in (43, 44):
                        torch.manual_seed(seed + rank)
                        x.copy_(torch.randn_like(x))
                        graph.replay()
                        torch.cuda.synchronize()
                        self.check_result(coord, x, shard, full)
                    del graph

                for shape, dtype in (
                    ((0, 4096), torch.bfloat16),
                    ((4096, 4096), torch.bfloat16),
                    ((4, 2048), torch.bfloat16),
                    ((4, 4096), torch.float32),
                    ((5, 4096), torch.bfloat16),
                ):
                    x = torch.zeros(shape, device=device, dtype=dtype)
                    self.assertIsNone(adapter.try_reduce_scatter(x))
                x = torch.zeros(4, 8192, device=device, dtype=torch.bfloat16)[:, ::2]
                self.assertIsNone(adapter.try_reduce_scatter(x))
                x = torch.zeros(4, 4096, device=device, dtype=torch.bfloat16)
                self.assertFalse(adapter.try_all_gather(x, x[:1]))
                self.assertTrue(torch.equal(x, torch.zeros_like(x)))

                adapter.close()
                adapter.close()
                coord.sm90_sp_comm = SM90SequenceParallel.create(
                    coord.cpu_group, device
                )
                self.assertIsNotNone(coord.sm90_sp_comm)
                shard = coord.sm90_sp_comm.try_reduce_scatter(x)
                self.assertTrue(torch.equal(shard, torch.zeros_like(shard)))
            finally:
                torch.cuda.synchronize()
                coord.destroy()
                dist.destroy_process_group()

    def check_result(self, coord, x, shard, full):
        reference = x.float()
        dist.all_reduce(reference, group=coord.device_group)
        reference = reference.chunk(4)[coord.rank_in_group].to(torch.bfloat16)
        torch.testing.assert_close(shard, reference, atol=0.04, rtol=0.015)
        relative_l2 = (
            shard.float() - reference.float()
        ).norm() / reference.float().norm()
        self.assertLess(relative_l2.item(), 0.01)
        expected_full = torch.empty_like(full)
        dist.all_gather_into_tensor(expected_full, shard, group=coord.device_group)
        self.assertTrue(torch.equal(full, expected_full))


if __name__ == "__main__":
    multigpu_launch(
        __name__,
        __file__,
        [4],
        env_key="_IS_TEST_MIMO_SM90_SP",
        inner=lambda: int(not unittest.main(exit=False).result.wasSuccessful()),
        kind="test",
        timeout=600,
    )
