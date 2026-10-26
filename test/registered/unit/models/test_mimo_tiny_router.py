import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.models.mimo_v2 import MoEGate
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestMiMoTinyRouter(CustomTestCase):
    @torch.inference_mode()
    def test_logits_routing_graph_and_fallback(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            self.skipTest("MiMo tiny router is enabled only on SM90")
        config = SimpleNamespace(
            moe_router_dtype="bfloat16",
            n_routed_experts=256,
            hidden_size=4096,
            topk_method="greedy",
        )
        with envs.SGLANG_OPT_MIMO_TINY_ROUTER.override(True):
            gate = MoEGate(config, None).cuda()
        self.assertTrue(gate._use_tiny_router)
        torch.manual_seed(42)
        gate.weight.normal_(std=0.02)
        for rows in (0, 1, 4, 16, 17):
            x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
            self.check_result(gate, x, gate(x))
            if rows in (1, 4, 16):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = gate(x)
                for _ in range(2):
                    x.copy_(torch.randn_like(x))
                    graph.replay()
                    self.check_result(gate, x, output)
                    torch.testing.assert_close(output, gate(x), atol=0, rtol=0)
        x = torch.randn(4, 8192, device="cuda", dtype=torch.bfloat16)[:, ::2]
        self.check_result(gate, x, gate(x))
        with (
            envs.SGLANG_OPT_MIMO_TINY_ROUTER.override(True),
            envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(True),
        ):
            deterministic_gate = MoEGate(config, None).cuda()
        deterministic_gate.weight.copy_(gate.weight)
        self.assertFalse(deterministic_gate._use_tiny_router)
        self.check_result(deterministic_gate, x, deterministic_gate(x))

    def check_result(self, gate, x, output):
        expected = torch.mm(x, gate.weight.t(), out_dtype=torch.float32)
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-4)
        if x.shape[0]:
            relative_l2 = (output - expected).norm() / expected.norm()
            self.assertLess(relative_l2.item(), 1e-5)
            self.assertTrue(
                torch.equal(
                    output.topk(8).indices.sort().values,
                    expected.topk(8).indices.sort().values,
                )
            )


if __name__ == "__main__":
    unittest.main()
