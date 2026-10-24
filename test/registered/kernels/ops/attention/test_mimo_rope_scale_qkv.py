import unittest

import torch

from sglang.kernels.ops.attention.mimo_rope_scale_qkv import mimo_rope_scale_qkv
from sglang.kernels.ops.attention.rope import rotary_embedding
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestMiMoRopeScaleQKV(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        angles = torch.randn(8192, 32, device="cuda")
        cls.cache = torch.cat((angles.cos(), angles.sin()), dim=-1).bfloat16()

    def inputs(self, tokens, kv_heads):
        qkv = torch.randn(
            tokens, 16 * 192 + kv_heads * 320, device="cuda", dtype=torch.bfloat16
        )
        positions = torch.randint(0, self.cache.shape[0], (tokens,), device="cuda")
        return qkv, positions

    def reference(self, qkv, positions, kv_heads, scale):
        q, k, v = qkv.clone().split([16 * 192, kv_heads * 192, kv_heads * 128], -1)
        rotary_embedding(positions, q, k, 192, self.cache, True)
        return q, k, v * scale

    def check_outputs(self, actual, expected):
        for output, reference in zip(actual, expected):
            self.assertTrue(output.is_contiguous())
            torch.testing.assert_close(output, reference, rtol=0, atol=0)

    def test_fallback_parity(self):
        """Preserve partial RoPE, BF16 intermediate rounding, and asymmetric V."""
        for tokens, kv_heads in [(0, 1), (1, 1), (17, 2), (128, 1), (4097, 2)]:
            for scale in (1.0, 0.707):
                with self.subTest(tokens=tokens, kv_heads=kv_heads, scale=scale):
                    qkv, positions = self.inputs(tokens, kv_heads)
                    original = qkv.clone()
                    actual = mimo_rope_scale_qkv(
                        qkv, positions, self.cache, 16, kv_heads, scale
                    )
                    if tokens:
                        expected = self.reference(qkv, positions, kv_heads, scale)
                    else:
                        expected = qkv.split(
                            [16 * 192, kv_heads * 192, kv_heads * 128], -1
                        )
                    self.check_outputs(actual, expected)
                    torch.testing.assert_close(qkv, original, rtol=0, atol=0)
        torch.cuda.synchronize()

    def test_graph_replay_with_new_inputs(self):
        """Captured kernels must read the updated positions and packed QKV."""
        qkv, positions = self.inputs(19, 2)
        mimo_rope_scale_qkv(qkv, positions, self.cache, 16, 2, 0.707)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = mimo_rope_scale_qkv(qkv, positions, self.cache, 16, 2, 0.707)
        for _ in range(2):
            new_qkv, new_positions = self.inputs(19, 2)
            qkv.copy_(new_qkv)
            positions.copy_(new_positions)
            graph.replay()
            self.check_outputs(outputs, self.reference(qkv, positions, 2, 0.707))


if __name__ == "__main__":
    unittest.main()
