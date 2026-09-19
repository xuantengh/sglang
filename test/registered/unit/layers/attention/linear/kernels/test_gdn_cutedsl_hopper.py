import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import CuteDSLGDNKernel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCuteDSLGDNHopperPolicy(CustomTestCase):
    def make_kernel(self, capability):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=capability),
        ):
            return CuteDSLGDNKernel()

    def test_sm90_enables_prefill_and_checkpoints(self):
        kernel = self.make_kernel((9, 0))
        self.assertTrue(kernel.supports_prefill)
        self.assertTrue(kernel.uses_state_checkpoints)

    def test_pre_hopper_keeps_prefill_disabled(self):
        kernel = self.make_kernel((8, 0))
        self.assertFalse(kernel.supports_prefill)
        self.assertFalse(kernel.uses_state_checkpoints)

    def test_blackwell_keeps_prefill_without_hopper_checkpoints(self):
        kernel = self.make_kernel((10, 0))
        self.assertTrue(kernel.supports_prefill)
        self.assertFalse(kernel.uses_state_checkpoints)

    def test_hopper_wrapper_keeps_gate_in_log_space_and_scatters_state(self):
        kernel = self.make_kernel((9, 0))
        seen = {}

        def fake_kernel(
            output,
            output_state,
            q,
            k,
            v,
            initial_state,
            log_gate,
            beta,
            cu_seqlens,
            scale,
            **kwargs,
        ):
            seen["log_gate"] = log_gate.clone()
            seen["initial_state"] = initial_state.clone()
            seen["cu_dtype"] = cu_seqlens.dtype
            seen["scale"] = scale
            output.fill_(3)
            output_state.copy_(initial_state + 2)

        kernel._extend_fn = fake_kernel
        q = torch.ones(4, 1, 2)
        k = torch.ones_like(q)
        v = torch.ones_like(q)
        g = -torch.arange(4, dtype=torch.float32).view(1, 4, 1)
        beta = torch.full((1, 4, 1), 0.25)
        state = torch.arange(12, dtype=torch.float32).view(3, 1, 2, 2)
        original = state.clone()
        indices = torch.tensor([2, 0], dtype=torch.int32)
        cu = torch.tensor([0, 2, 4], dtype=torch.int32)

        output, final_state, checkpoints = kernel._extend_hopper(
            q, k, v, g, beta, state, indices, cu, {}
        )

        torch.testing.assert_close(seen["log_gate"], g[0])
        torch.testing.assert_close(seen["initial_state"], original[indices.long()])
        self.assertEqual(seen["cu_dtype"], torch.int64)
        self.assertAlmostEqual(seen["scale"], 2**-0.5)
        torch.testing.assert_close(output, torch.full((1, 4, 1, 2), 3.0))
        torch.testing.assert_close(state[indices.long()], original[indices.long()] + 2)
        self.assertIsNone(final_state)
        self.assertIsNone(checkpoints)

    def test_hopper_wrapper_disables_an_empty_checkpoint_plan(self):
        kernel = self.make_kernel((9, 0))
        seen = {}

        def fake_kernel(output, output_state, *args, **kwargs):
            seen.update(kwargs)
            output.zero_()
            output_state.copy_(args[3])

        kernel._extend_fn = fake_kernel
        q = torch.ones(2, 1, 2)
        state = torch.ones(2, 1, 2, 2)
        kernel._extend_hopper(
            q,
            q,
            q,
            torch.zeros(1, 2, 1),
            torch.ones(1, 2, 1),
            state,
            torch.tensor([0]),
            torch.tensor([0, 2]),
            {
                "num_state_checkpoints": 0,
                "state_checkpoint_every_n_tokens": 64,
                "state_checkpoint_cu_starts": torch.tensor([0, 0]),
            },
        )

        self.assertIsNone(seen["state_checkpoints"])
        self.assertIsNone(seen["checkpoint_cu_starts"])
        self.assertEqual(seen["checkpoint_every_n_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
