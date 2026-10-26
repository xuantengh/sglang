import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo, ScheduleBatch
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Allocator:
    def __init__(self):
        self.live = set(range(1024))

    def free_swa_segment(self, slots, *, start_pos):
        for slot in slots.tolist():
            self.live.remove(slot)


class _SwaCache(RadixCache):
    def __init__(self):
        self.sliding_window_size = 128
        self.page_size = 16
        self.prefix_sharing = False
        self.swa = True
        self.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(1024).reshape(1, -1)
        )
        self.token_to_kv_pool_allocator = _Allocator()

    def supports_swa(self):
        return self.swa

    def supports_prefix_sharing(self):
        return self.prefix_sharing


class TestCompletedPrefillSWA(CustomTestCase):
    def make_batch(self):
        cache = _SwaCache()
        req = SimpleNamespace(
            origin_input_ids=list(range(515)),
            kv=ReqKvInfo(req_pool_idx=0),
            inflight_middle_chunks=0,
            is_retracted=False,
        )
        batch = ScheduleBatch(
            reqs=[req], tree_cache=cache, spec_algorithm=SpeculativeAlgorithm.NONE
        )
        return batch, req, cache.token_to_kv_pool_allocator

    def test_frees_only_completed_pages_before_window(self):
        with get_context().override_server_args(enable_hisparse=False):
            batch, req, allocator = self.make_batch()
            tree_cache = batch.tree_cache
            # Async result snapshots omit the cache; the scheduler owns it.
            batch = batch.copy()
            self.assertIsNone(batch.tree_cache)
            batch.maybe_evict_completed_prefill_swa(
                req, tree_cache, copy_done_synchronized=True
            )
            self.assertEqual(allocator.live, set(range(384, 1024)))
            self.assertEqual(req.kv.get_evicted_seqlen(ComponentType.SWA), 384)
            # A repeated completion must not free the same slots twice.
            batch.maybe_evict_completed_prefill_swa(
                req, tree_cache, copy_done_synchronized=True
            )
            self.assertEqual(allocator.live, set(range(384, 1024)))

    def test_ineligible_requests_keep_allocations(self):
        cases = (
            "pending_copy",
            "mixed",
            "speculative",
            "prefix_sharing",
            "hisparse",
            "middle_chunk",
            "retracted",
            "no_kv",
            "no_swa",
        )
        for case in cases:
            with (
                self.subTest(case=case),
                get_context().override_server_args(enable_hisparse=case == "hisparse"),
            ):
                batch, req, allocator = self.make_batch()
                batch.decoding_reqs = [req] if case == "mixed" else None
                if case == "speculative":
                    batch.spec_algorithm = SpeculativeAlgorithm.EAGLE
                batch.tree_cache.prefix_sharing = case == "prefix_sharing"
                batch.tree_cache.swa = case != "no_swa"
                req.inflight_middle_chunks = int(case == "middle_chunk")
                req.is_retracted = case == "retracted"
                if case == "no_kv":
                    req.kv.req_pool_idx = None
                batch.maybe_evict_completed_prefill_swa(
                    req, batch.tree_cache, copy_done_synchronized=case != "pending_copy"
                )
                self.assertEqual(allocator.live, set(range(1024)))
                self.assertEqual(req.kv.get_evicted_seqlen(ComponentType.SWA), 0)


if __name__ == "__main__":
    unittest.main()
