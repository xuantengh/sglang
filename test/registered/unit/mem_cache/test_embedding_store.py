"""Unit tests for multimodal embedding storage backends."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.embedding_store import (
    EmbeddingStoreConfig,
    EmbeddingStoreFactory,
)
from sglang.srt.mem_cache.storage import StorageBackendFactory
from sglang.srt.mem_cache.storage.hicache_embedding_store import (
    HiCacheEmbeddingStore,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeHiCacheStorage:
    def __init__(self):
        self.values = {}

    def get(self, key, target_location=None, target_sizes=None):
        value = self.values.get(key)
        if value is None:
            return None
        target_location.copy_(value)
        return target_location

    def set(
        self,
        key,
        value=None,
        target_location=None,
        target_sizes=None,
    ):
        self.values[key] = value.clone()
        return True

    def exists(self, key):
        return key in self.values


def _config(extra_config=None):
    return EmbeddingStoreConfig(
        tp_rank=0,
        tp_size=1,
        model_name="org/model",
        extra_config=extra_config or {},
    )


class TestHiCacheEmbeddingStore(CustomTestCase):
    def test_multi_buffer_round_trip(self):
        storage = _FakeHiCacheStorage()
        store = HiCacheEmbeddingStore(_config(), storage=storage)
        first = torch.tensor([1, 2, 3], dtype=torch.uint8)
        second = torch.tensor([4, 5], dtype=torch.uint8)

        put_results = store.batch_put_from_multi_buffers(
            ["hash"],
            [[first.data_ptr(), second.data_ptr()]],
            [[first.nbytes, second.nbytes]],
        )

        self.assertEqual(put_results, [True])
        self.assertEqual(storage.values["emb_hash"].tolist(), [1, 2, 3, 4, 5])

        first.zero_()
        second.zero_()
        get_results = store.batch_get_into_multi_buffers(
            ["hash"],
            [[first.data_ptr(), second.data_ptr()]],
            [[first.nbytes, second.nbytes]],
        )

        self.assertEqual(get_results, [True])
        self.assertEqual(first.tolist(), [1, 2, 3])
        self.assertEqual(second.tolist(), [4, 5])

    def test_flat_buffer_and_existence_interfaces(self):
        storage = _FakeHiCacheStorage()
        store = HiCacheEmbeddingStore(_config(), storage=storage)
        source = torch.tensor([7, 8, 9], dtype=torch.uint8)
        target = torch.zeros_like(source)

        self.assertEqual(
            store.batch_put(["present"], [source.data_ptr()], [source.nbytes]),
            [True],
        )
        self.assertEqual(store.batch_is_exist(["present", "missing"]), [True, False])
        self.assertEqual(
            store.batch_get(["present"], [target.data_ptr()], [target.nbytes]),
            [True],
        )
        self.assertTrue(torch.equal(target, source))

    def test_rejects_storage_backend_without_tensor_interface(self):
        with self.assertRaisesRegex(ValueError, "not supported"):
            HiCacheEmbeddingStore(_config({"storage_backend": "nixl"}))

    def test_factory_creates_hicache_backend_lazily(self):
        with patch.object(StorageBackendFactory, "create_backend") as create_storage:
            store = EmbeddingStoreFactory.create_backend(
                backend_name="hicache", config=_config()
            )

        self.assertIsInstance(store, HiCacheEmbeddingStore)
        create_storage.assert_not_called()

    def test_lazy_storage_creation_forwards_hicache_config(self):
        storage = _FakeHiCacheStorage()
        config = EmbeddingStoreConfig(
            tp_rank=2,
            tp_size=4,
            model_name="org/model",
            extra_config={"storage_backend": "file", "max_size": "2G"},
        )
        store = HiCacheEmbeddingStore(config)

        with patch.object(
            StorageBackendFactory, "create_backend", return_value=storage
        ) as create_storage:
            self.assertEqual(store.batch_is_exist(["missing"]), [False])

        create_storage.assert_called_once()
        call = create_storage.call_args.kwargs
        self.assertEqual(call["backend_name"], "file")
        self.assertIsNone(call["mem_pool_host"])
        storage_config = call["storage_config"]
        self.assertEqual(storage_config.tp_rank, 2)
        self.assertEqual(storage_config.tp_size, 4)
        self.assertEqual(storage_config.model_name, "mm-embedding/org/model")
        self.assertEqual(storage_config.extra_config, {"max_size": "2G"})


if __name__ == "__main__":
    unittest.main()
