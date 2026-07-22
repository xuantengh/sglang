# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

import ctypes
import logging
import threading
from typing import List, Optional, Sequence

import torch

from sglang.srt.mem_cache.embedding_store import EmbeddingStore, EmbeddingStoreConfig
from sglang.srt.mem_cache.hicache_storage import HiCacheStorage, HiCacheStorageConfig
from sglang.srt.mem_cache.storage import StorageBackendFactory

logger = logging.getLogger(__name__)

_SUPPORTED_STORAGE_BACKENDS = frozenset({"file"})


class HiCacheEmbeddingStore(EmbeddingStore):
    """Adapt an embedding blob to a tensor-oriented HiCache storage backend.

    The embedding cache exposes scatter/gather host pointers because entries can
    span non-contiguous pool runs. HiCache stores one value per key, so this
    adapter coalesces writes and scatters reads while keeping that detail out of
    both the controller and the storage backend.
    """

    backend_name = "HiCache"

    def __init__(
        self,
        config: EmbeddingStoreConfig,
        storage: Optional[HiCacheStorage] = None,
    ) -> None:
        self._config = config
        self._storage = storage
        self._storage_lock = threading.Lock()

        storage_backend = config.extra_config.get("storage_backend", "file")
        if storage_backend not in _SUPPORTED_STORAGE_BACKENDS:
            supported = sorted(_SUPPORTED_STORAGE_BACKENDS)
            raise ValueError(
                "HiCache multimodal embedding storage backend "
                f"'{storage_backend}' is not supported. "
                f"Supported backends: {supported}."
            )
        self._storage_backend = storage_backend
        self._storage_backend_extra_config = {
            key: value
            for key, value in config.extra_config.items()
            if key != "storage_backend"
        }

    def _get_storage(self) -> HiCacheStorage:
        if self._storage is not None:
            return self._storage

        with self._storage_lock:
            if self._storage is None:
                storage_config = HiCacheStorageConfig(
                    tp_rank=self._config.tp_rank,
                    tp_size=self._config.tp_size,
                    pp_rank=0,
                    pp_size=1,
                    attn_cp_rank=0,
                    attn_cp_size=1,
                    is_mla_model=False,
                    enable_storage_metrics=False,
                    is_page_first_layout=True,
                    model_name=(
                        f"mm-embedding/{self._config.model_name}"
                        if self._config.model_name is not None
                        else "mm-embedding"
                    ),
                    extra_config=self._storage_backend_extra_config,
                )
                self._storage = StorageBackendFactory.create_backend(
                    backend_name=self._storage_backend,
                    storage_config=storage_config,
                    mem_pool_host=None,
                )
        if self._storage is None:
            raise RuntimeError("HiCache embedding storage initialization failed.")
        return self._storage

    @staticmethod
    def _as_byte_tensor(ptr: int, size: int) -> torch.Tensor:
        if ptr <= 0:
            raise ValueError(f"Buffer pointer must be positive, got {ptr}.")
        if size < 0:
            raise ValueError(f"Buffer size must be non-negative, got {size}.")
        if size == 0:
            return torch.empty(0, dtype=torch.uint8)
        buffer = (ctypes.c_ubyte * size).from_address(ptr)
        return torch.frombuffer(buffer, dtype=torch.uint8)

    @staticmethod
    def _validate_batch_lengths(
        hashes: Sequence[str], ptrs: Sequence[object], sizes: Sequence[object]
    ) -> None:
        if len(hashes) != len(ptrs) or len(hashes) != len(sizes):
            raise ValueError(
                "Hash, pointer, and size counts must match: "
                f"{len(hashes)}, {len(ptrs)}, {len(sizes)}."
            )

    @classmethod
    def _coalesce_buffers(cls, ptrs: List[int], sizes: List[int]) -> torch.Tensor:
        if len(ptrs) != len(sizes):
            raise ValueError(
                f"Pointer and size counts differ: {len(ptrs)} != {len(sizes)}."
            )
        buffers = [cls._as_byte_tensor(ptr, size) for ptr, size in zip(ptrs, sizes)]
        if not buffers:
            return torch.empty(0, dtype=torch.uint8)
        if len(buffers) == 1:
            return buffers[0]
        return torch.cat(buffers)

    @classmethod
    def _scatter_buffer(
        cls, value: torch.Tensor, ptrs: List[int], sizes: List[int]
    ) -> bool:
        if len(ptrs) != len(sizes):
            raise ValueError(
                f"Pointer and size counts differ: {len(ptrs)} != {len(sizes)}."
            )
        expected_size = sum(sizes)
        value = value.contiguous().view(torch.uint8)
        if value.numel() != expected_size:
            logger.warning(
                "HiCache embedding size mismatch: expected %d bytes, got %d bytes.",
                expected_size,
                value.numel(),
            )
            return False

        offset = 0
        for ptr, size in zip(ptrs, sizes):
            target = cls._as_byte_tensor(ptr, size)
            target.copy_(value[offset : offset + size])
            offset += size
        return True

    def _get_buffers(self, mm_hash: str, ptrs: List[int], sizes: List[int]) -> bool:
        target = torch.empty(sum(sizes), dtype=torch.uint8)
        value = self._get_storage().get(self.get_key(mm_hash), target)
        if value is None:
            return False
        return self._scatter_buffer(value, ptrs, sizes)

    def _put_buffers(self, mm_hash: str, ptrs: List[int], sizes: List[int]) -> bool:
        value = self._coalesce_buffers(ptrs, sizes)
        return self._get_storage().set(self.get_key(mm_hash), value)

    def batch_get(
        self, hashes: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        self._validate_batch_lengths(hashes, ptrs, sizes)
        return [
            self._get_buffers(mm_hash, [ptr], [size])
            for mm_hash, ptr, size in zip(hashes, ptrs, sizes)
        ]

    def batch_put(
        self, hashes: List[str], ptrs: List[int], sizes: List[int]
    ) -> List[bool]:
        self._validate_batch_lengths(hashes, ptrs, sizes)
        return [
            self._put_buffers(mm_hash, [ptr], [size])
            for mm_hash, ptr, size in zip(hashes, ptrs, sizes)
        ]

    def batch_get_into_multi_buffers(
        self,
        hashes: List[str],
        ptrs: List[List[int]],
        sizes: List[List[int]],
    ) -> List[bool]:
        self._validate_batch_lengths(hashes, ptrs, sizes)
        return [
            self._get_buffers(mm_hash, ptr_list, size_list)
            for mm_hash, ptr_list, size_list in zip(hashes, ptrs, sizes)
        ]

    def batch_put_from_multi_buffers(
        self,
        hashes: List[str],
        ptrs: List[List[int]],
        sizes: List[List[int]],
    ) -> List[bool]:
        self._validate_batch_lengths(hashes, ptrs, sizes)
        return [
            self._put_buffers(mm_hash, ptr_list, size_list)
            for mm_hash, ptr_list, size_list in zip(hashes, ptrs, sizes)
        ]

    def batch_is_exist(self, hashes: List[str]) -> List[bool]:
        storage = self._get_storage()
        return [storage.exists(self.get_key(mm_hash)) for mm_hash in hashes]
