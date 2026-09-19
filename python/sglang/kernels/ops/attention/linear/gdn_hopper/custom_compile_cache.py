# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 FlashInfer team.

import functools
import warnings
from collections.abc import Hashable

import cutlass.cute as cute

_in_mem_compile_cache: dict = {}


def _as_options_tuple(options):
    if options is None:
        return ()
    if isinstance(options, tuple):
        return options
    return (options,)


class KeyedCompileMixin:
    def manual_cache_key(self, *attr_names):
        collected_attrs = tuple(
            (attr_name, getattr(self, attr_name)) for attr_name in attr_names
        )
        compile_key = (type(self).__mro__,) + collected_attrs
        hash(compile_key)
        setattr(self, "_KeyedCompileMixin_compile_key", compile_key)  # noqa: B010

    def _get_compile_key(self):
        compile_key = getattr(self, "_KeyedCompileMixin_compile_key", None)
        if compile_key is None:
            warnings.warn(
                f"{type(self).__name__} is using automatic DSL compile-cache key generation; "
                "call manual_cache_key(...) at the end of __init__ to avoid host launch overhead.",
                RuntimeWarning,
                stacklevel=2,
            )
            collected_attrs = tuple(
                (attr_name, attr_value)
                for attr_name, attr_value in sorted(self.__dict__.items())
                if not attr_name.startswith("_")
            )
            compile_key = (str(type(self).__mro__),) + tuple(collected_attrs)
            try:
                hash(compile_key)
            except TypeError:
                collected_attrs = tuple(
                    (attr_name, attr_value)
                    for attr_name, attr_value in collected_attrs
                    if isinstance(attr_value, Hashable)
                )
                compile_key = (str(type(self).__mro__),) + tuple(collected_attrs)
            setattr(self, "_KeyedCompileMixin_compile_key", compile_key)  # noqa: B010

        return compile_key


@functools.cache
def _compile_options_tuple_key(options):
    return tuple((type(option), option.value) for option in options)


def _compile_options_key(options):
    if options is None:
        return None
    return _compile_options_tuple_key(_as_options_tuple(options))


def cached_compile(func, *args, compile_options=None, **kwargs):
    cache_key = (func._get_compile_key(), _compile_options_key(compile_options))
    compiled_fn = _in_mem_compile_cache.get(cache_key)

    if compiled_fn is None:
        compiled_fn = cute.compile[compile_options](func, *args, **kwargs)
        _in_mem_compile_cache[cache_key] = compiled_fn

    return compiled_fn
