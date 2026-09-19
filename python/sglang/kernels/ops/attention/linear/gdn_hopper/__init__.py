# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 FlashInfer team.

"""Hopper (SM90) CuTe DSL Gated DeltaNet prefill kernel.

The kernel uses 64-token chunks, TMA staging, WGMMA, a hierarchical 64x64
triangular inverse, and an FP32 recurrent state held in registers across
chunks.  It is the Hopper counterpart to the Blackwell TMEM/tcgen05 path.
"""

from .delta_rule_sm90 import delta_rule_prefill_dsl_sm90

__all__ = ["delta_rule_prefill_dsl_sm90"]
