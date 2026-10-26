MiMo MXFP4 optimizations on SM90
==============================

The model includes QKV preparation fusion. Enable the additional router,
TP communication and completed-prefill SWA optimizations with:

  export SGLANG_OPT_MIMO_TINY_ROUTER=1
  export SGLANG_OPT_MIMO_SP_COLLECTIVES=1
  export SGLANG_OPT_EVICT_COMPLETED_PREFILL_SWA=1

These flags default to disabled. They use ordinary runtime entry points;
no import hooks or generated model files are needed. Install the DeepGEMM
revision pinned in python/pyproject.toml. Expert weights remain native MXFP4;
MegaMoE quantizes activations to FP8 during execution.

Measured setup: 8 H100 GPUs, TP8, EP8, attention-DP2 (attention-TP4), FA3,
BF16 model dtype, native checkpoint MXFP4 experts, --moe-runner-backend marlin,
--moe-a2a-backend megamoe, --disable-radix-cache. Speculation/MTP was disabled.
The checkpoint was XiaomiMiMo/MiMo-V2.6-Flash-RL revision
5711b268169967567844e1e560e8a3966da959b1. Context length was 40960,
chunked-prefill-size 8192 (resolved to 4096 per DP rank), mem-fraction-static
0.85, max-running-requests 2048 and cuda-graph-max-bs-decode 128.

Restrictions and numerical checks:
- Router: existing tiny_gemm_bf16 for 1..16 rows, 4096 hidden, 256 experts;
  BF16 input/weight and FP32 logits. All other shapes use the original path.
  47 real checkpoint gates and 2914 synthetic rows had no top-8 set changes.
- TP communication: TP4, Hopper multicast, BF16 contiguous hidden=4096;
  bucketed push collectives with NCCL fallback. Reduce-scatter allocates a
  disjoint output to avoid an input/output race. All-gather was bit-exact;
  reduce-scatter is not bit-exact because the reduction order changes.
  Only the MiMo attention-TP group owns the workspace. Rows >=4096, unsupported
  shapes, unavailable multicast and occupied kernel registries use the original
  collective path. MTP, overlap scheduling and deterministic inference disable
  this fast path. Destroying the group releases its workspace and registration.
- SWA release: completed, non-mixed prefill after the existing copy completion
  event; no speculation, prefix sharing or HiSparse. Other paths are unchanged.

Historical experiment results (2026-10-25, before runtime integration): five ISL/OSL pairs
1024/1024, 4096/512, 8192/512, 32768/128, 512/2048, each at concurrency
1/8/32/128, two repeats (seeds 42/43). Each repeat used max(2, 2*C) requests,
fixed input/output lengths, --tokenize-prompt, --random-range-ratio 1 and one
warmup request. Equal-weight E2E geometric mean: 1.113175x versus MegaMoE
alone and 1.609624x versus Marlin. These are full-matrix numbers, not a promise
for another topology or workload. Marlin used W4A16; both MegaMoE arms W4A8.
GSM8K 5-shot, 1314 questions, max 2048 output tokens, concurrency 32:
Marlin 1213, MegaMoE 1195, combined 1211 correct. These are observed counts,
not a claim of numerical equivalence. The additional combined Nsight capture
was deferred. Detailed historical records remain under logs/mimo_nomtp.
