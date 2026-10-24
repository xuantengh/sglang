import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.mimo_rope_scale_qkv import mimo_rope_scale_qkv
from sglang.kernels.ops.attention.rope import rotary_embedding
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def unfused(qkv, positions, cache, kv_heads):
    q, k, v = qkv.split([16 * 192, kv_heads * 192, kv_heads * 128], -1)
    rotary_embedding(positions, q, k, 192, cache, True)
    return q.contiguous(), k, v * 0.707


def fused(qkv, positions, cache, kv_heads):
    return mimo_rope_scale_qkv(qkv, positions, cache, 16, kv_heads, 0.707)


@marker.parametrize("repetition", [0, 1])
@marker.parametrize("tokens", [1, 16, 128, 1024, 4096, 8192], [16, 4096])
@marker.parametrize("kv_heads", [1, 2])
@marker.benchmark("impl", ["unfused", "fused"])
def benchmark(repetition: int, tokens: int, kv_heads: int, impl: str):
    torch.manual_seed(42 + repetition)
    qkv = torch.randn(
        tokens, 16 * 192 + kv_heads * 320, device="cuda", dtype=torch.bfloat16
    )
    angles = torch.randn(4096, 32, device="cuda")
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).bfloat16()
    positions = torch.randint(0, 4096, (tokens,), device="cuda")
    # Rotate all read buffers beyond L2; each captured call allocates its output.
    return marker.do_bench(
        fused if impl == "fused" else unfused,
        input_args=(qkv, positions, cache, kv_heads),
        graph_clone_args=(0, 1, 2),
        # Only indexed RoPE rows are read; full tensor bytes misstate bandwidth.
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
