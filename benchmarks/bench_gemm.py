"""bench_gemm.py: Unified benchmark harness for comparing Triton GEMM kernels.

Tests correctness, measures latency via triton.testing.do_bench,
computes TFLOPS, and compares against torch.matmul (cuBLAS).
"""

import argparse
import json
import torch
import triton
from benchmarks.shapes import BENCHMARK_SHAPES, get_flops
from kernels.common.v00_baseline import matmul_baseline
from kernels.common.v01_grouped_ordering import matmul_grouped
from kernels.common.v02_tuned import matmul_tuned
from kernels.common.v03_autotuned import matmul_autotuned
from kernels.common.v04_persistent import matmul_persistent


KERNELS = {
    "baseline": matmul_baseline,
    "grouped": matmul_grouped,
    "tuned": matmul_tuned,
    "autotuned": matmul_autotuned,
    "persistent": matmul_persistent,
}


def benchmark_shape(M: int, N: int, K: int, kernel_fn, dtype=torch.float16, device="cuda"):
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)

    # 1. Correctness check
    ref = torch.matmul(a, b)
    out = kernel_fn(a, b)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    # 2. Performance benchmark
    ms = triton.testing.do_bench(lambda: kernel_fn(a, b), warmup=25, rep=100)
    ref_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), warmup=25, rep=100)

    flops = get_flops(M, N, K)
    tflops = (flops / (ms * 1e-3)) / 1e12
    ref_tflops = (flops / (ref_ms * 1e-3)) / 1e12

    return {
        "M": M, "N": N, "K": K,
        "latency_ms": ms,
        "tflops": tflops,
        "ref_latency_ms": ref_ms,
        "ref_tflops": ref_tflops,
        "speedup_vs_ref": ref_ms / ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Triton GEMM Benchmark")
    parser.add_argument("--kernel", type=str, default="grouped", choices=list(KERNELS.keys()))
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    device_name = torch.cuda.get_device_name(0)
    print(f"Benchmarking on {device_name} with kernel: {args.kernel}\n")
    print(f"{'M':>6} {'N':>6} {'K':>6} | {'Kernel (ms)':>11} {'TFLOPS':>8} | {'Ref (ms)':>9} {'Ref TFLOPS':>10} | {'Rel Perf':>8}")
    print("-" * 75)

    results = []
    kernel_fn = KERNELS[args.kernel]

    for M, N, K in BENCHMARK_SHAPES:
        try:
            res = benchmark_shape(M, N, K, kernel_fn)
            results.append(res)
            print(f"{M:>6} {N:>6} {K:>6} | {res['latency_ms']:>11.3f} {res['tflops']:>8.2f} | {res['ref_latency_ms']:>9.3f} {res['ref_tflops']:>10.2f} | {res['speedup_vs_ref']:>7.2f}x")
        except Exception as e:
            print(f"{M:>6} {N:>6} {K:>6} | FAILED: {e}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"device": device_name, "kernel": args.kernel, "results": results}, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
