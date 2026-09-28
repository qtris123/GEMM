"""experiments.py: A100-specific tile sweep and micro-benchmarking experiments.

Sweeps BLOCK_M, BLOCK_N, BLOCK_K, num_warps, and num_stages to explore
occupancy and register pressure tradeoffs on Ampere (SM80).
"""

import itertools
import torch
import triton
from kernels.common.v01_grouped_ordering import _matmul_kernel_grouped
from configs.a100 import TILE_M_CANDIDATES, TILE_N_CANDIDATES, TILE_K_CANDIDATES, NUM_WARPS_CANDIDATES, NUM_STAGES_CANDIDATES


def run_sweep(M=2048, N=2048, K=2048):
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    print(f"Sweeping configs for {M}x{N}x{K}...")
    # Add parameter sweeps and latency tracking here
