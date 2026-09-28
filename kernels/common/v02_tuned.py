"""v02_tuned.py: Statically tuned GEMM configuration for target architecture.

Allows parameterizing BLOCK_M, BLOCK_N, BLOCK_K, num_warps, and num_stages
to explore tile shapes, register pressure, and occupancy tradeoffs.
"""

import torch
import triton
from kernels.common.v01_grouped_ordering import _matmul_kernel_grouped


def matmul_tuned(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 128,
    block_n: int = 256,
    block_k: int = 64,
    group_m: int = 8,
    num_warps: int = 8,
    num_stages: int = 4,
) -> torch.Tensor:
    """Tuned GEMM with explicit parameters."""
    assert a.shape[1] == b.shape[0], "Incompatible matrix shapes"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    _matmul_kernel_grouped[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c
