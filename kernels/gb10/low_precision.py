"""low_precision.py: Low-Precision (FP8 / FP4) GEMM on GB10.

Explores 8-bit floating point matrix multiplication with scaling factors
and accumulation precision.
"""

import torch

def matmul_fp8(a: torch.Tensor, b: torch.Tensor, scale_a: float = 1.0, scale_b: float = 1.0) -> torch.Tensor:
    """Low-precision FP8 GEMM template."""
    return torch.matmul(a.float(), b.float()) * (scale_a * scale_b)
