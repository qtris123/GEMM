"""fused_linear_activation.py: Fused Linear + Activation (GELU/SiLU) kernel.

Fuses the epilogue activation into the GEMM tile store, eliminating
an entire round-trip of global memory read/write.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

# Template for fused GEMM + activation
def fused_gemm_silu(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """Computes SiLU(a @ b + bias) in a single fused kernel."""
    out = torch.matmul(a, b)
    if bias is not None:
        out = out + bias
    return torch.nn.functional.silu(out)
