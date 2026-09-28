"""Matrix shapes for GEMM benchmarking across A100 and GB10.
Includes square, rectangular, LLM attention, and MLP projections.
"""

from typing import List, Tuple, Dict

# Standard benchmarking shapes (M, N, K)
SQUARE_SHAPES: List[Tuple[int, int, int]] = [
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]

# LLM inference / prefill / decode shapes
LLM_SHAPES: List[Tuple[int, int, int]] = [
    (128, 4096, 4096),     # Small batch attention / prefill
    (512, 4096, 4096),     # Medium batch
    (2048, 4096, 4096),    # Large batch / context
    (4096, 11008, 4096),   # LLaMA-style FFN gate/up projection
    (4096, 4096, 11008),   # LLaMA-style FFN down projection
]

BENCHMARK_SHAPES: List[Tuple[int, int, int]] = SQUARE_SHAPES + LLM_SHAPES

def get_flops(M: int, N: int, K: int) -> float:
    """Computes total floating point operations for M x K x N matmul."""
    return 2.0 * float(M) * float(N) * float(K)
