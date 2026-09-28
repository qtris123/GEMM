"""grouped_gemm.py: Grouped GEMM / Mixture-of-Experts (MoE) routing.

Handles multiple matrix multiplications with heterogeneous batch sizes
in a single kernel launch.
"""

import torch
from typing import List

def grouped_gemm_ref(inputs: List[torch.Tensor], weights: List[torch.Tensor]) -> List[torch.Tensor]:
    """Reference implementation of grouped GEMM."""
    return [torch.matmul(x, w) for x, w in zip(inputs, weights)]
