"""gated_mlp.py: Fused SwiGLU / Gated MLP kernel (LLaMA style).

Computes:
  gate = X @ W_gate
  up   = X @ W_up
  out  = (SiLU(gate) * up) @ W_down
"""

import torch

def gated_mlp_ref(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch implementation of Gated MLP."""
    gate = torch.matmul(x, w_gate)
    up = torch.matmul(x, w_up)
    return torch.nn.functional.silu(gate) * up
