#!/bin/bash
# Profile Triton GEMM kernel using Nsight Compute (ncu) or Nsight Systems (nsys)
# Usage: ./scripts/profile_ncu.sh <kernel_name>

KERNEL=${1:-"grouped"}
OUTPUT_DIR="results/profiles"
mkdir -p "$OUTPUT_DIR"

echo "Profiling kernel: $KERNEL"

# Profile with Nsight Systems
nsys profile --stats=true \
    -o "$OUTPUT_DIR/nsys_${KERNEL}" \
    python3 -c "
import torch
from kernels.common.v01_grouped_ordering import matmul_grouped
a = torch.randn((2048, 2048), device='cuda', dtype=torch.float16)
b = torch.randn((2048, 2048), device='cuda', dtype=torch.float16)
for _ in range(5):
    matmul_grouped(a, b)
"
