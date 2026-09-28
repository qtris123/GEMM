# CUDA Lab 1: Shared-Memory Tiled GEMM

## Objectives
- Understand block and thread index mapping (`blockIdx`, `threadIdx`).
- Use shared memory tiles (`__shared__`) to maximize L1/SM data reuse.
- Implement barrier synchronization (`__syncthreads`).
- Analyze memory coalescing and bank conflicts.

## Files
- `gemm_smem.cu`: CUDA implementation of tiled GEMM with timer and verification.

## Compilation
On Blackwell (ARM64 host):
```bash
nvcc -O3 -arch=native gemm_smem.cu -o gemm_smem
./gemm_smem
```
On A100 (x86_64 host):
```bash
nvcc -O3 -arch=sm_80 gemm_smem.cu -o gemm_smem
./gemm_smem
```
