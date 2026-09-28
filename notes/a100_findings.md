# A100 (SM80 / Ampere) Optimization Findings

## Summary
- **Architecture**: 108 SMs, 312 TFLOPS (dense FP16 Tensor Core), HBM2e (2039 GB/s).
- **Key Bottlenecks**:
  - Small tiles (e.g. 64x64) suffer from poor arithmetic intensity and memory bandwidth saturation.
  - Large tiles (e.g. 256x256) increase register pressure and reduce occupancy.
- **Optimal Configurations**:
  - `BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, num_warps=8, num_stages=4`.
