# CUDA Lab 2: Tensor Core Pipelines (`cp.async`, `ldmatrix`, `mma.sync`)

## Objectives
- Understand the low-level PTX instructions that `tl.dot` and Triton pipeline scheduling generate:
  - `cp.async`: Asynchronously copies data from Global Memory directly into Shared Memory without passing through general-purpose registers.
  - `ldmatrix`: Loads matrix fragments from shared memory into warp registers with the required layout.
  - `mma.sync`: Executes warp-synchronous Tensor Core matrix multiply-accumulate operations.
- Understand multi-stage software pipelining and double/triple buffering in shared memory.
