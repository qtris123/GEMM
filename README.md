# Triton GEMM Optimization Curriculum: A100 → GB10
## 1. Goal
This project teaches GPU kernel performance engineering through GEMM optimization, using Triton as the primary programming model.
Progression:
```text
A100 / Ampere (SM80)
        ↓
learn GEMM fundamentals
        ↓
GB10 / Blackwell (SM121)
        ↓
retune and study newer techniques
```
Main topics: tiling, memory hierarchy, data reuse, Tensor Cores, register pressure, occupancy, pipelines, persistent scheduling, architecture-aware tuning, profiling, and generated code.
By the end, you should be able to:
1. Write and optimize a Triton GEMM on A100.
2. Profile it and explain its bottlenecks.
3. Run the same shared kernels on GB10.
4. Retune them for Blackwell.
5. Explore Blackwell-oriented features.
6. Understand what Triton abstracts compared with CUDA.
## 2. Repository Structure
```text
GEMM/
├── README.md
├── requirements.txt
├── kernels/
│   ├── common/
│   │   ├── v00_baseline.py
│   │   ├── v01_grouped_ordering.py
│   │   ├── v02_tuned.py
│   │   ├── v03_autotuned.py
│   │   └── v04_persistent.py
│   ├── a100/
│   │   └── experiments.py
│   ├── gb10/
│   │   ├── v05_tensor_descriptor.py
│   │   ├── v06_warp_specialized.py
│   │   └── low_precision.py
│   └── inference/
│       ├── fused_linear_activation.py
│       ├── gated_mlp.py
│       └── grouped_gemm.py
├── configs/
│   ├── a100.py
│   └── gb10.py
├── cuda_labs/
│   ├── smem_gemm/
│   └── tensorcore_pipeline/
├── benchmarks/
│   ├── bench_gemm.py
│   ├── shapes.py
│   └── plot_results.py
├── scripts/
│   ├── run_a100.sh
│   ├── run_gb10.sh
│   └── profile_ncu.sh
├── results/
│   ├── a100/
│   └── gb10/
└── notes/
    ├── a100_findings.md
    ├── gb10_findings.md
    ├── architecture_comparison.md
    └── triton_vs_cuda.md
```
Structure rules:
- `kernels/common/`: shared Triton kernels for both GPUs.
- `configs/`: architecture-specific tuning.
- `kernels/a100/`, `kernels/gb10/`: architecture-specific experiments.
- `cuda_labs/`: small lower-level exercises.
Keep the early kernels shared so A100-vs-GB10 comparisons use the same source.
## 3. Benchmarking Setup
Build the benchmark harness first.
Correctness:
```python
ref = torch.matmul(a, b)
out = triton_matmul(a, b)
torch.testing.assert_close(out, ref, atol=..., rtol=...)
```
Test square, rectangular, small, large, and non-tile-aligned shapes.
Compare against:
- `torch.matmul`
- Triton's official matmul example
- optionally cuBLAS/cuBLASLt
Suggested shapes:
```text
2048×2048×2048
4096×4096×4096
8192×8192×8192
M=128,  N=4096,  K=4096
M=512,  N=4096,  K=4096
M=2048, N=4096,  K=4096
M=4096, N=11008, K=4096
M=4096, N=4096,  K=11008
```
Record:
```text
GPU, dtype, M, N, K
BLOCK_M, BLOCK_N, BLOCK_K
num_warps, num_stages
latency, TFLOP/s
reference TFLOP/s
relative performance
```
For selected runs profile registers, shared memory, occupancy, DRAM/L2 throughput, Tensor Core utilization, SM utilization, and warp stalls.
# Phase I — A100 / SM80
## 4. A0 — Triton Fundamentals
Learn `tl.program_id`, `tl.arange`, pointer arithmetic, masks, `tl.load`, `tl.store`, reductions, broadcasting, and `tl.dot`.
Warmups:
- vector addition
- tiled transpose
Deliverable: small Triton warmup examples.
## 5. A1 — Baseline Blocked GEMM
Implement:
```text
C[M,N] = A[M,K] × B[K,N]
```
Each Triton program computes one output tile and loops over K.
Focus on output tiling, K blocking, reuse, and FP32 accumulation for lower-precision inputs.
Deliverable:
```text
kernels/common/v00_baseline.py
```
## 6. A2 — Tile Shape Tuning
Sweep `BLOCK_M`, `BLOCK_N`, and `BLOCK_K`.
Example candidates:
```text
64×64×32
64×128×32
128×64×32
128×128×32
128×256×32
```
Core tradeoff:
```text
larger tile → more reuse → more registers → possibly lower occupancy
```
Deliverable:
```text
kernels/common/v02_tuned.py
```
## 7. A3 — L2-Aware Ordering
Implement grouped tile ordering similar to Triton's matmul tutorial.
Study output-tile traversal, L2 locality, and cache reuse.
Deliverable:
```text
kernels/common/v01_grouped_ordering.py
```
## 8. A4 — `num_warps`
Sweep:
```text
2, 4, 8
```
Study tile size, register pressure, occupancy, and parallelism.
Record:
```text
tile | num_warps | registers/thread | occupancy | TFLOP/s
```
## 9. A5 — `num_stages`
Sweep:
```text
2, 3, 4, 5
```
Study software pipelining, latency hiding, buffering cost, and occupancy.
Deliverable: explain the best stage count for one representative GEMM.
## 10. A6 — Autotuning
Use:
```python
@triton.autotune(...)
```
Tune `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, and `num_stages`.
Keep the search space interpretable.
Deliverable:
```text
kernels/common/v03_autotuned.py
```
Record selected configurations for several shapes.
## 11. A7 — Inspect Generated Code
Inspect:
```text
Triton source → Triton IR → PTX → SASS
```
Check:
- whether `tl.dot` uses Tensor Cores
- register changes from tile changes
- scheduling changes from `num_stages`
- differences across configurations
Deliverable:
```text
notes/a100_codegen.md
```
## 12. A8 — Persistent GEMM
Implement a persistent variant where a smaller worker grid processes multiple tiles.
Study persistent workers, tile scheduling, load balancing, and scheduling overhead.
Deliverable:
```text
kernels/common/v04_persistent.py
```
## 13. A9 — Two CUDA Micro-Labs
Triton remains the main language.
CUDA Lab 1:
- shared-memory tiled GEMM
- `threadIdx`, `blockIdx`
- `__shared__`, `__syncthreads`
- coalescing and bank conflicts
CUDA Lab 2:
```text
cp.async
ldmatrix
mma.sync
```
Goal: understand the machinery behind `tl.dot`, not reach peak performance.
## 14. A100 Exit Criteria
Move to GB10 once you understand:
- tiling and K blocking
- arithmetic intensity and L2 locality
- `num_warps` and `num_stages`
- Tensor Core GEMM
- register pressure and occupancy
- autotuning and persistent scheduling
- basic Nsight Compute analysis
You do not need to match cuBLAS exactly.
# Phase II — GB10 / SM121
## 15. B1 — Architecture Transfer
Run the shared kernels unchanged on GB10:
```text
v00_baseline.py
v01_grouped_ordering.py
v02_tuned.py
v03_autotuned.py
v04_persistent.py
```
For representative shapes compare:
```text
A100 best config
GB10 using A100 config
GB10 best config
```
Compare tile sizes, `num_warps`, `num_stages`, occupancy, register pressure, and generated code.
Deliverable:
```text
notes/architecture_comparison.md
```
Label each comparison as:
```text
same source + same config
same source + different config
architecture-specific source
```
## 16. B2 — Tensor Descriptors / TMA-Oriented Data Movement
Explore tensor-descriptor APIs and modern multidimensional data movement supported by the Triton + GB10 stack.
Study tensor descriptors, TMA-style transfers, multidimensional loads, and interaction with tiling/pipelines.
Deliverable:
```text
kernels/gb10/v05_tensor_descriptor.py
```
## 17. B3 — Persistent GEMM Retuning
Retune the shared persistent GEMM for GB10.
Sweep tile size, worker count, warp count, stage count, and traversal order.
Deliverable: GB10 persistent results in `notes/gb10_findings.md`.
## 18. B4 — Warp Specialization
If supported by your Triton path, experiment with warp-specialized execution.
Study producer/consumer roles, data-movement warps, compute warps, synchronization, and latency hiding.
Deliverable:
```text
kernels/gb10/v06_warp_specialized.py
```
If unsupported or unstable, study an existing implementation instead.
## 19. B5 — Low-Precision GEMM
Explore one supported path:
- FP8
- FP4 / NVFP4
- block scaling
- quantized layouts
- scale application
- accumulation precision
Deliverable:
```text
kernels/gb10/low_precision.py
```
Measure correctness/error, throughput, memory reduction, and performance relative to BF16/FP16.
# Phase III — Inference-Oriented Kernels
## 20. Fused Linear + Activation
Implement:
```text
Y = activation(XW + b)
```
Use GELU or SiLU and compare fused vs unfused execution.
Deliverable:
```text
kernels/inference/fused_linear_activation.py
```
## 21. Gated MLP
Implement:
```text
gate = XW_gate
up   = XW_up
Y = SiLU(gate) * up
```
Study GEMM-heavy inference, epilogue fusion, and memory-traffic reduction.
Deliverable:
```text
kernels/inference/gated_mlp.py
```
## 22. Grouped GEMM / MoE
Study grouped GEMM for expert workloads.
Focus on variable matrix sizes, expert imbalance, tile scheduling, and persistent workers.
Deliverable:
```text
kernels/inference/grouped_gemm.py
```
## 23. Profiling Workflow
For each major optimization ask:
```text
1. Compute-bound or memory-bound?
2. Are Tensor Cores used effectively?
3. What limits occupancy?
4. Is memory traffic efficient?
5. Which stalls dominate?
6. Did the change fix the intended bottleneck?
```
Record:
```text
hypothesis → change → measurement → conclusion
```
## 24. Suggested Timeline
Week 1: Triton basics, benchmark harness, baseline GEMM.
Week 2: tile tuning, L2 ordering, `num_warps`, `num_stages`.
Week 3: autotuning, Nsight Compute, PTX/SASS, persistent GEMM.
Week 4: CUDA micro-labs and A100 summary.
Week 5: shared kernels on GB10, retuning, architecture comparison.
Week 6: tensor descriptors/TMA, persistent retuning, warp specialization.
Week 7: low-precision GEMM.
Week 8: inference capstone and final write-up.
## 25. Final Deliverables
Code:
- baseline, grouped-ordering, tuned, autotuned, and persistent GEMMs
- one GB10-specific experiment
- one low-precision experiment
- one inference-oriented kernel
Benchmark data:
```text
hardware
shape
dtype
kernel
configuration
latency
TFLOP/s
reference performance
```
Plots:
- performance progression
- Triton vs reference
- A100 vs GB10
- selected configuration by shape
Notes:
```text
notes/a100_findings.md
notes/gb10_findings.md
notes/architecture_comparison.md
notes/triton_vs_cuda.md
```
Final write-up:
1. What made the baseline slow?
2. Which optimizations mattered most?
3. Which settings transferred from A100 to GB10?
4. Which needed retuning?
5. What changed in generated code?
6. Which Blackwell-oriented techniques were useful?
7. What did Triton abstract away?
8. When would CUDA/CuTe be preferable?
9. What still changes for B200/SM100?
## 26. Optional B200 / SM100 Follow-Up
GB10 is Blackwell, but it is not B200/SM100.
After the main project, study:
```text
TCGen05
Tensor Memory / TMEM
SM100 Tensor Core pipelines
multi-CTA MMA
cluster-level execution
multicast
SM100 scheduling
```
If B200 access becomes available, port the mature benchmark harness and selected kernels.
# Triton vs CUDA
## 27. Why Triton Makes Learning Faster
Triton raises the programming level.
Instead of manually managing individual CUDA threads, you often work with logical blocks:
```python
pid = tl.program_id(0)
offsets = ...
x = tl.load(...)
```
A GEMM inner loop may look like:
```python
a = tl.load(...)
b = tl.load(...)
acc += tl.dot(a, b)
```
This lets you focus earlier on tile shape, reuse, scheduling, pipeline depth, fusion, and architecture tuning.
Changing `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, and `num_stages` is usually faster than rewriting equivalent CUDA code.
Triton's autotuning support also makes parameter exploration easier.
## 28. What Triton Abstracts Away
Compared with lower-level CUDA, Triton may hide:
```text
thread-level mapping
Tensor Core fragment layout
manual vectorized loads
shared-memory operand layout
explicit MMA instructions
async-copy bookkeeping
barrier management
pipeline-buffer bookkeeping
```
`tl.dot(a, b)` can compile to architecture-specific Tensor Core instructions without manual MMA code.
`num_stages=4` can influence pipelining without manually programming every buffer and barrier.
## 29. What Triton Does Not Abstract Away
You still need to reason about:
```text
tile size
arithmetic intensity
data reuse
register pressure
occupancy
cache locality
pipeline depth
matrix shape
load balancing
persistent scheduling
datatype
```
Triton removes implementation complexity, not performance reasoning.
## 30. What You Risk Missing
If you only work at the Triton level, these can remain too abstract:
```text
shared-memory banks
warp execution
Tensor Core instruction shapes
operand fragments
async copies
barriers
PTX
SASS
```
That is why the curriculum includes PTX/SASS inspection, Nsight Compute, and two small CUDA labs.
## 31. When CUDA or CuTe Becomes Useful
Triton is a strong default when rapid iteration, fusion, shape specialization, and block-oriented computation matter.
CUDA/CuTe/CUTLASS becomes more useful when:
- exact instruction control matters
- Triton does not expose a hardware feature
- compiler scheduling is insufficient
- unusual synchronization is required
- explicit memory-layout control matters
- architecture-specific kernel engineering is the main objective
Recommended mental model:
```text
Triton
  ↓
rapid kernel development
  ↓
profiling
  ↓
inspect generated code
  ↓
CUDA / CuTe / PTX when needed
```
## 32. Success Criteria
The project is complete when you can answer:
Algorithm:
- What does each Triton program compute?
- What data is reused?
- What tile shape makes sense?
Memory:
- What data movement dominates?
- Is cache reuse effective?
- Is the kernel memory-bound?
Compute:
- Are Tensor Cores being used?
- Is compute utilization high?
Resources:
- What limits occupancy?
Pipeline:
- Are loads and compute overlapped?
- Would persistence help?
Architecture:
- Why does A100 prefer one configuration?
- Why does GB10 prefer another?
Tooling:
- Can Nsight Compute identify the bottleneck?
- Can PTX/SASS explain surprising compiler behavior?
If you can answer these from measurements instead of guesses, the project has achieved its goal.
## 33. Starting References
- Huy Nguyen — `hopper-gemm-101`: https://github.com/HuyNguyen-hust/hopper-gemm-101
- FP32 Matmul Optimization on A100: https://rwtarpit.github.io/blog/posts/2026-07-25-fp32_matmul_A100.html
- Triton documentation: https://triton-lang.org/
- NVIDIA CUTLASS: https://github.com/NVIDIA/cutlass
Use these as references, but keep the repository centered on your own experiments, profiling, and architecture comparisons.
