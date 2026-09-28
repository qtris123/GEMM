# Triton GEMM Optimization Curriculum: A100 → GB10

## Project Goal

The goal of this project is to learn **GPU kernel performance engineering through GEMM optimization**, using **Triton as the primary programming model** and progressing across two NVIDIA GPU generations:

1. **A100 / Ampere (SM80)** — learn the core performance concepts in a relatively mature, well-documented environment.
2. **GB10 / Blackwell (SM121)** — retune the same ideas on a newer architecture and study how modern GPU kernels use persistent scheduling, tensor descriptors/TMA-style data movement, warp specialization, and lower-precision Tensor Core paths.

This is not meant to become a months-long attempt to reproduce every detail of cuBLAS. The target outcome is a strong mental model for:

- tiling,
- data reuse,
- memory hierarchy,
- Tensor Core utilization,
- occupancy,
- register pressure,
- software pipelining,
- persistent scheduling,
- architecture-aware tuning,
- and profiling-driven optimization.

The project should end with a repository that tells a clear performance story:

> Start from a simple Triton GEMM, improve it step by step on A100, port it to GB10, retune it for Blackwell, and explain why the optimal kernel configuration changes across architectures.

---

# 1. Guiding Principles

## 1.1 Optimize by measurement, not intuition alone

Every optimization should answer three questions:

1. **What bottleneck am I trying to remove?**
2. **What code or launch configuration did I change?**
3. **What changed in the profiler and benchmark results?**

Do not make a version simply because a tutorial says it is "better."

For each kernel revision, record:

- latency,
- achieved TFLOP/s,
- percentage of a strong reference implementation,
- register usage,
- shared-memory usage,
- occupancy,
- relevant memory bandwidth,
- Tensor Core utilization when applicable,
- dominant stall reasons.

The point is not merely to get a larger TFLOP/s number. The point is to be able to explain **why** it got larger.

---

## 1.2 Keep the A100 phase intentionally bounded

A100 is the foundation, not the final destination.

Use it to learn:

- block tiling,
- register reuse,
- L2-aware scheduling,
- `num_warps`,
- `num_stages`,
- Tensor Core GEMM,
- autotuning,
- persistent kernels,
- profiler interpretation.

Once those concepts are clear, move on.

Do **not** spend weeks chasing the last 1–2% of cuBLAS on Ampere unless a specific experiment teaches something important.

---

## 1.3 GB10 is a second architecture, not just a faster target

Do not treat the GB10 phase as:

> "Run the A100 kernel again and report the new number."

Instead, use it to ask:

- Does the same tile shape still work?
- Does the same number of warps still work?
- Does the same pipeline depth still work?
- Does the persistent schedule behave differently?
- Does the compiler lower `tl.dot` differently?
- Do register pressure and occupancy tradeoffs move?
- What newer Blackwell-oriented programming features can Triton expose on this target?

The most interesting result may be that the best A100 configuration is **not** the best GB10 configuration.

---

# 2. Recommended Repository Structure

Use a **hybrid structure**:

- shared Triton kernels live under `kernels/common/`,
- architecture-specific experiments live under `kernels/a100/` and `kernels/gb10/`,
- tuning/search spaces live in architecture-specific config files,
- benchmark code is shared,
- results and notes are separated by GPU.

```text
triton-gemm-learning/
│
├── README.md
├── requirements.txt
│
├── kernels/
│   │
│   ├── common/
│   │   ├── v00_baseline.py
│   │   ├── v01_grouped_ordering.py
│   │   ├── v02_tuned.py
│   │   ├── v03_autotuned.py
│   │   └── v04_persistent.py
│   │
│   ├── a100/
│   │   ├── configs.py
│   │   └── experiments.py
│   │
│   ├── gb10/
│   │   ├── configs.py
│   │   ├── v05_tensor_descriptor.py
│   │   ├── v06_warp_specialized.py
│   │   └── low_precision.py
│   │
│   └── inference/
│       ├── fused_linear_activation.py
│       ├── gated_mlp.py
│       └── grouped_gemm.py
│
├── cuda_labs/
│   ├── smem_gemm/
│   └── tensorcore_pipeline/
│
├── benchmarks/
│   ├── bench_gemm.py
│   ├── shapes.py
│   └── plot_results.py
│
├── configs/
│   ├── a100.py
│   └── gb10.py
│
├── scripts/
│   ├── run_a100.sh
│   ├── run_gb10.sh
│   └── profile_ncu.sh
│
├── results/
│   ├── a100/
│   └── gb10/
│
└── notes/
    ├── a100_findings.md
    ├── gb10_findings.md
    ├── architecture_comparison.md
    └── triton_vs_cuda.md
```

## 2.1 Why keep early kernels in `common/`?

The early stages are intentionally meant to be **the same algorithm running on two architectures**.

For example:

```text
kernels/common/v02_tuned.py
        │
        ├── A100 / SM80
        │     └── A100 tuning configuration
        │
        └── GB10 / SM121
              └── GB10 tuning configuration
```

This gives you a controlled experiment.

If you instead immediately create:

```text
kernels/a100/v02.py
kernels/gb10/v02.py
```

the two implementations can gradually drift apart.

Then if performance differs, you may no longer know whether the difference comes from:

- the architecture,
- the compiler,
- the tuning configuration,
- or source-code changes.

Keeping V0–V4 shared lets you hold the **algorithm constant** while changing the **hardware target and tuning parameters**.

---

## 2.2 Separate kernel logic from architecture tuning

Avoid scattering checks such as:

```python
if is_a100:
    BLOCK_M = 128
    BLOCK_N = 128
elif is_gb10:
    BLOCK_M = 64
    BLOCK_N = 128
```

throughout the kernel source.

Instead, separate:

```text
kernel algorithm
        +
architecture-specific tuning policy
```

For example:

```python
# configs/a100.py

A100_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
        num_warps=4,
        num_stages=3,
    ),
]
```

and:

```python
# configs/gb10.py

GB10_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=8,
        num_stages=4,
    ),
]
```

Both can drive the same kernel:

```text
kernels/common/v03_autotuned.py
```

Conceptually:

```text
                  shared algorithm
                        │
                        ▼
                 Triton kernel
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
        A100 configs           GB10 configs
             │                     │
             ▼                     ▼
      SM80 compilation       SM121 compilation
             │                     │
             ▼                     ▼
           A100                  GB10
```

This structure makes architecture comparisons much cleaner.

---

## 2.3 When should a kernel leave `common/`?

Use this rule:

> If the same algorithm can reasonably execute on both GPUs, keep it in `common/`.  
> If the implementation fundamentally depends on an architecture-specific feature, place it under that architecture.

### Good candidates for `kernels/common/`

```text
v00_baseline.py
v01_grouped_ordering.py
v02_tuned.py
v03_autotuned.py
v04_persistent.py
```

These are valuable specifically because you can compare the same source across A100 and GB10.

### Good candidates for `kernels/gb10/`

Examples:

```text
v05_tensor_descriptor.py
v06_warp_specialized.py
low_precision.py
```

when they rely on newer Blackwell-oriented mechanisms or software paths that are not part of the A100 study.

### Good candidates for `kernels/a100/`

This directory will likely be smaller because Triton abstracts away much of the Ampere-specific machinery.

Use it for things such as:

```text
Ampere-specific code-generation experiments
A100-only tuning experiments
comparison kernels that intentionally target SM80 behavior
```

Most explicit study of:

```text
cp.async
ldmatrix
mma.sync
```

should live under `cuda_labs/`, because these concepts are below the normal Triton abstraction.

---

## 2.4 Why this structure is especially useful with Triton

One of the most valuable experiments in this project is:

```python
acc += tl.dot(a, b)
```

compiled for two architectures.

Conceptually:

```text
same Triton source
        │
        ├── Triton compiler → SM80 path → A100
        │
        └── Triton compiler → SM121 path → GB10
```

This lets you study how the compiler and hardware differ while keeping the programming model constant.

That is one of Triton's biggest advantages for a cross-architecture learning project.

---

## 2.5 Think of the repository as three layers

### Layer 1 — shared algorithm

```text
kernels/common/
```

Contains the architecture-neutral Triton implementation.

### Layer 2 — architecture-specific tuning

```text
configs/a100.py
configs/gb10.py
```

Contains:

```text
tile sizes
num_warps
num_stages
autotuning search spaces
scheduler parameters
```

### Layer 3 — architecture-specific mechanisms

```text
kernels/a100/
kernels/gb10/
cuda_labs/
```

Contains experiments that intentionally depend on lower-level or generation-specific features.

This separation is useful beyond this project; it is similar to how production kernel libraries separate:

```text
operation
    ↓
architecture policy
    ↓
hardware-specific implementation
```

---

# 3. Benchmarking Setup

Before optimizing anything, build a reliable benchmark harness.

## 3.1 Correctness

Compare against PyTorch:

```python
ref = torch.matmul(a, b)
out = triton_matmul(a, b)

torch.testing.assert_close(
    out,
    ref,
    atol=...,
    rtol=...,
)
```

Use tolerances appropriate for the datatype.

Test:

- square matrices,
- rectangular matrices,
- non-multiples of block sizes,
- small dimensions,
- large dimensions,
- odd dimensions.

Correctness should be tested separately from benchmarking.

---

## 3.2 Reference implementations

Compare against at least:

- `torch.matmul`,
- Triton's official matmul tutorial implementation,
- optionally cuBLAS/cuBLASLt through PyTorch or another benchmark path.

Do not obsess over beating cuBLAS. The reference is primarily there to tell you whether your implementation is in the right performance regime.

---

## 3.3 Shape suite

Do not benchmark only `8192 × 8192 × 8192`.

Use multiple regimes.

### Large square GEMM

```text
2048 × 2048 × 2048
4096 × 4096 × 4096
8192 × 8192 × 8192
```

### Transformer-style GEMMs

Examples:

```text
M = batch × sequence
K = hidden dimension
N = projection dimension
```

Representative shapes might include:

```text
M = 128,   N = 4096,  K = 4096
M = 512,   N = 4096,  K = 4096
M = 2048,  N = 4096,  K = 4096
M = 4096,  N = 11008, K = 4096
M = 4096,  N = 4096,  K = 11008
```

The exact set is less important than including:

- small-M inference-like GEMMs,
- balanced square GEMMs,
- tall-skinny or wide GEMMs.

---

## 3.4 Metrics

For every kernel version, record:

```text
GPU
dtype
M
N
K
BLOCK_M
BLOCK_N
BLOCK_K
num_warps
num_stages
latency_us
TFLOP/s
reference_TFLOP/s
relative_performance
```

When profiling with Nsight Compute, also record:

```text
registers/thread
shared memory / block
occupancy
DRAM throughput
L2 throughput
Tensor Core utilization
SM utilization
warp stall breakdown
```

---

# 4. Phase I — A100: Triton GEMM Fundamentals

The purpose of this phase is to understand the performance model before moving to Blackwell.

---

## Stage A0 — Understand the Triton execution model

Before optimizing GEMM, be comfortable with:

```python
@triton.jit
def kernel(...):
    pid = tl.program_id(0)
```

Understand the relationship between:

```text
Triton program instance
    ↕
CUDA thread block / CTA conceptually

program_id
    ↕
which output tile this program computes
```

You should understand:

- program IDs,
- block pointers or explicit pointer arithmetic,
- masks,
- `tl.load`,
- `tl.store`,
- broadcasting,
- reductions,
- `tl.arange`,
- `tl.dot`.

### Deliverable

Write a tiny vector addition and a tiled transpose before starting GEMM.

---

# 5. Stage A1 — Simple Blocked GEMM

Implement a straightforward blocked GEMM.

Conceptually:

```text
C[M, N] = A[M, K] × B[K, N]
```

Each Triton program computes one tile:

```text
C[m:m+BM, n:n+BN]
```

and loops through K in chunks of `BLOCK_K`.

Pseudo-structure:

```python
pid_m = ...
pid_n = ...

acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

for k in range(0, K, BLOCK_K):
    a = tl.load(...)
    b = tl.load(...)
    acc += tl.dot(a, b)

tl.store(...)
```

### Questions to answer

- Why does tiling improve locality?
- What data gets reused inside one program?
- Why is `BLOCK_K` different from `BLOCK_M` and `BLOCK_N`?
- Why is the accumulator commonly FP32 even when inputs are FP16/BF16?
- Which dimensions affect register pressure?

### Deliverable

`kernels/common/v00_baseline.py`

Include benchmark results against `torch.matmul`.

---

# 6. Stage A2 — Tile Shape Exploration

Sweep:

```text
BLOCK_M ∈ {32, 64, 128}
BLOCK_N ∈ {32, 64, 128, 256}
BLOCK_K ∈ {16, 32, 64}
```

Do not test every Cartesian-product combination blindly. Start with plausible configurations and use results to narrow the search.

Example configurations:

```text
64  × 64  × 32
64  × 128 × 32
128 × 64  × 32
128 × 128 × 32
128 × 256 × 32
```

### Learn

How tile shape controls:

```text
larger tiles
    ↓
more data reuse
    ↓
potentially higher arithmetic intensity
```

but also:

```text
larger tiles
    ↓
larger accumulators
    ↓
more registers
    ↓
possibly lower occupancy
```

### Goal

Be able to explain why:

```text
"larger tile"
```

does **not** automatically mean:

```text
"faster GEMM"
```

---

# 7. Stage A3 — L2-Aware Tile Ordering

Implement grouped program ordering similar to Triton's matmul tutorial.

Instead of processing output tiles in simple row-major order, group nearby tiles so consecutive programs reuse nearby regions of A or B.

Conceptually compare:

```text
naive ordering

(0,0) (0,1) (0,2) ...
(1,0) (1,1) (1,2) ...
```

against grouped ordering.

### Measure

- latency,
- L2 hit behavior,
- performance changes across large matrix sizes.

### Questions

- Which operand benefits from grouping?
- When does L2-aware ordering help?
- When does it matter less?

### Deliverable

`kernels/common/v01_grouped_ordering.py`

---

# 8. Stage A4 — `num_warps`

Sweep:

```python
num_warps=2
num_warps=4
num_warps=8
```

for multiple tile sizes.

### Learn

More warps can increase available parallelism, but can also affect:

- register allocation,
- occupancy,
- scheduling overhead,
- instruction-level behavior.

Do not record only performance.

Profile configurations where performance changes substantially.

### Deliverable

A table showing:

```text
tile shape
num_warps
registers/thread
occupancy
TFLOP/s
```

Then explain the result.

---

# 9. Stage A5 — `num_stages`

Sweep software-pipeline depth:

```python
num_stages=2
num_stages=3
num_stages=4
num_stages=5
```

### Mental model

Without enough overlap:

```text
load
wait
compute
load
wait
compute
```

With effective pipelining:

```text
load next data
      ↘
       compute current data
```

Triton handles much of the low-level scheduling, but you still need to reason about the tradeoff:

```text
more stages
    ↓
more overlap
```

versus:

```text
more stages
    ↓
more buffering/resources
    ↓
possibly lower occupancy
```

### Deliverable

For one representative GEMM, explain why the best stage count is best.

---

# 10. Stage A6 — Autotuning

Introduce:

```python
@triton.autotune(...)
```

Tune over:

```text
BLOCK_M
BLOCK_N
BLOCK_K
num_warps
num_stages
```

Key the tuning cache using relevant matrix dimensions.

### Important lesson

Autotuning is not a substitute for understanding.

Before adding it, you should already have intuition about why the candidate configurations are reasonable.

### Deliverable

`kernels/common/v03_autotuned.py`

Generate a table showing the selected configuration for several matrix shapes.

Ask:

> Why does the autotuner choose different configurations for different shapes?

---

# 11. Stage A7 — Inspect What Triton Generated

This is a required stage.

Do not treat Triton as a black box.

For selected kernels, inspect:

```text
Triton source
    ↓
Triton IR / TTIR
    ↓
lowered GPU IR
    ↓
PTX
    ↓
SASS
```

You do not need to become an assembly expert.

The goal is to connect high-level operations such as:

```python
tl.load(...)
tl.dot(...)
```

to actual GPU behavior.

### Questions

For `tl.dot`:

- Did the compiler use Tensor Core instructions?
- What datatype path is used?
- How is the K loop lowered?

For different `num_stages`:

- Does the generated scheduling change?
- Is data movement overlapped differently?

For different tile sizes:

- Does register usage increase as expected?

### Deliverable

`notes/a100_codegen.md`

Include several small source → PTX/SASS observations.

---

# 12. Stage A8 — Persistent GEMM

Now change the scheduling model.

Traditional approach:

```text
program 0 → tile 0
program 1 → tile 1
program 2 → tile 2
...
```

Persistent approach:

```text
worker 0 → tile 0 → tile P → tile 2P → ...
worker 1 → tile 1 → tile P+1 → ...
...
```

where the number of persistent workers is related to the available SMs.

### Learn

- why persistent kernels exist,
- reduced scheduling/launch overhead,
- better control over tile assignment,
- why persistent scheduling is relevant to modern GEMMs and inference workloads.

### Questions

- Does persistence help every matrix shape?
- Does it help more at certain problem sizes?
- How sensitive is it to load balancing?

### Deliverable

`kernels/common/v04_persistent.py`

---

# 13. Stage A9 — Small CUDA Micro-Labs

Triton remains the main language, but complete two short CUDA exercises so the abstraction does not become a black box.

## CUDA Lab 1 — Shared-memory tiled GEMM

Implement:

```text
GMEM
 ↓
shared memory
 ↓
register computation
 ↓
GMEM
```

Understand:

- `blockIdx`,
- `threadIdx`,
- `__shared__`,
- `__syncthreads`,
- bank conflicts,
- coalescing,
- register tiling.

The goal is understanding, not cuBLAS-level performance.

---

## CUDA Lab 2 — Tensor Core / asynchronous pipeline

Build or carefully study one small example involving concepts such as:

```text
asynchronous global→shared copy
        ↓
shared-memory layout
        ↓
Tensor Core MMA
        ↓
register accumulator
```

On Ampere this may expose mechanisms such as:

```text
cp.async
ldmatrix
mma.sync
```

You do not need to hand-optimize the entire kernel.

### Goal

After this exercise, when you see:

```python
acc += tl.dot(a, b)
```

you should understand that Triton still has to eventually solve problems such as:

- operand layout,
- shared-memory staging,
- Tensor Core instruction selection,
- pipeline scheduling.

---

# 14. A100 Exit Criteria

Move to GB10 when you can confidently explain:

- CTA/program-level tiling,
- K blocking,
- arithmetic intensity,
- register tiling,
- L2 locality,
- `num_warps`,
- `num_stages`,
- Tensor Core GEMM,
- register pressure,
- occupancy,
- autotuning,
- persistent scheduling,
- basic Nsight Compute interpretation.

You do **not** need to reach exactly 100% of cuBLAS.

A reasonable target is:

> The kernel is clearly in the high-performance regime, and I understand the remaining bottlenecks.

---

# 15. Phase II — GB10 / SM121: Blackwell Retuning

The first GB10 experiment should be extremely simple:

> Run the same shared Triton source code from `kernels/common/` unchanged.

This is why the repository does not immediately fork all kernels into separate A100 and GB10 copies.

At this stage, the controlled variables should be:

```text
same algorithm
same source file
same benchmark shapes
```

while the independent variables are:

```text
GPU architecture
compiler lowering
tuning configuration
```


Do **not** autotune immediately.

Record the performance of the A100-selected configuration on GB10.

Then rerun the tuning search.

This gives you an architecture-transfer experiment.

---

# 16. Stage B1 — Cross-Architecture Comparison

For each representative matrix shape, record:

```text
A100 best config
GB10 performance using A100 config
GB10 best config
```

Compare:

```text
BLOCK_M
BLOCK_N
BLOCK_K
num_warps
num_stages
```

### Questions

- Which parameters transfer?
- Which parameters change?
- Does register pressure behave differently?
- Does the preferred pipeline depth change?
- Do small-M and large-M GEMMs behave differently?
- Does the compiler generate different low-level instructions?

### Deliverable

`notes/architecture_comparison.md`

This should become one of the most important documents in the project.

For each comparison, explicitly record whether it uses:

```text
same source + same config
same source + architecture-specific config
different architecture-specific source
```

This prevents accidental apples-to-oranges conclusions.

---

# 17. Stage B2 — Tensor Descriptors / Modern Data Movement

Study Triton's tensor-descriptor APIs and any TMA-backed paths available for your current Triton + GB10 stack.

Conceptually, compare ordinary pointer arithmetic:

```python
ptrs = base + offsets
x = tl.load(ptrs)
```

against descriptor-based multidimensional data movement.

The learning goal is not merely syntax.

Understand why newer GPUs increasingly use hardware-assisted bulk multidimensional transfers.

### Questions

- Which address-generation work is reduced?
- How does the pipeline change?
- How does descriptor-based loading interact with tiling?
- Which layouts are naturally supported?
- What resource costs appear?

### Deliverable

`kernels/gb10/v05_tensor_descriptor.py`

Compare against the previous implementation on GB10.

---

# 18. Stage B3 — Persistent GEMM on Blackwell

Revisit the persistent kernel.

Do not assume the A100 persistent schedule is optimal.

Retune:

- number of persistent workers,
- tile size,
- warp count,
- stage count,
- traversal order.

### Questions

- Does persistent scheduling help more or less on GB10?
- How do different matrix shapes affect load balance?
- Does the ideal tile scheduler change?

---

# 19. Stage B4 — Warp Specialization

If supported by the Triton path you are using, experiment with warp-specialized execution.

Conceptual model:

```text
warp group / warp(s)
    ├── data movement
    ├── compute
    └── epilogue / store work
```

The important idea is that different warps can take specialized roles instead of every warp repeatedly performing the same full sequence.

### Learn

- producer/consumer pipelines,
- role specialization,
- synchronization,
- latency hiding,
- why modern Hopper/Blackwell GEMMs increasingly use this style.

### Deliverable

`kernels/gb10/v06_warp_specialized.py`

If a specific API is unavailable or unstable on GB10, study an existing implementation and document the architecture instead of forcing the experiment.

---

# 20. Stage B5 — Low-Precision GEMM

After BF16/FP16 GEMM is understood, explore Blackwell-relevant low-precision paths supported on GB10 and your software stack.

Potential topics:

- FP8,
- FP4/NVFP4-style formats,
- block scaling,
- quantized matrix layouts,
- scale application,
- accumulation datatype,
- dequantization/fused epilogue behavior.

The exact datatype should depend on current hardware/software support.

### Why this matters

Modern inference performance increasingly depends on:

```text
lower precision
    +
efficient scaling
    +
specialized Tensor Core paths
```

rather than only BF16 GEMM.

### Deliverable

One quantized or block-scaled GEMM experiment with:

- correctness analysis,
- throughput,
- memory reduction,
- error/accuracy discussion.

---

# 21. Phase III — Inference-Oriented Extensions

Once GEMM itself is understood, stop optimizing isolated matrix multiplication for a while.

Use what you learned to build kernels that look more like inference workloads.

---

## Project C1 — Fused Linear + Activation

Implement:

```text
Y = activation(XW + b)
```

Possible activations:

- GELU,
- SiLU.

Compare:

```text
separate GEMM + activation kernels
```

against:

```text
fused Triton epilogue
```

Measure:

- latency,
- eliminated memory traffic,
- improvement at different M values.

---

## Project C2 — Gated MLP

Implement a simplified gated FFN:

```text
gate = XW_gate
up   = XW_up

Y = SiLU(gate) * up
```

Potentially fuse portions of the epilogue.

This directly connects GEMM optimization to transformer inference.

---

## Project C3 — Grouped GEMM / MoE

Study grouped GEMM behavior for variable expert workloads.

Key topics:

- variable matrix shapes,
- load imbalance,
- scheduling,
- tile assignment,
- persistent workers.

This is an excellent next step because it forces you to think beyond one dense GEMM.

---

# 22. Profiling Curriculum

Do not open Nsight Compute and stare at hundreds of counters.

Use a hypothesis-driven workflow.

---

## Step 1 — Is the kernel obviously memory-bound or compute-bound?

Estimate arithmetic intensity.

Compare:

```text
achieved FLOP/s
```

and:

```text
achieved memory bandwidth
```

against hardware limits.

---

## Step 2 — Check utilization

Look at:

- SM utilization,
- Tensor Core utilization,
- memory throughput.

---

## Step 3 — Check occupancy and resources

Inspect:

- registers/thread,
- shared-memory usage,
- active warps,
- theoretical/achieved occupancy.

---

## Step 4 — Inspect stalls

Ask which stalls dominate.

Examples:

```text
waiting on memory
dependency stalls
barrier/synchronization stalls
instruction issue limitations
```

Then form a concrete hypothesis.

Example:

> Increasing `BLOCK_N` improved reuse but raised accumulator register pressure enough to reduce occupancy, so total throughput fell.

That is the type of conclusion this project should produce.

---

# 23. Suggested Timeline

This is a flexible sequence rather than a strict deadline.

## Week 1 — Triton fundamentals

- Triton execution model
- basic GEMM
- correctness harness
- benchmark harness

## Week 2 — A100 optimization

- tile-size sweeps
- grouped ordering
- `num_warps`
- `num_stages`

## Week 3 — A100 advanced work

- autotuning
- PTX/SASS inspection
- Nsight Compute analysis
- persistent GEMM

## Week 4 — CUDA micro-labs

- shared-memory tiled GEMM
- Tensor Core/asynchronous pipeline study

## Week 5 — GB10 migration

- run A100 kernel unchanged
- retune
- architecture comparison
- inspect generated code

## Week 6 — Modern Blackwell-oriented features

- tensor descriptors / modern data movement
- persistent scheduling
- warp-specialization experiments

## Week 7 — Low precision

- FP8/FP4-style experiment
- scaling/quantization considerations

## Week 8 — Inference capstone

Choose one:

- fused gated MLP,
- grouped GEMM / MoE,
- specialized inference GEMM suite.

Write the final performance report.

---

# 24. Final Project Deliverables

By the end, the repository should contain more than fast code.

## Code

At minimum:

```text
basic Triton GEMM
optimized A100 GEMM
autotuned GEMM
persistent GEMM
GB10-retuned GEMM
one modern Blackwell-oriented variant
one inference-oriented fused kernel
```

---

## Performance data

CSV or JSON data containing:

```text
hardware
shape
dtype
kernel version
configuration
latency
TFLOP/s
reference performance
```

---

## Plots

Useful plots include:

### Performance progression

```text
V0 → V1 → V2 → ... → final
```

### Triton vs reference

Across matrix shapes.

### A100 vs GB10

Show:

```text
same source
same configuration
architecture-specific tuned configuration
```

### Tile selection

Show which configurations autotuning selects by shape.

---

## Technical write-up

The final report should answer:

1. What made the baseline slow?
2. Which optimizations gave the largest gains?
3. Why?
4. Which A100 optimizations transferred directly to GB10?
5. Which required retuning?
6. What changed in generated code?
7. What did Triton abstract away?
8. Which concepts required looking below Triton?
9. What would still need to change for a datacenter Blackwell GPU such as B200?

---

# 25. Stretch Goal — Preparing for B200 / SM100

GB10 is Blackwell, but it is not the same execution target as datacenter B200/SM100.

Use the final stage of the project to study SM100-specific concepts even if you cannot benchmark them locally.

Topics to read about include:

```text
TCGen05
Tensor Memory (TMEM)
SM100 Tensor Core pipelines
1-CTA vs multi-CTA MMA organization
cluster-level execution
multicast
SM100 persistent scheduling
```

The purpose is to separate:

```text
general GEMM knowledge
```

from:

```text
GB10 / SM121-specific implementation details
```

and:

```text
B200 / SM100-specific implementation details
```

If B200 access becomes available later, port only after the GB10 kernel and benchmark harness are already mature.

---

# 26. Triton vs CUDA: What Changes in the Learning Process?

This project intentionally uses Triton because it lets you reach the interesting performance questions much faster.

However, Triton and CUDA teach different layers of the system.

---

## 26.1 The same GEMM idea at two abstraction levels

In Triton, a matrix multiplication inner loop may conceptually look like:

```python
a = tl.load(a_ptrs)
b = tl.load(b_ptrs)

acc += tl.dot(a, b)
```

This expresses:

> Load two matrix tiles and multiply them efficiently.

In lower-level CUDA/PTX, implementing the same operation may require reasoning about:

```text
global-memory transactions
shared-memory staging
shared-memory layouts
asynchronous copies
barriers
warp-level operand layouts
Tensor Core fragment layouts
MMA instructions
register allocation
double/triple buffering
```

Triton asks the compiler to solve much of this machinery for you.

---

## 26.2 What Triton abstracts away

Depending on the kernel and compiler path, Triton may abstract much of:

### Thread-level indexing

Instead of manually mapping:

```cpp
threadIdx.x
threadIdx.y
blockIdx.x
blockIdx.y
```

you usually express work in terms of:

```python
tl.program_id(...)
tl.arange(...)
```

You operate on blocks of values instead of individual scalar threads.

---

### Vectorized memory operations

CUDA often requires explicit care around:

```text
load width
alignment
coalescing
vector types
```

Triton's block-oriented programming model allows the compiler to infer many efficient memory operations from tensor-shaped accesses.

You still need to design a good access pattern, but you write less low-level address-manipulation code.

---

### Tensor Core instruction selection

In CUDA/PTX, you may explicitly work with operations such as:

```text
mma.sync
wgmma
tcgen05
```

depending on architecture.

In Triton, much of this may start from:

```python
tl.dot(...)
```

The compiler selects the appropriate lowering when possible.

---

### Operand-fragment layout

Raw Tensor Core programming requires understanding how matrix fragments map across lanes and registers.

Triton usually lets you reason in terms of logical tiles rather than manually distributing each fragment across threads.

---

### Software pipelining mechanics

In lower-level CUDA, implementing overlap may require:

```text
async copies
buffer selection
barriers
producer/consumer synchronization
stage bookkeeping
```

Triton can expose pipeline depth through compact knobs such as:

```python
num_stages
```

or higher-level loop/scheduling constructs.

The compiler handles much of the actual transformation.

---

### Autotuning infrastructure

Instead of manually compiling dozens of kernel variants and writing your own selector, Triton provides:

```python
@triton.autotune(...)
```

This makes architecture- and shape-specific tuning significantly easier.

---

# 27. Why Triton Makes Learning Faster

The biggest advantage is iteration speed.

Suppose you want to test:

```text
BLOCK_M = 128 → 64
BLOCK_N = 128 → 256
num_warps = 4 → 8
num_stages = 3 → 5
```

In Triton, these are often simple configuration changes.

In hand-written CUDA, changing tile geometry can require rewriting:

- thread mappings,
- shared-memory layouts,
- fragment mappings,
- copy loops,
- synchronization,
- MMA invocation structure.

That difference means Triton allows you to spend more time on:

```text
"What is the performance hypothesis?"
```

and less time on:

```text
"Why is my warp-fragment indexing wrong?"
```

This is especially valuable early in the learning process.

---

# 28. What Triton Does NOT Abstract Away

Triton does not eliminate performance engineering.

You still need to understand:

```text
tile size
data reuse
arithmetic intensity
memory hierarchy
register pressure
occupancy
parallelism
cache locality
pipeline depth
load balancing
persistent scheduling
matrix shape
datatype
```

A poorly designed Triton kernel can still be extremely slow.

Triton removes a large amount of implementation complexity, not the need for architecture awareness.

---

# 29. What You Risk Missing If You Learn Only Triton

The primary risk is developing a mental model such as:

```python
tl.dot(a, b)
```

means:

```text
"the GPU multiplies these blocks"
```

without understanding what must happen underneath.

For serious kernel and inference engineering, eventually you should understand concepts such as:

```text
shared-memory banks
warp-level execution
Tensor Core instruction shapes
operand layouts
async copies
barriers
register allocation
pipeline hazards
PTX
SASS
```

This is why the curriculum includes:

- PTX/SASS inspection,
- Nsight Compute,
- two small CUDA micro-labs.

You do not need to implement every production kernel in CUDA to understand these mechanisms.

---

# 30. When CUDA Becomes Necessary

Triton is an excellent default when:

- the operation maps naturally to tensor blocks,
- rapid iteration matters,
- you are experimenting with fusion,
- you need shape-specialized kernels,
- the compiler exposes the architecture features you need.

CUDA/CuTe/CUTLASS becomes more important when:

- you need exact control over hardware instructions,
- you need an architecture feature Triton does not expose,
- compiler-generated scheduling is not good enough,
- you need very unusual synchronization,
- you need precise control over shared-memory/register layouts,
- you are debugging compiler code generation itself,
- you are implementing the lowest-level building block for other frameworks.

---

# 31. Recommended Mental Model

Do not think:

```text
Triton instead of CUDA
```

Think:

```text
Triton
  ↓
rapidly learn and experiment with kernel algorithms
  ↓
inspect generated GPU code
  ↓
drop into CUDA/PTX only when necessary
```

The ideal skill stack for modern inference work is:

```text
PyTorch / model-level understanding
            ↓
Triton kernel development
            ↓
GPU architecture + profiling
            ↓
CUDA / CuTe / PTX when deeper control is required
```

That gives you both productivity and depth.

---

# 32. Final Success Criteria

The project is complete when you can take a new GEMM-like operator and systematically answer:

### Algorithm

- What data should each program/CTA own?
- What gets reused?
- What tile shape makes sense?

### Memory

- Which data movement dominates?
- Is access coalesced?
- Is L2 reuse effective?
- Is shared-memory staging useful?

### Compute

- Are Tensor Cores being used?
- Is the kernel compute-bound or memory-bound?
- Is there enough independent work?

### Resources

- How many registers are used?
- What limits occupancy?
- Does a larger tile improve reuse but damage residency?

### Pipeline

- Are loads and compute overlapped?
- Is `num_stages` appropriate?
- Would persistent scheduling help?

### Architecture

- Why does the best A100 configuration differ from the best GB10 configuration?
- Which behavior comes from Triton's compiler?
- Which behavior comes from the hardware?

### Tooling

- Can you use Nsight Compute to identify the bottleneck?
- Can you inspect generated PTX/SASS when Triton's behavior is surprising?

If you can answer those questions from measurements rather than guesses, the project has achieved its main goal.

---

# References / Starting Points

- Huy Nguyen, **hopper-gemm-101**  
  https://github.com/HuyNguyen-hust/hopper-gemm-101

- RwTarpit, **FP32 Matmul Optimization on A100**  
  https://rwtarpit.github.io/blog/posts/2026-07-25-fp32_matmul_A100.html

- Triton documentation and tutorials  
  https://triton-lang.org/

- NVIDIA CUTLASS  
  https://github.com/NVIDIA/cutlass

Use these references to understand techniques, but keep your own repository centered on **measured experiments and explanations**, not copying an existing implementation line by line.
