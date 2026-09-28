# Blackwell GEMM Benchmark & Profiling Setup

This directory contains a complete CUDA test and profiling suite tailored for the **NVIDIA GB10 (Blackwell architecture)** nodes on the Scholar cluster.

---

## Files

- [gemm_test.cu](file:///home/vo43/GEMM/GB10/gemm_test.cu): CUDA C++ matrix multiplication benchmark (shared-memory tiled GEMM) with correctness verification, latency timing, and TFLOPS calculation.
- [submit_blackwell.slurm](file:///home/vo43/GEMM/GB10/submit_blackwell.slurm): Slurm batch script to compile natively on the Blackwell node, execute the benchmark, and generate an Nsight Systems (`nsys`) profile.
- [run_interactive.sh](file:///home/vo43/GEMM/GB10/run_interactive.sh): Quick helper script to launch an interactive bash session (`srun`) on the `spark-interactive` partition.

---

## 1. Submitting via Slurm Batch Job

Submit the job to the Blackwell `spark-batch` partition:

```bash
cd /home/vo43/GEMM/GB10
sbatch submit_blackwell.slurm
```

### Check Job Status
```bash
squeue -u $USER
```

### Outputs
Once finished, you will find all logs and reports inside the `output/` directory:
- `output/blackwell_gemm_<jobid>.out`: Output logs containing GPU device properties, benchmark results (TFLOPS), and kernel timing summary.
- `output/blackwell_gemm_<jobid>.err`: Slurm stderr log.
- `output/gemm_nsys_<jobid>.nsys-rep`: Complete Nsight Systems trace file (viewable in the NVIDIA Nsight Systems GUI).

---

## 2. Running Interactively

To develop, compile, and test kernels iteratively without queueing batch scripts:

```bash
cd /home/vo43/GEMM/GB10
./run_interactive.sh
```

Once on the compute node:
```bash
# 1. Load Blackwell environment
module --force purge
module use /opt/scholar-dgx-modulefiles/dgx/Core
module load gcc/12.5.0 cuda/13.1.1

# 2. Compile for Blackwell natively
nvcc -O3 -arch=native gemm_test.cu -o gemm_test

# 3. Run
./gemm_test

# 4. Profile with Nsight Systems
nsys profile --stats=true -o my_profile ./gemm_test
```

---

## 3. Key Notes for Kernel Optimization on GB10

1. **Host Architecture (`aarch64`)**: The Spark nodes are ARM Grace CPUs. Compiling on the x86 login node will fail with `Exec format error`. Always compile on the node itself.
2. **CUDA Target**: Use `-arch=native` (or `-arch=sm_121`) with CUDA 13.1.1.
3. **Coherent Memory**: The GB10 features 128 GB coherent unified LPDDR5X memory shared between CPU and GPU.
