#!/bin/bash
#SBATCH -A scholar
#SBATCH -p spark-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --job-name=blackwell_gemm
#SBATCH --output=output/%x_%j.out
#SBATCH --error=output/%x_%j.err

echo "=========================================================="
echo "Starting Slurm job: $SLURM_JOB_ID on node $(hostname)"
echo "Target Partition  : $SLURM_JOB_PARTITION"
echo "Date              : $(date)"
echo "=========================================================="

# Move to script directory
if [ -f "$SLURM_SUBMIT_DIR/gemm_test.cu" ]; then
    cd "$SLURM_SUBMIT_DIR" || exit 1
elif [ -d "/home/vo43/GEMM/GB10" ]; then
    cd "/home/vo43/GEMM/GB10" || exit 1
fi

# Ensure output directory exists
mkdir -p output

# 1. Clean environment and load Blackwell/ARM64 specific modules
module --force purge
module use /opt/scholar-dgx-modulefiles/dgx/Core
module load gcc/12.5.0 cuda/13.1.1

echo "Loaded modules:"
module list

echo -e "\nCompiler & Profiler info:"
command -v nvcc
nvcc --version
command -v nsys

# 2. Compile natively on ARM64 host for Blackwell GPU (sm_121)
echo -e "\nCompiling gemm_test.cu for Blackwell (native)..."
nvcc -O3 -arch=native gemm_test.cu -o gemm_test
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi
echo "Compilation successful!"

# 3. Run the GEMM test binary
echo -e "\n--- Running GEMM Test Binary ---"
./gemm_test

# 4. Profile with NVIDIA Nsight Systems (nsys)
echo -e "\n--- Running Nsight Systems Profiling ---"
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN nsys profile --stats=true -o "output/gemm_nsys_${SLURM_JOB_ID}" ./gemm_test

echo "=========================================================="
echo "Job completed successfully at $(date)"
echo "Nsight report generated: output/gemm_nsys_${SLURM_JOB_ID}.nsys-rep"
echo "=========================================================="
