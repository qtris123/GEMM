#!/bin/bash
#SBATCH -A scholar
#SBATCH -p spark-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=bench_gb10
#SBATCH --output=results/gb10/%x_%j.out
#SBATCH --error=results/gb10/%x_%j.err

echo "=== Running Triton GEMM Benchmarks on GB10 ==="
echo "Node: $(hostname) | Date: $(date)"

cd "$SLURM_SUBMIT_DIR" || cd /home/vo43/GEMM || exit 1
mkdir -p results/gb10

# Load Blackwell modules
module --force purge
module use /opt/scholar-dgx-modulefiles/dgx/Core
module load gcc/12.5.0 cuda/13.1.1

export PYTHONPATH="$PWD:$PYTHONPATH"

# Run benchmark across kernels
for kernel in baseline grouped tuned autotuned persistent; do
    echo -e "\n--> Running kernel: $kernel"
    python3 benchmarks/bench_gemm.py --kernel "$kernel" --output "results/gb10/${kernel}_results.json"
done

echo "Benchmark run complete."
