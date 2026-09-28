#!/bin/bash
# Slurm script for running on A100 (e.g. Gilbreth, Gautschi, or Anvil)
# Adjust partition and account to match your target A100 cluster

echo "=== Running Triton GEMM Benchmarks on A100 ==="
echo "Node: $(hostname) | Date: $(date)"

cd "$SLURM_SUBMIT_DIR" || cd /home/vo43/GEMM || exit 1
mkdir -p results/a100

export PYTHONPATH="$PWD:$PYTHONPATH"

for kernel in baseline grouped tuned autotuned persistent; do
    echo -e "\n--> Running kernel: $kernel"
    python3 benchmarks/bench_gemm.py --kernel "$kernel" --output "results/a100/${kernel}_results.json"
done
