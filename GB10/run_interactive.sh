#!/bin/bash
# Helper script to launch an interactive bash session on a Blackwell node
# Usage: ./run_interactive.sh

echo "Launching interactive session on Blackwell (spark-interactive)..."
srun -A scholar \
     -p spark-interactive \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=20 \
     --gres=gpu:1 \
     --time=01:00:00 \
     --pty bash
