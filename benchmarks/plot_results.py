"""plot_results.py: Visualizes GEMM benchmarking results.
Compares achieved TFLOPS across shapes and architectures.
"""

import argparse
import json
import matplotlib.pyplot as plt


def plot_benchmark(json_path: str, out_image: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    device = data.get("device", "GPU")
    kernel = data.get("kernel", "Kernel")
    results = data["results"]

    shapes = [f"{r['M']}x{r['N']}x{r['K']}" for r in results]
    kernel_tflops = [r["tflops"] for r in results]
    ref_tflops = [r["ref_tflops"] for r in results]

    x = range(len(shapes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width/2 for i in x], kernel_tflops, width, label=f"Triton ({kernel})", color="#2ecc71")
    ax.bar([i + width/2 for i in x], ref_tflops, width, label="torch.matmul (cuBLAS)", color="#3498db")

    ax.set_ylabel("Achieved TFLOPS")
    ax.set_title(f"GEMM Performance on {device}")
    ax.set_xticks(x)
    ax.set_xticklabels(shapes, rotation=35, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_image, dpi=300)
    print(f"Saved plot to {out_image}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input benchmark JSON")
    parser.add_argument("--output", type=str, default="benchmark_plot.png", help="Output PNG path")
    args = parser.parse_args()
    plot_benchmark(args.input, args.output)
