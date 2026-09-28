# Architecture Comparison: A100 (SM80) vs GB10 (SM121)

## Hardware Overview

| Specification | NVIDIA A100 (SXM4) | NVIDIA GB10 |
| :--- | :--- | :--- |
| **Architecture** | Ampere (SM80) | Blackwell (SM121) |
| **SM Count** | 108 | 48 |
| **Host CPU** | x86_64 | 20-Core ARM Grace (`aarch64`) |
| **Memory** | 80 GB HBM2e (2.0 TB/s) | 128 GB LPDDR5X (Unified Coherent) |
| **Max Shared Mem / SM** | 164 KB | 100 KB |

## Tuning Comparisons

| Kernel & Shape | A100 Best Config | GB10 (A100 Config) | GB10 Best Config | Comparison Type |
| :--- | :--- | :--- | :--- | :--- |
| **2048 x 2048 x 2048** | 128x256x64, w8, s4 | TBD | TBD | same source + different config |
| **4096 x 4096 x 4096** | 128x256x64, w8, s4 | TBD | TBD | same source + different config |
