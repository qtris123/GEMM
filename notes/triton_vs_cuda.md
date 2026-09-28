# Triton vs CUDA: Abstraction and Performance Tradeoffs

## What Triton Abstracts Away
1. **Thread and Warp Indexing**: Logical 2D tiles replace manual thread-to-fragment index calculations.
2. **Tensor Core MMA Fragment Layouts**: `tl.dot(a, b)` maps automatically to architecture-specific MMA instructions (`mma.sync` on Ampere, etc.).
3. **Asynchronous Memory Transfers**: `num_stages` manages `cp.async` double/multi-buffering without manual barrier tracking.

## What You Still Must Tune
1. **Tile Shape (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`)**: Dictates register pressure, occupancy, and arithmetic intensity.
2. **Cache Locality (`GROUP_M`)**: Reorders CTA execution to preserve tile reuse in L2 cache.
3. **Pipeline Stages (`num_stages`)**: Balances memory latency hiding against shared memory footprint.
