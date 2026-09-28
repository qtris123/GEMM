"""Configuration and hardware specifications for NVIDIA GB10 (Blackwell / SM121)."""

ARCH = "sm121"
NAME = "NVIDIA GB10"

# Hardware parameters (measured on Scholar Spark nodes)
NUM_SMS = 48
WARP_SIZE = 32
MAX_SHARED_MEM_PER_SM_KB = 100
TOTAL_MEMORY_GB = 128.0  # LPDDR5X unified coherent memory

# Tuning search spaces
TILE_M_CANDIDATES = [64, 128, 256]
TILE_N_CANDIDATES = [64, 128, 256]
TILE_K_CANDIDATES = [32, 64, 128]
NUM_WARPS_CANDIDATES = [4, 8]
NUM_STAGES_CANDIDATES = [2, 3, 4]
GROUP_M_DEFAULT = 8
