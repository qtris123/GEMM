# GB10 (SM121 / Blackwell) Optimization Findings

## Summary
- **Architecture**: 48 SMs, Compute Capability 12.1, 128 GB LPDDR5X unified memory.
- **Key Characteristics**:
  - High capacity unified memory (124 GB usable).
  - Fewer SMs than datacenter B200 or A100 (48 vs 108/144), making persistent grid scheduling and tile load balancing especially critical.
  - Software pipelining: `num_stages=3` or `4` provides effective latency hiding without exhausting shared memory.
