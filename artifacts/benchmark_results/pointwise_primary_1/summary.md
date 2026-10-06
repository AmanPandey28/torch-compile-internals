# Benchmark result

This result applies only to the environment and shape recorded below.

| Field | Value |
|---|---:|
| Workload | PointwiseReduction |
| Device | NVIDIA GeForce RTX 5050 Laptop GPU |
| PyTorch | 2.12.0+cu130 |
| Shape | 1024 × 4096 |
| Parameters | 0 |
| Dtype | float32 |
| Inductor cache | fresh temporary cache |
| Dynamo graphs captured | 1 |
| Maximum absolute error | 1.192e-07 |
| Eager median | 0.2349 ms |
| Compiled first call | 4692.3147 ms |
| Compiled steady-state median | 0.1134 ms |
| Steady-state speedup | 2.071× |
| Estimated break-even | ~38625 calls |

The first call was measured with a fresh temporary cache; it can include tracing, code
generation, compilation, and initial allocations. The speedup compares
steady-state medians after warm-up. The break-even value is an estimate, not a
production SLA.
