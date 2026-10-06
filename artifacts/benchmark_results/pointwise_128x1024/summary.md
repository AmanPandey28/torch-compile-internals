# Benchmark result

This result applies only to the environment and shape recorded below.

| Field | Value |
|---|---:|
| Workload | PointwiseReduction |
| Device | NVIDIA GeForce RTX 5050 Laptop GPU |
| PyTorch | 2.12.0+cu130 |
| Shape | 128 × 1024 |
| Parameters | 0 |
| Dtype | float32 |
| Inductor cache | fresh temporary cache |
| Dynamo graphs captured | 1 |
| Maximum absolute error | 1.192e-07 |
| Eager median | 0.0302 ms |
| Compiled first call | 1713.3704 ms |
| Compiled steady-state median | 0.0566 ms |
| Steady-state speedup | 0.534× |
| Estimated break-even | not reached (compiled was not faster) |

The first call was measured with a fresh temporary cache; it can include tracing, code
generation, compilation, and initial allocations. The speedup compares
steady-state medians after warm-up. The break-even value is an estimate, not a
production SLA.
