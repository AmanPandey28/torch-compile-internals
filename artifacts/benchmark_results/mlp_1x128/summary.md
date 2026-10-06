# Benchmark result

This result applies only to the environment and shape recorded below.

| Field | Value |
|---|---:|
| Workload | GatedMLP |
| Device | NVIDIA GeForce RTX 5050 Laptop GPU |
| PyTorch | 2.12.0+cu130 |
| Shape | 1 × 128 × 768 |
| Parameters | 4,718,592 |
| Dtype | float16 |
| Inductor cache | fresh temporary cache |
| Dynamo graphs captured | 1 |
| Maximum absolute error | 3.662e-04 |
| Eager median | 0.0851 ms |
| Compiled first call | 1859.5784 ms |
| Compiled steady-state median | 0.1238 ms |
| Steady-state speedup | 0.688× |
| Estimated break-even | not reached (compiled was not faster) |

The first call was measured with a fresh temporary cache; it can include tracing, code
generation, compilation, and initial allocations. The speedup compares
steady-state medians after warm-up. The break-even value is an estimate, not a
production SLA.
