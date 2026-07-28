# Baseline result

This result applies only to the environment and shape recorded below.

| Field | Value |
|---|---:|
| Device | NVIDIA GeForce RTX 5050 Laptop GPU |
| PyTorch | 2.12.0+cu130 |
| Shape | 1024 × 4096 |
| Dtype | float32 |
| Inductor cache | fresh temporary cache |
| Dynamo graphs captured | 1 |
| Maximum absolute error | 1.192e-07 |
| Eager median | 0.2332 ms |
| Compiled first call | 1454.7573 ms |
| Compiled steady-state median | 0.1006 ms |
| Steady-state speedup | 2.319× |
| Estimated break-even | ~10968 calls |

The first call was measured with a fresh temporary cache; it can include tracing, code
generation, compilation, and initial allocations. The speedup compares
steady-state medians after warm-up. The break-even value is an estimate, not a
production SLA.
