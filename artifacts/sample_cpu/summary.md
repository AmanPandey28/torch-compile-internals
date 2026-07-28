# Baseline result

This result applies only to the environment and shape recorded below.

| Field | Value |
|---|---:|
| Device | x86_64 |
| PyTorch | 2.12.0+cu130 |
| Shape | 1024 × 4096 |
| Dtype | float32 |
| Inductor cache | fresh temporary cache |
| Dynamo graphs captured | 1 |
| Maximum absolute error | 3.576e-07 |
| Eager median | 7.5241 ms |
| Compiled first call | 4369.8435 ms |
| Compiled steady-state median | 1.0567 ms |
| Steady-state speedup | 7.121× |
| Estimated break-even | ~676 calls |

The first call was measured with a fresh temporary cache; it can include
tracing, code generation, compilation, and initial allocations. The speedup
compares steady-state medians after warm-up. The break-even value is an
estimate, not a production SLA.
