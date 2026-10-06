# Gated MLP Inductor code-generation analysis

| Field | Value |
|---|---:|
| Device | NVIDIA GeForce RTX 5050 Laptop GPU |
| PyTorch | 2.12.0+cu130 |
| Input shape | 4 × 128 × 768 |
| Dtype | float16 |
| External wrapper calls | 3 × mm |
| Generated Triton kernels | triton_poi_fused__unsafe_view_mul_silu_0 |
| Compiled wrapper launch sites | 4 |
| Eager profiled launches/invocation | 5.0 |
| Compiled profiled launches/invocation | 4.0 |

## Runtime breakdown

| Median component | Eager | Compiled |
|---|---:|---:|
| Host enqueue | 50.93 µs | 94.93 µs |
| Post-enqueue completion wait | 229.23 µs | 220.66 µs |
| Synchronized total | 278.01 µs | 315.07 µs |
| Synchronized speedup | — | 0.882× |

Inductor leaves all three linear projections as external `mm` calls and emits
one Triton kernel for the SiLU/multiply chain. The profiler therefore observes
five eager launches and four compiled launches per invocation.

For this configuration, the compiled path reduces post-enqueue completion time but increases host enqueue time. The added host-side cost is larger than the completion-time reduction, so synchronized single-call latency regresses. This conclusion is limited to the recorded software, hardware,
dtype, and shape.
