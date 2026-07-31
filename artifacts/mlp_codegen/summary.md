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
| Host enqueue | 51.21 µs | 94.53 µs |
| Post-enqueue completion wait | 230.55 µs | 222.00 µs |
| Synchronized total | 280.35 µs | 316.98 µs |
| Synchronized speedup | — | 0.884× |

Inductor leaves all three linear projections as external `mm` calls and emits
one Triton kernel for the SiLU/multiply chain. The profiler therefore observes
five eager launches and four compiled launches per invocation.

For this configuration, the compiled path reduces post-enqueue completion time
but increases host enqueue time. The added host-side cost is larger than the
completion-time reduction, so synchronized single-call latency regresses. This
conclusion is limited to the recorded software, hardware, dtype, and shape.
