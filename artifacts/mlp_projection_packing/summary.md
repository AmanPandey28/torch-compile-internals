# Gated MLP projection-packing analysis

The transformation concatenates the gate and up projection weights once during
model conversion. The packed model computes both projections with one linear
operation, splits the result into views, and preserves the down
projection. Parameter count remains 4,718,592.

## Generated execution

| Structural metric | Unpacked | Packed |
|---|---:|---:|
| External GEMM calls in Inductor wrapper | 3 | 2 |
| Triton kernel launches in wrapper | 1 | 1 |
| Compiled CUDA launches per invocation | 4.0 | 3.0 |
| Eager CUDA launches per invocation | 5.0 | 4.0 |

The representative generated wrapper changes from three external GEMM calls to
two while retaining one fused Triton SiLU/multiply kernel. The split is a view
and does not introduce another profiled CUDA launch.

## Compiled latency

| Input shape | Unpacked | Packed | Speedup |
|---|---:|---:|---:|
| 1 × 1 × 768 | 102.23 µs | 87.35 µs | 1.170× |
| 1 × 128 × 768 | 138.03 µs | 125.26 µs | 1.102× |
| 4 × 128 × 768 | 342.73 µs | 329.62 µs | 1.040× |

Packed compiled execution was faster in 3 of 3 measured shapes. The largest observed speedup was 1.170× at shape `1 × 1 × 768`.

Packing did not improve eager latency, and the packed compiled path did not beat unpacked eager execution in these cases. The measured win is specifically packed compiled versus unpacked compiled.

For the representative shape, median host enqueue changed from
120.92 µs to 99.81 µs and post-enqueue completion
wait changed from 217.46 µs to 227.31 µs.
The lower host enqueue time outweighed the longer GPU completion wait for this shape.

All eager and compiled outputs passed float16 tolerance against the unpacked
model; the maximum observed absolute error was 3.662e-04. The result is
specific to the recorded shapes, software stack, and RTX 5050 Laptop GPU.
