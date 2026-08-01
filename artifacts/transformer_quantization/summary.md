# Minimal transformer quantization analysis

The packed transformer block is quantized with TorchAO 0.17.0.
Every strategy uses BF16 activations at the block boundary and is compiled with
`max-autotune`; latency is compared with the unquantized packed BF16 block.

| Shape | Strategy | Median | Speedup | Storage reduction | Cosine similarity | INT8 decode fallback |
|---|---|---:|---:|---:|---:|---:|
| 1 × 1 × 768 | bf16 | 100.78 µs | 1.000× | 1.00× | 0.999995 | no |
| 1 × 1 × 768 | int8-weight-only | 118.59 µs | 0.850× | 1.99× | 0.999993 | no |
| 1 × 1 × 768 | int8-dynamic | 109.65 µs | 0.919× | 1.99× | 0.999993 | yes |
| 1 × 1 × 768 | fp8-weight-only | 107.41 µs | 0.938× | 1.99× | 0.999916 | no |
| 1 × 1 × 768 | fp8-dynamic | 115.49 µs | 0.873× | 2.00× | 0.999837 | no |
| 1 × 128 × 768 | bf16 | 180.66 µs | 1.000× | 1.00× | 0.999996 | no |
| 1 × 128 × 768 | int8-weight-only | 9996.52 µs | 0.018× | 1.99× | 0.999995 | no |
| 1 × 128 × 768 | int8-dynamic | 173.87 µs | 1.039× | 1.99× | 0.999992 | no |
| 1 × 128 × 768 | fp8-weight-only | 214.44 µs | 0.842× | 1.99× | 0.999981 | no |
| 1 × 128 × 768 | fp8-dynamic | 172.02 µs | 1.050× | 2.00× | 0.999965 | no |
| 4 × 128 × 768 | bf16 | 471.77 µs | 1.000× | 1.00× | 0.999996 | no |
| 4 × 128 × 768 | int8-weight-only | 39937.19 µs | 0.012× | 1.99× | 0.999995 | no |
| 4 × 128 × 768 | int8-dynamic | 340.93 µs | 1.384× | 1.99× | 0.999992 | no |
| 4 × 128 × 768 | fp8-weight-only | 518.04 µs | 0.911× | 1.99× | 0.999981 | no |
| 4 × 128 × 768 | fp8-dynamic | 314.74 µs | 1.499× | 2.00× | 0.999966 | no |

Dynamic INT8 uses weight-only decode fallback when the flattened token dimension
is at most 16 because the generated integer matrix kernel requires a larger M
dimension. Prefill cases use dynamic activation and weight quantization.

Dynamic INT8 prefill speedups were 1.039×, 1.384×; dynamic FP8 prefill speedups
were 1.050×, 1.499×. None of the quantized paths improved single-token decode
latency. The INT8 weight-only `_weight_int8pack_mm` path was substantially
slower for prefill on this backend, demonstrating why quantization choices
require workload- and hardware-specific measurement.

## Representative generated paths

| Strategy | Model storage | External calls |
|---|---:|---|
| bf16 | 13.50 MiB | `mm` × 4 |
| int8-weight-only | 6.78 MiB | `_weight_int8pack_mm` × 4 |
| int8-dynamic | 6.78 MiB | `_int_mm` × 4 |
| fp8-weight-only | 6.78 MiB | `mm` × 4 |
| fp8-dynamic | 6.75 MiB | `_scaled_mm` × 4 |

All quantized outputs were finite and exceeded cosine similarity 0.99 against
the packed BF16 reference. Storage figures include parameters and quantization
metadata reported by TorchAO. Results are specific to the recorded software,
shapes, and RTX 5050 Laptop GPU.
