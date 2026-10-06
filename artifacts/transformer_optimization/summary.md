# Minimal transformer optimization analysis

The workload is a pre-norm causal transformer block with two RMSNorm layers,
multi-head scaled-dot-product attention, residual connections, and a gated MLP.
The optimized block packs Q/K/V and gate/up weights once during model conversion.

## Block-level results

| Shape | Unpacked eager | Unpacked compiled | Packed compiled | Best packed mode | Best latency |
|---|---:|---:|---:|---:|---:|
| 1 × 1 × 768 | 147.41 µs | 183.34 µs | 162.28 µs | max-autotune | 101.59 µs |
| 1 × 128 × 768 | 169.83 µs | 217.00 µs | 185.86 µs | reduce-overhead | 175.95 µs |
| 4 × 128 × 768 | 469.52 µs | 484.15 µs | 475.61 µs | default | 475.61 µs |

Default compiled packing produced speedups of
1.130×, 1.168×, 1.018× across the measured
shapes by reducing the generated external GEMM count from seven to four.

## Compilation boundary

| Shape | Whole block | Attention and MLP regions | Whole-block speedup |
|---|---:|---:|---:|
| 1 × 1 × 768 | 162.28 µs | 227.34 µs | 1.401× |
| 1 × 128 × 768 | 185.86 µs | 262.42 µs | 1.412× |
| 4 × 128 × 768 | 475.61 µs | 475.26 µs | 0.999× |

Whole-block compilation was 1.401× and
1.412× faster for the batch-one cases; the two boundaries were
effectively tied at the largest shape (0.999×).

## Compile-mode matrix

| Packed strategy | 1 × 1 × 768 | 1 × 128 × 768 | 4 × 128 × 768 |
|---|---:|---:|---:|
| default | 162.28 µs | 185.86 µs | 475.61 µs |
| reduce-overhead | 119.42 µs | 175.95 µs | 480.29 µs |
| max-autotune-no-cudagraphs | 163.04 µs | 198.65 µs | 484.75 µs |
| max-autotune | 101.59 µs | 178.84 µs | 483.27 µs |
| shape-padding | 165.65 µs | 188.83 µs | 478.88 µs |

## Generated execution

| Representative structural metric | Unpacked | Packed |
|---|---:|---:|
| External GEMM calls | 7 | 4 |
| Generated Triton launch sites | 4 | 4 |
| Total wrapper launch sites | 11 | 8 |
| Profiled CUDA launches/invocation | 12.0 | 9.0 |

The packed graph replaces three Q/K/V GEMMs with one and replaces two gated-MLP
input GEMMs with one. Inductor also fuses the first residual addition with the
second RMSNorm and fuses the final residual addition into a generated pointwise
kernel.

## RMSNorm representation

| Form | Wrapper launch sites | Profiled launches/invocation | Median latency |
|---|---:|---:|---:|
| Native RMSNorm | 1 | 1.0 | 57.45 µs |
| Decomposed arithmetic | 1 | 1.0 | 54.04 µs |

Both forms lowered to one generated launch, so manually decomposing RMSNorm did
not remove another kernel. All strategies passed numerical comparison with
unpacked eager execution. Results are specific to the recorded software,
shapes, and RTX 5050 Laptop GPU.
