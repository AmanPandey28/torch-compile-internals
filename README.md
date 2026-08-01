# Torch Compile Internals

A reproducible benchmark and graph-analysis suite for examining PyTorch 2
compiler behavior on synthetic kernels and transformer components.

The current implementation evaluates:

- Dynamo graph capture and full-graph compatibility.
- First-call compilation cost versus steady-state execution.
- Numerical equivalence between eager and compiled execution.
- Data-dependent control flow and graph fragmentation.
- Compiler behavior on a gated transformer MLP.
- Generated Inductor wrappers, external calls, Triton kernels, and launch counts.
- Host enqueue time versus post-enqueue GPU completion time.
- Guard failures, recompilation counts, and dynamic-shape specialization.
- Reversible gated-MLP projection packing with state, graph, and GPU evidence.
- A minimal pre-norm transformer block with RMSNorm, causal attention,
  residual connections, and a gated MLP.
- Whole-block versus regional compilation and compile-mode selection.
- TorchAO INT8 and FP8 quantization with latency, storage, quality, and
  generated-kernel analysis.

All reported measurements include the workload shape, dtype, software version,
hardware target, cache policy, and correctness result.

## Results

### Pointwise reduction

The baseline combines bias addition, SiLU, multiplication, and reduction. Both
CPU and GPU runs use float32 input with shape `[1024, 4096]`, 20 warm-up
iterations, and 100 measured iterations.

| Metric | RTX 5050 Laptop GPU | CPU |
|---|---:|---:|
| Dynamo graphs | 1 | 1 |
| Maximum absolute error | 1.19e-7 | 3.58e-7 |
| Eager median | 0.2332 ms | 7.5241 ms |
| Compiled median | 0.1006 ms | 1.0567 ms |
| Steady-state speedup | 2.32× | 7.12× |
| First compiled call | 1.45 s | 4.37 s |
| Estimated break-even | ~10,968 calls | ~676 calls |

Artifacts:

- GPU: [summary](artifacts/sample_gpu/summary.md), [raw
  result](artifacts/sample_gpu/results.json), [FX
  graph](artifacts/sample_gpu/dynamo_fx_graph.py)
- CPU: [summary](artifacts/sample_cpu/summary.md), [raw
  result](artifacts/sample_cpu/results.json), [FX
  graph](artifacts/sample_cpu/dynamo_fx_graph.py)

### Data-dependent control flow

The graph-break case compares a Python branch on a tensor value with an
equivalent `torch.cond` implementation.

| Result | Python branch | `torch.cond` |
|---|---:|---:|
| Positive and negative outputs verified | yes | yes |
| Graphs captured across both paths | 3 | 1 |
| Compatible with `fullgraph=True` | no | yes |

The Python implementation produces a predicate graph and separate continuation
graphs for each branch. The `torch.cond` implementation represents the
conditional inside one captured graph.

Artifacts: [summary](artifacts/graph_break_case/summary.md), [Python graph
regions](artifacts/graph_break_case/python_if_graphs.py), [`torch.cond`
graph](artifacts/graph_break_case/torch_cond_graph.py)

### Gated transformer MLP

The transformer workload implements:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

The benchmark uses a 4,718,592-parameter MLP, float16 input with shape
`[4, 128, 768]`, hidden dimension 2048, and an RTX 5050 Laptop GPU.

| Metric | Result |
|---|---:|
| Dynamo graphs | 1 |
| Maximum absolute error | 2.44e-4 |
| Eager median | 0.2402 ms |
| Compiled median | 0.2897 ms |
| Steady-state speedup | 0.829× |
| First compiled call | 1.55 s |

Compiled execution was 17% slower in the primary run and 19% slower in a
fresh-process repeat. A separate 500-iteration, alternating-order analysis
described below reproduced the regression while isolating its runtime
components.

Artifacts: [summary](artifacts/sample_mlp_gpu/summary.md), [primary
result](artifacts/sample_mlp_gpu/results.json), [repeat
result](artifacts/sample_mlp_gpu/repeat_results.json), [FX
graph](artifacts/sample_mlp_gpu/dynamo_fx_graph.py)

### Inductor code generation and runtime breakdown

The generated wrapper contains three external `mm` calls and one generated
Triton kernel. Inductor leaves the gate, up, and down projections on external
GEMM paths while fusing SiLU and multiplication into a single in-place kernel.

| Structural result | Eager | Compiled |
|---|---:|---:|
| Kernel launches per invocation | 5 | 4 |
| External GEMM calls | 3 | 3 |
| Pointwise kernels | 2 | 1 |

| Median runtime component | Eager | Compiled |
|---|---:|---:|
| Host enqueue | 51.21 µs | 94.53 µs |
| Post-enqueue completion wait | 230.55 µs | 222.00 µs |
| Synchronized total | 280.35 µs | 316.98 µs |

The compiled path reduces launch count and post-enqueue completion time, but
its additional host enqueue cost is larger than the completion-time reduction.
The resulting synchronized speedup is `0.884×`, or an 11.6% regression, for
this software, hardware, dtype, and shape.

Artifacts: [analysis](artifacts/mlp_codegen/summary.md), [raw
result](artifacts/mlp_codegen/results.json), [sanitized generated
wrapper](artifacts/mlp_codegen/inductor_output_code.py)

### Gated MLP projection packing

The first model transformation concatenates `gate_proj` and `up_proj` weights
once during model conversion. The packed module replaces two equal-size input
projections with one projection producing twice the hidden dimension, then uses
view-based splitting before SiLU and multiplication. The conversion is
reversible and retains the original 4,718,592 parameters.

| Structural metric | Original | Packed |
|---|---:|---:|
| External GEMM calls in generated wrapper | 3 | 2 |
| Compiled CUDA launches, representative shape | 4 | 3 |
| Eager CUDA launches, representative shape | 5 | 4 |

| Input shape | Original compiled | Packed compiled | Speedup |
|---|---:|---:|---:|
| `[1, 1, 768]` | 102.23 µs | 87.35 µs | 1.170× |
| `[1, 128, 768]` | 138.03 µs | 125.26 µs | 1.102× |
| `[4, 128, 768]` | 342.73 µs | 329.62 µs | 1.040× |

Packing improved compiled latency at all three measured shapes and removed one
CUDA launch per invocation. At the representative `[4, 128, 768]` shape, lower
host enqueue time outweighed a longer GPU completion wait. Packing did not
improve eager latency, and packed compiled execution did not beat original eager
execution; the measured result is specifically a comparison between the packed
and original compiled paths.

Artifacts: [analysis](artifacts/mlp_projection_packing/summary.md), [raw
result](artifacts/mlp_projection_packing/results.json), [before/after Dynamo
graphs](artifacts/mlp_projection_packing/dynamo_graphs.py), [original
wrapper](artifacts/mlp_projection_packing/original_inductor.py), [packed
wrapper](artifacts/mlp_projection_packing/packed_inductor.py)

### Minimal transformer block

The block combines two RMSNorm layers, causal scaled-dot-product attention,
residual connections, and a gated MLP at model dimension 768, 12 attention
heads, and hidden dimension 2048. A reversible conversion packs Q/K/V and the
MLP gate/up projections while preserving parameters and outputs.

| Shape | Original eager | Original compiled | Packed compiled | Best packed mode | Best latency |
|---|---:|---:|---:|---|---:|
| `[1, 1, 768]` | 147.41 µs | 183.34 µs | 162.28 µs | `max-autotune` | 101.59 µs |
| `[1, 128, 768]` | 169.83 µs | 217.00 µs | 185.86 µs | `reduce-overhead` | 175.95 µs |
| `[4, 128, 768]` | 469.52 µs | 484.15 µs | 475.61 µs | `default` | 475.61 µs |

Packing reduced external GEMMs from seven to four and profiled CUDA launches
from twelve to nine at the representative shape. Default compiled packing was
1.130×, 1.168×, and 1.018× faster than the original compiled block. Compiling
the entire packed block was about 1.4× faster than compiling attention and MLP
as separate regions for the two batch-one workloads, while the approaches tied
at `[4, 128, 768]`. Manual RMSNorm decomposition did not reduce the generated
launch count.

Artifacts: [analysis](artifacts/transformer_optimization/summary.md), [raw
result](artifacts/transformer_optimization/results.json), [before/after Dynamo
graphs](artifacts/transformer_optimization/dynamo_graphs.py), [original
wrapper](artifacts/transformer_optimization/original_inductor.py), [packed
wrapper](artifacts/transformer_optimization/packed_inductor.py)

### INT8 and FP8 quantization

TorchAO quantization is applied to the packed BF16 transformer block before
full-graph compilation with `max-autotune`. Each strategy runs in an isolated
process and Inductor cache.

| Strategy | Storage | `[1, 1, 768]` | `[1, 128, 768]` | `[4, 128, 768]` |
|---|---:|---:|---:|---:|
| BF16 | 13.50 MiB | 100.78 µs | 180.66 µs | 471.77 µs |
| INT8 weight-only | 6.78 MiB | 118.59 µs | 9996.52 µs | 39937.19 µs |
| Dynamic INT8 | 6.78 MiB | 109.65 µs | 173.87 µs | 340.93 µs |
| FP8 weight-only | 6.78 MiB | 107.41 µs | 214.44 µs | 518.04 µs |
| Dynamic FP8 | 6.75 MiB | 115.49 µs | 172.02 µs | 314.74 µs |

Dynamic INT8 achieved 1.039× and 1.384× prefill speedups; dynamic FP8 achieved
1.050× and 1.499×. None of the quantized paths improved single-token decode.
All quantized models reduced storage by approximately 2× and exceeded 0.9998
cosine similarity. The slow INT8 weight-only prefill path uses
`_weight_int8pack_mm` on this backend and demonstrates that reduced precision
does not guarantee lower latency.

Artifacts: [analysis](artifacts/transformer_quantization/summary.md), [raw
result](artifacts/transformer_quantization/results.json), [BF16
wrapper](artifacts/transformer_quantization/bf16_inductor.py), [INT8
wrapper](artifacts/transformer_quantization/int8_inductor.py), [FP8
wrapper](artifacts/transformer_quantization/fp8_inductor.py)

### Dynamic-shape specialization

A controlled gated-MLP experiment varies sequence length across
`[32, 64, 128, 32, 64]`. Each policy runs in a separate process with a
pass-through backend so the result isolates Dynamo graph capture from Inductor
code generation.

| Policy | `dynamic` argument | Graph compilations | Guard-triggered recompiles |
|---|---:|---:|---:|
| Static | `False` | 3 | 2 |
| Automatic | `None` | 2 | 1 |
| Upfront dynamic | `True` | 1 | 0 |

Static mode creates one graph per unique sequence length. Automatic mode first
captures a specialized graph, then recompiles once with a symbolic sequence
dimension. Upfront dynamic mode captures the symbolic graph immediately. When
64 and 128 are repeated, all policies reuse their existing cached graphs.

Artifacts: [analysis](artifacts/dynamic_shapes/summary.md), [raw
result](artifacts/dynamic_shapes/results.json), [sanitized recompile
log](artifacts/dynamic_shapes/recompiles.txt), [automatic-mode FX
graphs](artifacts/dynamic_shapes/automatic_graphs.py)

## Methodology

- Eager output is computed before compilation and used as the correctness
  reference.
- Compiled output must satisfy dtype-appropriate `torch.allclose` tolerances
  before timing results are written.
- CUDA measurements synchronize before and after each timed invocation.
- First-call latency is recorded separately from steady-state latency.
- Inductor uses a fresh temporary cache by default.
- Latency reports include median, p90, minimum, and maximum values.
- Break-even estimates compare observed first-call overhead with per-call
  steady-state savings.
- FX graphs are collected with a custom `torch.compile` backend used only for
  inspection.
- Generated wrapper calls are classified directly from an isolated Inductor
  cache; absolute cache paths are removed from published source artifacts.
- CUDA profiler events verify eager and compiled launch counts.
- Runtime variants alternate measurement order and separate host enqueue from
  post-enqueue completion wait.
- Dynamic-shape policies run in isolated processes with `TORCH_LOGS=recompiles`;
  a pass-through backend counts graph compilations and every output is checked
  against eager execution.
- Projection packing uses a one-time state conversion, equal parameter counts,
  identical inputs, isolated Inductor caches, alternating measurement order,
  multiple shape regimes, and eager/compiled correctness checks.
- Transformer strategies run in fresh processes and isolated Inductor caches;
  the mode matrix covers default, CUDA-graph-oriented, autotuned, no-CUDA-graph,
  and shape-padding configurations.
- Quantization reports model storage, cosine similarity, SQNR, generated external
  calls, and latency against a compiled BF16 reference.

The included results are specific to the recorded hardware, software, dtype,
and input shape. They are not portability or production-performance claims.

## Installation

Python 3.11+ and PyTorch 2.x are required. Install the appropriate PyTorch build
for the target CPU or CUDA environment before installing this package.

```bash
python -m venv .venv
source .venv/bin/activate

# Install PyTorch for the target platform first.
pip install -e ".[dev]"
python -m pytest
```

Install TorchAO for the quantization experiment:

```bash
pip install -e ".[quantization]"
```

## Usage

Pointwise baseline:

```bash
python -m compilelab \
  --device auto \
  --shape 1024,4096 \
  --warmup 20 \
  --iterations 100 \
  --output-dir artifacts/sample_run
```

Graph-break analysis:

```bash
python -m compilelab.graph_break \
  --device cpu \
  --output-dir artifacts/graph_break_case
```

Gated MLP benchmark:

```bash
python -m compilelab \
  --workload gated-mlp \
  --device cuda \
  --dtype float16 \
  --batch-size 4 \
  --sequence-length 128 \
  --model-dim 768 \
  --hidden-dim 2048 \
  --warmup 20 \
  --iterations 100 \
  --output-dir artifacts/sample_mlp_gpu
```

Generated-code and runtime analysis:

```bash
python -m compilelab.codegen \
  --dtype float16 \
  --batch-size 4 \
  --sequence-length 128 \
  --model-dim 768 \
  --hidden-dim 2048 \
  --warmup 50 \
  --iterations 500 \
  --profile-iterations 20 \
  --output-dir artifacts/mlp_codegen
```

Dynamic-shape specialization analysis:

```bash
python -m compilelab.dynamic_shapes \
  --sequence-lengths 32,64,128,32,64 \
  --model-dim 64 \
  --hidden-dim 128 \
  --output-dir artifacts/dynamic_shapes
```

Gated MLP projection-packing analysis:

```bash
python -m compilelab.projection_packing \
  --cases 1x1,1x128,4x128 \
  --dtype float16 \
  --model-dim 768 \
  --hidden-dim 2048 \
  --warmup 50 \
  --iterations 500 \
  --output-dir artifacts/mlp_projection_packing
```

Minimal transformer optimization analysis:

```bash
python -m compilelab.transformer_optimization \
  --cases 1x1,1x128,4x128 \
  --model-dim 768 \
  --num-heads 12 \
  --hidden-dim 2048 \
  --warmup 30 \
  --iterations 300 \
  --output-dir artifacts/transformer_optimization
```

Transformer quantization analysis:

```bash
python -m compilelab.quantization \
  --cases 1x1,1x128,4x128 \
  --model-dim 768 \
  --num-heads 12 \
  --hidden-dim 2048 \
  --warmup 30 \
  --iterations 300 \
  --output-dir artifacts/transformer_quantization
```

Artifact directories contain:

- `results.json`: environment, configuration, correctness, and measured data.
- `summary.md`: formatted result summary.
- Case-specific FX graphs, generated wrappers, or sanitized compiler logs.

## Repository structure

```text
compilelab/
  benchmark.py       Benchmark runner and artifact generation
  codegen.py         Inductor wrapper and runtime analysis
  dynamic_shapes.py  Guard and shape-specialization analysis
  graph_break.py     Data-dependent control-flow experiment
  quantization.py    TorchAO INT8 and FP8 comparison
  projection_packing.py  Gated-MLP state conversion and GPU comparison
  transformer.py     Minimal and projection-packed transformer blocks
  transformer_optimization.py  Compile-mode and fusion analysis
  workload.py        Pointwise and gated-MLP workloads
tests/
  test_compilelab.py Correctness and graph-capture tests
artifacts/           Curated benchmark and graph-capture results
```

## Current scope

This repository currently covers compiler capture, graph breaks, timing
methodology, generated-code inspection, launch profiling, dynamic-shape
specialization, transformer projection packing, compilation boundaries,
compile-mode selection, RMSNorm lowering, and TorchAO INT8/FP8 quantization. It
does not currently cover sparsity, full-model evaluation, kernel-level Nsight
analysis, training optimization, or production inference integration.
