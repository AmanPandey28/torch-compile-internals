# Torch Compile Internals

A reproducible benchmark and graph-analysis suite for examining PyTorch 2
compiler behavior on synthetic kernels and transformer components.

The current implementation evaluates:

- Dynamo graph capture and full-graph compatibility.
- First-call compilation cost versus steady-state execution.
- Numerical equivalence between eager and compiled execution.
- Data-dependent control flow and graph fragmentation.
- Compiler behavior on a gated transformer MLP.

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
fresh-process repeat. Full-graph capture therefore did not produce a
performance improvement for this configuration. The current evidence does not
attribute the regression to a specific kernel or compiler transformation;
generated-code and profiler analysis would be required for that conclusion.

Artifacts: [summary](artifacts/sample_mlp_gpu/summary.md), [primary
result](artifacts/sample_mlp_gpu/results.json), [repeat
result](artifacts/sample_mlp_gpu/repeat_results.json), [FX
graph](artifacts/sample_mlp_gpu/dynamo_fx_graph.py)

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
pytest
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

Each benchmark run writes:

- `results.json`: environment, configuration, correctness, and timing data.
- `summary.md`: formatted result summary.
- `dynamo_fx_graph.py`: captured Dynamo FX graph.

## Repository structure

```text
compilelab/
  benchmark.py       Benchmark runner and artifact generation
  graph_break.py     Data-dependent control-flow experiment
  workload.py        Pointwise and gated-MLP workloads
tests/
  test_compilelab.py Correctness and graph-capture tests
artifacts/           Curated benchmark and graph-capture results
```

## Current scope

This repository currently covers compiler capture, graph breaks, timing
methodology, and one transformer MLP component. It does not currently include
quantization, sparsity, full-model evaluation, generated-kernel analysis, or
production inference integration.
