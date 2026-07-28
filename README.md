# TorchCompile Lab

An incremental, hands-on study of how PyTorch 2 turns eager model code into
compiled programs—and how to measure whether a model optimization actually
helps.

> **Status: v0.2 / work in progress.** The repository currently implements a
> reproducible eager-vs-`torch.compile` baseline, a data-dependent graph-break
> case study, and a gated transformer MLP benchmark.

## What is implemented

- A small `nn.Module` containing bias addition, SiLU, multiplication, and a
  reduction.
- A 4.7-million-parameter gated MLP with the gate/up/down projection structure
  used by decoder-only transformers.
- A data-dependent Python branch and an equivalent `torch.cond` formulation.
- Eager and compiled execution on CPU or CUDA.
- Separate first-call and steady-state latency measurements.
- A fresh temporary Inductor cache by default so first-call runs do not silently
  reuse an older generated program.
- Numerical equivalence checks before reporting performance.
- Environment and run metadata saved as JSON.
- Dynamo graph capture through a custom `torch.compile` backend.
- Six automated tests covering workload, graph-capture, branch, and MLP
  correctness.

The first experiment asks a deliberately narrow question:

> Can `torch.compile` preserve this workload's output, capture it as one FX
> graph, and improve steady-state execution enough to repay compilation cost?

## First measured results

Illustrative float32 runs with PyTorch 2.12.0, inputs of shape
`[1024, 4096]`, and 100 timed iterations produced:

| Metric | RTX 5050 Laptop GPU | CPU |
|---|---:|---:|
| Dynamo graphs captured | 1 | 1 |
| Maximum absolute error | 1.19e-7 | 3.58e-7 |
| Eager steady-state median | 0.2332 ms | 7.5241 ms |
| Compiled steady-state median | 0.1006 ms | 1.0567 ms |
| Steady-state speedup | 2.32× | 7.12× |
| Compiled first call | 1.45 s | 4.37 s |
| Estimated break-even | ~10,968 calls | ~676 calls |

The important finding is the tradeoff: compilation substantially improved this
workload's steady state on both devices, but only a repeatedly invoked workload
can amortize the one-time cost. These are preliminary, single-process
measurements—not general CPU or GPU claims.

GPU artifacts:
[summary](artifacts/sample_gpu/summary.md),
[raw result](artifacts/sample_gpu/results.json), and
[captured FX graph](artifacts/sample_gpu/dynamo_fx_graph.py).

CPU artifacts:
[summary](artifacts/sample_cpu/summary.md),
[raw result](artifacts/sample_cpu/results.json), and
[captured FX graph](artifacts/sample_cpu/dynamo_fx_graph.py).

## Graph-break case study

The original module branches in Python on a tensor value:

```python
if x.sum() > 0:
    return torch.sin(y)
return torch.cos(y)
```

Across positive and negative inputs, Dynamo captures a predicate graph and two
separate continuation graphs. The original therefore produces three graphs and
fails with `Data-dependent branching` under `fullgraph=True`.

The revised module expresses the same branch using `torch.cond`. Both paths
remain numerically equivalent, while Dynamo captures the revised module as one
full graph.

| Result | Python branch | `torch.cond` |
|---|---:|---:|
| Positive and negative outputs correct | yes | yes |
| Graphs captured across both paths | 3 | 1 |
| Accepted by `fullgraph=True` | no | yes |

See the [case-study summary](artifacts/graph_break_case/summary.md), the
[three original graph regions](artifacts/graph_break_case/python_if_graphs.py),
and the [single revised graph](artifacts/graph_break_case/torch_cond_graph.py).
This is a structural result; it does not assume that fewer graphs are
automatically faster.

## Gated transformer MLP

The model-relevant workload implements:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

On the RTX 5050 with float16 input `[4, 128, 768]` and hidden dimension 2048:

| Metric | Result |
|---|---:|
| Parameters | 4,718,592 |
| Dynamo graphs captured | 1 |
| Maximum absolute error | 2.44e-4 |
| Eager steady-state median | 0.2402 ms |
| Compiled steady-state median | 0.2897 ms |
| Steady-state speedup | 0.829× |
| Compiled first call | 1.55 s |

Compilation was approximately 17% slower at this shape; a fresh-process repeat
was approximately 19% slower. The current hypothesis is that three GEMMs
dominate the block and already use optimized library kernels, leaving too
little pointwise work for compilation to recover its overhead. Generated-code
inspection is required before treating that as the causal explanation.

See the [MLP summary](artifacts/sample_mlp_gpu/summary.md), [primary raw
result](artifacts/sample_mlp_gpu/results.json), [fresh-process
repeat](artifacts/sample_mlp_gpu/repeat_results.json), and [captured FX
graph](artifacts/sample_mlp_gpu/dynamo_fx_graph.py).

## Quick start

Python 3.11+ and PyTorch 2.x are required. Install the correct PyTorch build for
your CPU or CUDA environment first, then install this project:

```bash
python -m venv .venv
source .venv/bin/activate

# Install PyTorch using the selector at https://pytorch.org/get-started/locally/
pip install -e ".[dev]"
pytest
```

Run the baseline:

```bash
python -m compilelab \
  --device auto \
  --shape 1024,4096 \
  --warmup 20 \
  --iterations 100 \
  --output-dir artifacts/sample_run
```

Run the graph-break case:

```bash
python -m compilelab.graph_break \
  --device cpu \
  --output-dir artifacts/graph_break_case
```

Run the gated MLP on CUDA:

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

The command creates:

- `results.json`: environment, correctness, first-call latency, latency
  distribution, and break-even estimate.
- `summary.md`: a human-readable result table.
- `dynamo_fx_graph.py`: readable code for the graph captured by Dynamo.

Results are hardware- and shape-specific. Do not treat a CPU result as evidence
of GPU performance, and do not describe first-call latency as steady-state
latency.

## Current learning outcomes

The completed milestones demonstrate five ideas that are easy to obscure in a
single “compiled latency” number:

1. `torch.compile` is lazy: substantial compilation work occurs on the first
   invocation.
2. Correctness must be checked before timing an optimized variant.
3. A steady-state speedup is useful only in relation to compilation cost and
   the number of calls needed to amortize it.
4. Data-dependent Python control flow fragments capture, while structured
   tensor control flow can keep the decision inside a full graph.
5. Capturing a model block as one graph does not guarantee a speedup,
   particularly when optimized library GEMMs dominate execution.

## Short roadmap

The next milestones are intentionally small:

1. Inspect generated Inductor code to test the explanation for the MLP
   regression.
2. Vary tensor shapes and record guard failures and recompilations.
3. Implement one tested FX rewrite, such as packing the gate and up
   projections, and connect it to generated code and latency.

## Repository layout

```text
compilelab/                  Runnable experiment code
tests/                       Fast correctness and capture tests
artifacts/                   Generated runs (ignored except curated samples)
```

## Scope and claims

This is a learning project, not a production inference system. Current results
cover one synthetic workload, one graph-break microcase, and one untrained
transformer MLP. Quantization, sparsity, transformer graph rewrites, and GPU
kernel forensics are planned work and should not be described as implemented
until their code and evidence are committed.
