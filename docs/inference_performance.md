# Compiler and Qwen inference: methodology and code walkthrough

This project starts with small functions whose behavior can be inspected, then
applies the same measurement discipline to Qwen2.5-0.5B-Instruct. It studies
compiler-generated kernels and model-forward inference. It does not implement
custom attention kernels or production serving performance.

## Read the code in this order

1. `compilelab/workload.py`: understand the pointwise expression and three-projection gated MLP.
2. `compilelab/benchmark.py`: follow inputs, eager reference, first compiled call, warmup, alternating timing, and FX capture.
3. `compilelab/graph_break.py`: compare Python tensor-dependent branching with `torch.cond`; inspect both predicates and boundary cases.
4. `compilelab/codegen.py`: connect generated wrapper calls to actual kernel counts and runtime decomposition.
5. `compilelab/trace_analysis.py`: see why only GPU `kernel` execution slices count, and how interval overlap is handled.
6. `compilelab/qwen.py`: follow prompt preparation, reference continuation, fresh cache creation, compilation, validation, timing, and profiling.
7. `compilelab/experiments.py`: follow process/cache isolation, exact commands, source hashes, device snapshots, and failure logs.
8. `compilelab/evidence.py`: export raw measurements, source hashes, replay commands, and result tables.

Projection packing, the minimal transformer, and quantization are additional
controlled investigations. Read `workload.py::PackedGatedMLP`, then
`projection_packing.py`, `transformer.py`, `transformer_optimization.py`, and
`quantization.py`. The synthetic block's sequence-length-one case is
**decode-shaped**; `qwen.py` is the real KV-cached decode experiment.

## What each compiler stage does

Dynamo observes Python execution and captures tensor operations into an FX
graph. Its guards describe assumptions under which a compiled result can be
reused. A guard mismatch can lead to another graph capture. `fullgraph=True`
makes a graph break fail for the compiled callable; it does not force every
possible input shape, cache state, or Python value to share one graph.

The backend receives the FX graph and example inputs. Qwen's recording backend
writes that graph, records its input signature, then delegates to
`torch._inductor.compile`. Inductor's path includes AOTAutograd graph processing
and lowering; these inference experiments do not compile or benchmark a
backward pass. The backend may fuse elementwise/reduction operations into
Triton kernels and retain matrix multiplies or attention on external paths.

An FX node is not necessarily one GPU kernel. Several nodes may fuse; a single
library call may launch multiple kernels. The generated wrapper explains the
static implementation, and the CUDA trace establishes what actually executed.
Use the [PyTorch compile API](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
and [compile profiling guide](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html)
alongside the local code.

## Why prefill and decode must be separate

Prefill processes all prompt tokens together and produces K/V state at every
attention layer. The benchmark requests only the final position's logits with
`logits_to_keep=1`. That avoids computing vocabulary logits for positions that
are not used in this experiment. The argument is identical for both variants.

Decode feeds one continuation token per forward call and appends its K/V state
to the cache. Attention reads the retained prefix. Workload shape and the
relative costs of matrix multiplication, cache movement, and dispatch differ
from prefill. An eight-step decode measurement covers eight forwards; dividing
its time by eight gives an average, not a per-token p90 or a complete service
TPOT measurement.

`prepare_work("decode", ...)` creates a fresh prefilled cache **before** each
trial's timer or trace starts. The returned closure consumes that cache once.
Reusing the closure for another timing sample would silently extend the context
and invalidate the comparison. A unit test explicitly detects that mistake.
See the [Transformers cache documentation](https://huggingface.co/docs/transformers/main/en/kv_cache)
for the current cache classes; support and capture behavior vary by version.

For each dtype and prompt length, the eager path generates one greedy reference
continuation. Both variants replay those exact token tensors. This controls
context and token-selection differences while permitting logit and argmax
comparison at each measured position. It does not benchmark the token-selection
or text-output loop and does not establish task accuracy.

The native chat prompt has 44 tokens for the pinned tokenizer. Other
lengths use deterministic repetition/truncation of its token IDs. These are
controlled synthetic length cases rather than a natural-language evaluation
dataset. The exact prompt and continuation IDs are saved in `tokens_<length>.json`.

## Measurement boundaries

Model loading, tokenization, and eager reference generation occur outside
timing. Each phase records its first observed eager and compiled call. Only the
first compiled phase in the fresh worker has a cold process; later first calls
may reuse prior graphs or compile additional specializations. Calling them all
“cold compile times” would be inaccurate.

Each variant warms the complete tested phase, including all decode steps.
Correctness is checked before performance results are accepted. Timed trials
alternate eager/compiled order and synchronize CUDA before and after the phase.
The JSON includes every sample, median, p90, minimum, and maximum. Prefill
latency and total decode-phase latency have separate rows.

The timer also records the interval from invocation to model return and the
subsequent synchronization wait. The GPU can run earlier kernels while the CPU
submits later ones. The wait therefore measures outstanding work plus
synchronization overhead, not the entire GPU compute duration. Allocations or
blocking calls can contribute to the host-return interval. Independent
component medians do not add to the median total.

The harness serializes workers to avoid benchmark interference. Each worker
gets its own Inductor and Triton directories; remote FX graph caching is
disabled. A fresh process still benefits from OS page caches and loaded GPU
driver state, so the measured first call is not a complete reboot-cold cost.
Power/clocks are sampled and left uncontrolled. This laptop GPU also drives
the display, which limits generalization from small timing differences.

## Correctness and BF16 interpretation

FP32 uses `atol=rtol=1e-3` with TF32 disabled. BF16 uses
`atol=rtol=2e-2`, reflecting its lower arithmetic precision. The CLI can set
explicit tolerances, but changes must be disclosed with the resulting errors.
The BF16 configuration preserves intermediate precision casts with Inductor's
`emulate_precision_casts=True`. This version-specific setting is recorded in
each result's compile options. Math attention is pinned equally for both
variants. The sweep records accuracy across all tested prompt/decode lengths.
Mismatching cases have no
accepted steady-state timing or launch-count comparison. The JSON's
`study_status="rejected"` distinguishes a completed rejected study from a
validated performance result.
The completed sweep rejected all 48 BF16 cases (four prompt lengths, three
phases, two configurations, two trials). Maximum absolute errors ranged from
0.23828125 to 0.984375, and eight cases disagreed on at least one greedy token.
FP32 passed all cases in both configurations. These observations describe this
model and tested software stack, not BF16 compilation in general.
Every vocabulary logit must satisfy `allclose`, actual values must be finite,
and maximum/mean absolute errors are recorded. Greedy argmax agreement is
reported separately: close logits can still swap two nearly tied tokens.

A failed tolerance stops a normal single-case run and preserves its result.
The harness explicitly enables `--continue-on-mismatch` for the BF16 accuracy
sweep: it records the rejection, skips that case's performance measurements,
and continues coverage. A completed rejected study returns normally with
`study_status="rejected"`; unexpected runtime failures still exit nonzero.
It must not be presented as a successful optimization. Replaying common tokens
also means that matching argmaxes at those positions is a local consistency
check, not proof of identical free-running generation for arbitrary requests.

## CUDA Graph configurations

`no-cudagraphs` sets `triton.cudagraphs=False`. It isolates the compiler's
generated implementation without replay bundling. The trace reader rejects any
CUDA Graph replay in that configuration. `reduce-overhead` requests the mode's
installed Inductor options, including graph support where available.

Requesting CUDA Graphs does not prove they were used. Cache mutation, dynamic
shapes, or other conditions can prevent graphing. The trace records actual
`cudaGraphLaunch`/`cuGraphLaunch` calls. GPU kernel events still count work
executed during replay; host launch API counts can fall without the same
reduction in executed GPU kernels. Never equate the two measurements.

## Prove fusion and attention changes

For the representative FP32 no-graph run, open these public artifacts:

- [prefill/decode FX captures](../artifacts/benchmark_results/qwen_float32_no-cudagraphs_1/dynamo_graph_1.py).
- [generated wrapper](../artifacts/benchmark_results/qwen_float32_no-cudagraphs_1/generated/inductor_1.py).
- [kernel names, durations, numerical checks, and raw latencies](../artifacts/benchmark_results/qwen_float32_no-cudagraphs_1/results.json).

Search the FX graph for normalization's `pow`, `mean`, `rsqrt`, and multiply
operations. Find the corresponding generated reduction kernels and inspect
their actual arithmetic. Search generated pointwise kernels for SiLU and
multiply; match their names against the executed trace summary. A suggestive
kernel name alone is weaker evidence than agreement between graph operations,
generated source, and runtime execution.

In the measured 44-token FP32 case, eager prefill has 24 softmax kernel calls
associated with math attention, while compiled prefill has 24
`fmha_cutlassF_f32...` calls. The generated wrapper calls efficient attention.
The SDPA API setting is the same, but the selected implementation differs.
Consequently, attribution of the entire launch reduction to pointwise fusion
would be incorrect.

## Guards and dynamic shapes

The recording backend counts graph compilations and saves all graph signatures.
The worker logs `TORCH_LOGS=recompiles,graph_breaks`, and the evidence exporter
retains sanitized guard-failure records. Backend invocation counts and log
records answer different questions; neither should be silently substituted for
the other. The worker explicitly verifies that no new backend compilation occurs
during steady-state timing or profiler capture.

For an isolated capture example, `dynamic_shapes.py` uses an eager
pass-through backend with sequence lengths `[32,64,128,32,64]`. Static mode
captures three graphs, automatic mode two, and upfront dynamic mode one in the
recorded experiment. This isolates Dynamo specialization, not Inductor speed.
The full Qwen experiment adds Python/cache-state guards, which may need separate
specializations even with `dynamic=True`.

## Read a trace without double-counting

The process's main Python thread records CPU operators, wrapper work, and launch
APIs. GPU stream rows record executed kernels. Colors are not a reliable way to
identify the device. CPU parent ranges such as `aten::linear`, `aten::matmul`,
and `aten::mm` overlap; adding their durations double-counts work. GPU annotation
ranges can likewise span kernels and must not count as additional executions.

`summarize_trace` accepts only complete GPU `kernel` events. It reports exact
names, count, and duration totals. Overlapping kernel intervals are merged to
compute activity coverage within the first-to-last kernel window. That number
does not measure SM occupancy, compute throughput, memory bandwidth, or general
hardware utilization. Diagnose those questions with appropriate hardware
counters instead of asserting that short kernels prove a bandwidth bottleneck.

Search Perfetto for the phase annotation, then `triton`, `gemm`, `gemv`,
`softmax`, or `fmha`. Follow launch correlations to the GPU row. Distinguish
CPU submission duration, GPU queue delay, kernel duration, and the final CPU
synchronization wait. The latency result comes from the unprofiled trials.

## Reproduce and audit

```bash
python -m pytest
python -m compilelab.experiments --suite all --output-dir artifacts/benchmark_runs
python -m compilelab.evidence --run-dir artifacts/benchmark_runs --output-dir artifacts/benchmark_results
```

The default suite has three primary microbenchmark trials and two trials for
each of four Qwen dtype/configuration combinations. It writes `manifest.json`,
per-worker `results.json`, standard output/error logs, FX graphs, token IDs,
generated wrappers, and Chrome traces. `--resume` reuses completed workers in
the same directory; independent trials require separate run directories.

The result exporter verifies raw-file hashes and retains representative FX
graphs, generated wrappers, and parsed guard failures. Replay commands keep the
measured workload flags while normalizing the Python executable and output path.
Worker source copies, diagnostic logs, and large traces stay in the run directory;
source hashes in the public metadata identify the measured worker files.

## Output ownership during CUDA Graph replay

CUDA Graph output buffers can be reused by subsequent invocations. Retaining
cache tensors in that storage risks reading overwritten values. The ownership
policy marks each invocation and clones logits and DynamicCache
K/V outside the compiled function. It applies to **both** variants in the
practical configuration, and the timing/counts include those copies. Its eager
baseline is therefore different from the no-graph baseline; compare within
each configuration. This is a correctness-first replay experiment, not a claim
that cloning the whole cache is an efficient serving design. Static cache or
persistent retained buffers require separate lifetime and performance validation. See
[CUDA Graph Trees output lifetimes](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html).

Logit tolerance establishes numerical agreement for the tested positions and
dtype. Free-running generation, task accuracy, and broader prompts require
additional evaluation. NVIDIA timings do not establish performance on other
accelerators; their compiler, runtime, and serving paths must be measured on
their own targets.
