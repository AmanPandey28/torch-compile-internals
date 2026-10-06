# Gated MLP regression: evidence and limits

The workload in `compilelab/workload.py::GatedMLP` computes
`down_proj(silu(gate_proj(x)) * up_proj(x))`. With model dimension 768 and hidden
dimension 2048 it contains `3 * 768 * 2048 = 4,718,592` parameters. The main
comparison uses FP16 input `[4,128,768]`, flattened to 512 rows for the GEMMs.

## Latency comparison

The benchmark uses three fresh processes, isolated Inductor and Triton
caches, 20 warmups, 100 samples per variant, and alternating execution order.
It measures 13.1–16.2% greater compiled latency across the three trials. See
[raw results and replay commands](../artifacts/benchmark_results/results.md).

For eager latency `Te` and compiled latency `Tc`, the latency increase is
`Tc / Te - 1`, while the steady-state ratio is `Te / Tc`. Throughput decrease
uses a different denominator: `1 - Te / Tc`. Reporting the metric explicitly
avoids treating those percentages as interchangeable.

## What the generated wrapper establishes

`compilelab/codegen.py` reads the generated wrapper before its temporary cache
is removed. The wrapper contains three external `mm` calls and one generated
Triton SiLU/multiply kernel. The three projections remain GEMMs. Eager has
separate SiLU and multiply kernels, giving five kernels; compilation fuses those
operations and gives four. These are compiler-generated kernels, not kernels
authored in the project.

The profiler counts `cat="kernel", ph="X"` events from exported traces. It
excludes CPU operators, GPU annotation ranges, memory copies, and memsets.
`compilelab/trace_analysis.py` centralizes that rule so annotations cannot inflate
counts in different experiments.

## What the runtime decomposition establishes

The 500-sample runtime experiment gives these medians:

| Component | Eager | Compiled |
|---|---:|---:|
| Host enqueue | 50.93 µs | 94.93 µs |
| Post-enqueue completion wait | 229.23 µs | 220.66 µs |
| Synchronized total | 278.01 µs | 315.07 µs |

The regression persists despite one fewer kernel. The increase occurs mainly
in the interval from entering the model to its return, while the subsequent
completion wait decreases. This localizes an important part of the overhead
to the host-return interval for this workload and software stack.

The measurements do **not** isolate the cost of one particular guard, wrapper
instruction, allocation, or launch API. A stronger explanation would require
host stack sampling or a targeted ablation. GPU kernels can already be running
while the CPU submits later work. Completion wait is therefore outstanding
work plus synchronization overhead, rather than total GPU compute time.
Component medians are independent and need not sum to the median total.
Profiler device durations also contain instrumentation overhead and cannot
replace the unprofiled latency measurement.

## Evidence and diagnostic limits

| Hypothesis | Existing evidence | What would test it more directly |
|---|---|---|
| Launch-count reduction guarantees a win | Rejected: five to four kernels, yet longer synchronized latency | Compare latency across token counts; retain regressions |
| Cold compilation explains steady-state slowdown | Rejected: compilation and warmup finish before timing | Record backend compilations and first-call time separately |
| Additional host dispatch/wrapper cost matters | Supported by greater host-return interval | Host stacks, guard/wrapper ablation, or observed CUDA Graph replay |
| The fused kernel is worse | Not established by wrapper structure alone | Match kernel shapes and measure device execution outside broad model instrumentation |
| Shape matters | Batch-one MLP cases and small pointwise cases also regress | Sweep more row counts under controlled clocks/power |
| GEMMs became compiler-authored | Rejected in the inspected wrapper | Inspect external calls and actual kernel names |

Reversible projection packing is a measured model transformation:
concatenate gate/up weights once, perform one larger input GEMM, split its output
with views, then apply SiLU/multiply. This reduces compiled external GEMMs from
three to two. Its measured 1.04–1.17× improvements compare packed and unpacked
**compiled** paths; they do not establish a win over unpacked eager execution.

## Conclusion

Full-graph capture and SiLU/multiply fusion remove one GPU kernel but do not
improve this workload's latency. The alternating-order timing split localizes
additional cost to the host-return interval. Host stacks or a targeted ablation
would be needed to attribute that cost to a specific wrapper component.
