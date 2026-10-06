# Qwen trace interpretation: a measured example

This walkthrough uses `qwen_float32_no-cudagraphs_1` at 44 prompt tokens.
The [raw result](../artifacts/benchmark_results/qwen_float32_no-cudagraphs_1/results.json)
contains exact names, counts, durations, and timing samples. The conclusions
apply to that captured run. Use the two-trial table to assess repeatability.

| Phase | Eager median | Compiled median | Eager GPU kernels | Compiled GPU kernels |
|---|---:|---:|---:|---:|
| Prefill | 30.189 ms | 26.367 ms | 1,385 | 482 |
| Eight decode steps | 97.875 ms | 66.800 ms | 8,968 | 3,280 |
| Thirty-two decode steps | 386.481 ms | 341.257 ms | 35,872 | 13,120 |

The largest launch reductions do not translate proportionally into latency
reductions. Prefill's ratio is about 1.15×, eight-step decode about 1.47×,
and 32-step decode about 1.13× in this trial. These phase ratios differ even
though the decode kernels per forward are stable.

## Compiler-generated normalization and MLP fusion

The eager trace has separate power, mean, reciprocal-square-root, and multiply
work associated with RMS normalization. Generated Inductor reductions contain
those operations together, sometimes with neighboring residual or embedding
operations. The MLP's SiLU and multiply are represented in generated pointwise
kernels. Inspect the saved FX nodes, actual kernel arithmetic, and matching
executed names before identifying a fusion.

The model has 24 decoder layers. A count of 24 fused MLP calls in prefill and
192 over eight decode forwards is consistent with one per layer per invocation.
Matrix projections continue to perform substantial external GEMM/GEMV work;
fusion does not remove the model's matrix multiplications.

## Attention backend selection contributes

Eager FP32 prefill uses math attention, including 24 softmax kernels. Compiled
prefill executes 24 `fmha_cutlassF_f32...` efficient-attention kernels. Both paths
request the SDPA API, but their selected implementations differ. That changes
intermediate work and launch counts beyond normalization or MLP fusion.

Inspect `cpu_operator_counts` for the SDPA operations, `trace_summary.kernels`
for executed names, and the saved generated wrapper for external efficient
attention calls. Kernel-name substrings alone are a starting point for an
investigation, rather than proof of an entire algorithm or its memory traffic.

## Decode timing supports a bounded conclusion

| Eight-step decode observation | Eager | Compiled |
|---|---:|---:|
| Unprofiled host-return interval median | 96.367 ms | 46.998 ms |
| Unprofiled outstanding completion-wait median | 1.516 ms | 20.281 ms |
| Profiled sum of kernel durations | 55.307 ms | 64.294 ms |
| Profiled GEMV-named kernel duration sum | 47.166 ms | 60.311 ms |
| Kernel interval coverage within the trace's kernel window | 36.6% | 84.3% |

Most recorded **device kernel time** is in GEMV-named kernels. Eager's traced
kernel window also has many gaps, while its unprofiled model invocation returns
near completion. Compiled execution has a shorter host-return interval and a
denser kernel timeline in this run. This supports investigating submission and
dispatch gaps as part of the decode latency difference.

It does not prove that the workload is purely launch-bound, nor that the
compiled GEMVs execute faster. The profiled compiled device sum is actually
larger in this capture. Profiling thousands of short events can perturb the
schedule, and clock/power state can change. Kernel interval coverage does not
measure hardware occupancy or achieved bandwidth. Establishing a bandwidth
bottleneck would require counters such as achieved memory throughput and a
workload-specific byte model.

Do not add host enqueue to summed GPU kernel durations: execution can overlap
submission. Do not combine independent median components as an exact causal
decomposition. The synchronized phase time, measured outside the profiler,
is the latency result.

## Graphs, guards, and warm execution

This worker recorded five backend invocations across the sweep. The guard logs
include a keyword-count mismatch between prefill and cached decode, and a
cache-key stride mismatch on the transition to appended cache storage. Dynamic
sequence dimensions do not eliminate those Python-state/layout guards.

Accepted timed phases report zero new backend compilations. Profiler captures
are also checked for compilation markers and unexpected CUDA Graph replay.
The first compiled prefill took about 37.79 seconds in this fresh worker;
subsequent phase-first calls can include additional specializations and must
not all be described as independently cold compilations.

## Conclusion

The 44-token FP32 trace shows compiler-generated normalization/MLP fusion and
an attention-backend change. Decode's device time is mainly in GEMV-named
kernels, while eager execution also has substantial submission gaps. The
compiled path has a shorter host-return interval in this trial. Hardware
counters and host stacks are needed for a stronger bottleneck attribution.
