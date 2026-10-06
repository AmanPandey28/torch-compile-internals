# Compiler and Qwen inference evidence

Measurements apply to the recorded hardware, software, and inputs. Raw result files are retained byte-for-byte; replay commands and measured source hashes are in [evidence.json](evidence.json).

## Fresh-process compiler measurements

| Case | Eager median ms | Compiled median ms | Ratio | First call s | Break-even calls | Evidence |
|---|---:|---:|---:|---:|---:|---|
| pointwise_primary_1 | 0.2349 | 0.1134 | 2.071× | 4.692 | 38625 | [JSON](pointwise_primary_1/results.json) |
| mlp_primary_1 | 0.2637 | 0.3037 | 0.868× | 2.583 | none | [JSON](mlp_primary_1/results.json) |
| pointwise_primary_2 | 0.2335 | 0.1104 | 2.116× | 2.380 | 19323 | [JSON](pointwise_primary_2/results.json) |
| mlp_primary_2 | 0.2673 | 0.3023 | 0.884× | 2.220 | none | [JSON](mlp_primary_2/results.json) |
| pointwise_primary_3 | 0.2344 | 0.1128 | 2.079× | 1.712 | 14072 | [JSON](pointwise_primary_3/results.json) |
| mlp_primary_3 | 0.2445 | 0.2840 | 0.861× | 1.906 | none | [JSON](mlp_primary_3/results.json) |
| pointwise_64x257 | 0.0320 | 0.0593 | 0.540× | 1.709 | none | [JSON](pointwise_64x257/results.json) |
| pointwise_128x1024 | 0.0302 | 0.0566 | 0.534× | 1.713 | none | [JSON](pointwise_128x1024/results.json) |
| mlp_1x1 | 0.0648 | 0.1007 | 0.643× | 1.864 | none | [JSON](mlp_1x1/results.json) |
| mlp_1x128 | 0.0851 | 0.1238 | 0.688× | 1.860 | none | [JSON](mlp_1x128/results.json) |

Break-even uses `ceil(max(0, first_call_ms - compiled_median_ms) / (eager_median_ms - compiled_median_ms))`. It is undefined when compiled execution is slower. First-call time includes capture, lowering, compilation, and initial execution; it is not pure compiler CPU time. Across-trial variation is empirical variation, not a confidence interval.

## Qwen prefill and decode

Times are whole measured phases, outside the profiler. Decode excludes its prefill setup and token selection; per-forward decode values are averages, not token-latency percentiles.

| Run | Prompt | Phase/steps | Eager ms | Compiled ms | Ratio | GPU kernels eager → compiled | Max logit error | Greedy IDs equal | Graph replay calls |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|
| [qwen_float32_no-cudagraphs_1](qwen_float32_no-cudagraphs_1/results.json) | 32 | prefill/0 | 20.035 | 18.673 | 1.073× | 1409 → 506 | 3.72e-05 | True | 0 |
| [qwen_float32_no-cudagraphs_1](qwen_float32_no-cudagraphs_1/results.json) | 32 | decode/8 | 97.667 | 82.200 | 1.188× | 8968 → 3280 | 3.91e-05 | True | 0 |
| [qwen_float32_no-cudagraphs_1](qwen_float32_no-cudagraphs_1/results.json) | 32 | decode/32 | 384.104 | 349.152 | 1.100× | 35872 → 13120 | 0.000191 | True | 0 |
| [qwen_float32_no-cudagraphs_1](qwen_float32_no-cudagraphs_1/results.json) | 44 | prefill/0 | 30.189 | 26.367 | 1.145× | 1385 → 482 | 2.55e-05 | True | 0 |
| [qwen_float32_no-cudagraphs_1](qwen_float32_no-cudagraphs_1/results.json) | 44 | decode/8 | 97.875 | 66.800 | 1.465× | 8968 → 3280 | 6.1e-05 | True | 0 |
| [qwen_float32_no-cudagraphs_1](qwen_float32_no-cudagraphs_1/results.json) | 44 | decode/32 | 386.481 | 341.257 | 1.133× | 35872 → 13120 | 6.1e-05 | True | 0 |
| [qwen_float32_no-cudagraphs_1](qwen_float32_no-cudagraphs_1/results.json) | 128 | prefill/0 | 35.075 | 33.242 | 1.055× | 1337 → 434 | 6.46e-05 | True | 0 |
| [qwen_float32_no-cudagraphs_1](qwen_float32_no-cudagraphs_1/results.json) | 128 | decode/8 | 100.727 | 72.116 | 1.397× | 8968 → 3280 | 0.000115 | True | 0 |
| [qwen_float32_no-cudagraphs_1](qwen_float32_no-cudagraphs_1/results.json) | 128 | decode/32 | 383.556 | 326.318 | 1.175× | 35872 → 13120 | 0.000115 | True | 0 |
| [qwen_float32_no-cudagraphs_1](qwen_float32_no-cudagraphs_1/results.json) | 512 | prefill/0 | 131.692 | 113.655 | 1.159× | 1337 → 434 | 3.62e-05 | True | 0 |
| [qwen_float32_no-cudagraphs_1](qwen_float32_no-cudagraphs_1/results.json) | 512 | decode/8 | 99.200 | 84.574 | 1.173× | 8968 → 3280 | 4.1e-05 | True | 0 |
| [qwen_float32_no-cudagraphs_1](qwen_float32_no-cudagraphs_1/results.json) | 512 | decode/32 | 396.501 | 348.218 | 1.139× | 35872 → 13120 | 5.63e-05 | True | 0 |
| [qwen_float32_reduce-overhead_1](qwen_float32_reduce-overhead_1/results.json) | 32 | prefill/0 | 20.365 | 18.441 | 1.104× | 1409 → 506 | 3.72e-05 | True | 1 |
| [qwen_float32_reduce-overhead_1](qwen_float32_reduce-overhead_1/results.json) | 32 | decode/8 | 106.666 | 70.668 | 1.509× | 8968 → 3280 | 3.91e-05 | True | 8 |
| [qwen_float32_reduce-overhead_1](qwen_float32_reduce-overhead_1/results.json) | 32 | decode/32 | 401.251 | 348.916 | 1.150× | 35872 → 13120 | 0.000191 | True | 32 |
| [qwen_float32_reduce-overhead_1](qwen_float32_reduce-overhead_1/results.json) | 44 | prefill/0 | 31.591 | 26.342 | 1.199× | 1385 → 482 | 2.55e-05 | True | 1 |
| [qwen_float32_reduce-overhead_1](qwen_float32_reduce-overhead_1/results.json) | 44 | decode/8 | 105.502 | 66.887 | 1.577× | 8968 → 3280 | 6.1e-05 | True | 8 |
| [qwen_float32_reduce-overhead_1](qwen_float32_reduce-overhead_1/results.json) | 44 | decode/32 | 405.379 | 332.982 | 1.217× | 35872 → 13120 | 6.1e-05 | True | 32 |
| [qwen_float32_reduce-overhead_1](qwen_float32_reduce-overhead_1/results.json) | 128 | prefill/0 | 35.490 | 34.117 | 1.040× | 1337 → 434 | 6.46e-05 | True | 1 |
| [qwen_float32_reduce-overhead_1](qwen_float32_reduce-overhead_1/results.json) | 128 | decode/8 | 104.207 | 79.838 | 1.305× | 8968 → 3280 | 0.000115 | True | 8 |
| [qwen_float32_reduce-overhead_1](qwen_float32_reduce-overhead_1/results.json) | 128 | decode/32 | 415.650 | 345.569 | 1.203× | 35872 → 13120 | 0.000115 | True | 32 |
| [qwen_float32_reduce-overhead_1](qwen_float32_reduce-overhead_1/results.json) | 512 | prefill/0 | 139.504 | 118.958 | 1.173× | 1337 → 434 | 3.62e-05 | True | 1 |
| [qwen_float32_reduce-overhead_1](qwen_float32_reduce-overhead_1/results.json) | 512 | decode/8 | 106.370 | 90.120 | 1.180× | 8968 → 3280 | 4.1e-05 | True | 8 |
| [qwen_float32_reduce-overhead_1](qwen_float32_reduce-overhead_1/results.json) | 512 | decode/32 | 422.093 | 363.799 | 1.160× | 35872 → 13120 | 5.63e-05 | True | 32 |
| [qwen_float32_no-cudagraphs_2](qwen_float32_no-cudagraphs_2/results.json) | 32 | prefill/0 | 19.931 | 18.284 | 1.090× | 1409 → 506 | 3.72e-05 | True | 0 |
| [qwen_float32_no-cudagraphs_2](qwen_float32_no-cudagraphs_2/results.json) | 32 | decode/8 | 103.084 | 80.711 | 1.277× | 8968 → 3280 | 3.91e-05 | True | 0 |
| [qwen_float32_no-cudagraphs_2](qwen_float32_no-cudagraphs_2/results.json) | 32 | decode/32 | 384.810 | 352.805 | 1.091× | 35872 → 13120 | 0.000191 | True | 0 |
| [qwen_float32_no-cudagraphs_2](qwen_float32_no-cudagraphs_2/results.json) | 44 | prefill/0 | 32.264 | 26.392 | 1.222× | 1385 → 482 | 2.55e-05 | True | 0 |
| [qwen_float32_no-cudagraphs_2](qwen_float32_no-cudagraphs_2/results.json) | 44 | decode/8 | 97.676 | 67.285 | 1.452× | 8968 → 3280 | 6.1e-05 | True | 0 |
| [qwen_float32_no-cudagraphs_2](qwen_float32_no-cudagraphs_2/results.json) | 44 | decode/32 | 395.501 | 344.773 | 1.147× | 35872 → 13120 | 6.1e-05 | True | 0 |
| [qwen_float32_no-cudagraphs_2](qwen_float32_no-cudagraphs_2/results.json) | 128 | prefill/0 | 35.636 | 33.227 | 1.072× | 1337 → 434 | 6.46e-05 | True | 0 |
| [qwen_float32_no-cudagraphs_2](qwen_float32_no-cudagraphs_2/results.json) | 128 | decode/8 | 98.128 | 72.013 | 1.363× | 8968 → 3280 | 0.000115 | True | 0 |
| [qwen_float32_no-cudagraphs_2](qwen_float32_no-cudagraphs_2/results.json) | 128 | decode/32 | 382.969 | 329.292 | 1.163× | 35872 → 13120 | 0.000115 | True | 0 |
| [qwen_float32_no-cudagraphs_2](qwen_float32_no-cudagraphs_2/results.json) | 512 | prefill/0 | 131.267 | 114.398 | 1.147× | 1337 → 434 | 3.62e-05 | True | 0 |
| [qwen_float32_no-cudagraphs_2](qwen_float32_no-cudagraphs_2/results.json) | 512 | decode/8 | 98.065 | 83.228 | 1.178× | 8968 → 3280 | 4.1e-05 | True | 0 |
| [qwen_float32_no-cudagraphs_2](qwen_float32_no-cudagraphs_2/results.json) | 512 | decode/32 | 384.411 | 347.929 | 1.105× | 35872 → 13120 | 5.63e-05 | True | 0 |
| [qwen_float32_reduce-overhead_2](qwen_float32_reduce-overhead_2/results.json) | 32 | prefill/0 | 20.510 | 18.253 | 1.124× | 1409 → 506 | 3.72e-05 | True | 1 |
| [qwen_float32_reduce-overhead_2](qwen_float32_reduce-overhead_2/results.json) | 32 | decode/8 | 105.615 | 67.823 | 1.557× | 8968 → 3280 | 3.91e-05 | True | 8 |
| [qwen_float32_reduce-overhead_2](qwen_float32_reduce-overhead_2/results.json) | 32 | decode/32 | 406.637 | 324.824 | 1.252× | 35872 → 13120 | 0.000191 | True | 32 |
| [qwen_float32_reduce-overhead_2](qwen_float32_reduce-overhead_2/results.json) | 44 | prefill/0 | 32.620 | 24.858 | 1.312× | 1385 → 482 | 2.55e-05 | True | 1 |
| [qwen_float32_reduce-overhead_2](qwen_float32_reduce-overhead_2/results.json) | 44 | decode/8 | 102.104 | 68.233 | 1.496× | 8968 → 3280 | 6.1e-05 | True | 8 |
| [qwen_float32_reduce-overhead_2](qwen_float32_reduce-overhead_2/results.json) | 44 | decode/32 | 405.733 | 338.937 | 1.197× | 35872 → 13120 | 6.1e-05 | True | 32 |
| [qwen_float32_reduce-overhead_2](qwen_float32_reduce-overhead_2/results.json) | 128 | prefill/0 | 35.771 | 33.074 | 1.082× | 1337 → 434 | 6.46e-05 | True | 1 |
| [qwen_float32_reduce-overhead_2](qwen_float32_reduce-overhead_2/results.json) | 128 | decode/8 | 102.691 | 80.364 | 1.278× | 8968 → 3280 | 0.000115 | True | 8 |
| [qwen_float32_reduce-overhead_2](qwen_float32_reduce-overhead_2/results.json) | 128 | decode/32 | 400.123 | 341.435 | 1.172× | 35872 → 13120 | 0.000115 | True | 32 |
| [qwen_float32_reduce-overhead_2](qwen_float32_reduce-overhead_2/results.json) | 512 | prefill/0 | 137.655 | 115.990 | 1.187× | 1337 → 434 | 3.62e-05 | True | 1 |
| [qwen_float32_reduce-overhead_2](qwen_float32_reduce-overhead_2/results.json) | 512 | decode/8 | 103.405 | 87.695 | 1.179× | 8968 → 3280 | 4.1e-05 | True | 8 |
| [qwen_float32_reduce-overhead_2](qwen_float32_reduce-overhead_2/results.json) | 512 | decode/32 | 399.960 | 352.551 | 1.134× | 35872 → 13120 | 5.63e-05 | True | 32 |

GPU kernel durations include profiling overhead. Kernel interval coverage is a trace-derived activity fraction, not measured GPU occupancy/utilization or bandwidth. CUDA Graph settings are requested policy; actual replay calls are reported separately. A backend graph count records compilations across tested shapes, rather than claiming one graph supports every input.

## Numerical rejections

These cases failed the fixed logit tolerance. Steady-state timing and profiling were skipped; they are not accepted performance improvements.

| Run | Prompt | Phase/steps | Maximum error | Mean error | Greedy IDs equal | Tolerance |
|---|---:|---|---:|---:|---|---|
| [qwen_bfloat16_no-cudagraphs_1](qwen_bfloat16_no-cudagraphs_1/results.json) | 32 | prefill/0 | 0.2383 | 0.04292 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_1](qwen_bfloat16_no-cudagraphs_1/results.json) | 32 | decode/8 | 0.25 | 0.03444 | False | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_1](qwen_bfloat16_no-cudagraphs_1/results.json) | 32 | decode/32 | 0.9844 | 0.05457 | False | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_1](qwen_bfloat16_no-cudagraphs_1/results.json) | 44 | prefill/0 | 0.2812 | 0.04987 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_1](qwen_bfloat16_no-cudagraphs_1/results.json) | 44 | decode/8 | 0.375 | 0.0445 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_1](qwen_bfloat16_no-cudagraphs_1/results.json) | 44 | decode/32 | 0.375 | 0.0454 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_1](qwen_bfloat16_no-cudagraphs_1/results.json) | 128 | prefill/0 | 0.375 | 0.05864 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_1](qwen_bfloat16_no-cudagraphs_1/results.json) | 128 | decode/8 | 0.4102 | 0.05052 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_1](qwen_bfloat16_no-cudagraphs_1/results.json) | 128 | decode/32 | 0.4102 | 0.04252 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_1](qwen_bfloat16_no-cudagraphs_1/results.json) | 512 | prefill/0 | 0.4375 | 0.04256 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_1](qwen_bfloat16_no-cudagraphs_1/results.json) | 512 | decode/8 | 0.25 | 0.03769 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_1](qwen_bfloat16_no-cudagraphs_1/results.json) | 512 | decode/32 | 0.5156 | 0.04665 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_1](qwen_bfloat16_reduce-overhead_1/results.json) | 32 | prefill/0 | 0.2383 | 0.04292 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_1](qwen_bfloat16_reduce-overhead_1/results.json) | 32 | decode/8 | 0.25 | 0.03444 | False | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_1](qwen_bfloat16_reduce-overhead_1/results.json) | 32 | decode/32 | 0.9844 | 0.05457 | False | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_1](qwen_bfloat16_reduce-overhead_1/results.json) | 44 | prefill/0 | 0.2812 | 0.04987 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_1](qwen_bfloat16_reduce-overhead_1/results.json) | 44 | decode/8 | 0.375 | 0.0445 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_1](qwen_bfloat16_reduce-overhead_1/results.json) | 44 | decode/32 | 0.375 | 0.0454 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_1](qwen_bfloat16_reduce-overhead_1/results.json) | 128 | prefill/0 | 0.375 | 0.05864 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_1](qwen_bfloat16_reduce-overhead_1/results.json) | 128 | decode/8 | 0.4102 | 0.05052 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_1](qwen_bfloat16_reduce-overhead_1/results.json) | 128 | decode/32 | 0.4102 | 0.04252 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_1](qwen_bfloat16_reduce-overhead_1/results.json) | 512 | prefill/0 | 0.4375 | 0.04256 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_1](qwen_bfloat16_reduce-overhead_1/results.json) | 512 | decode/8 | 0.25 | 0.03769 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_1](qwen_bfloat16_reduce-overhead_1/results.json) | 512 | decode/32 | 0.5156 | 0.04665 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_2](qwen_bfloat16_no-cudagraphs_2/results.json) | 32 | prefill/0 | 0.2383 | 0.04292 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_2](qwen_bfloat16_no-cudagraphs_2/results.json) | 32 | decode/8 | 0.25 | 0.03444 | False | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_2](qwen_bfloat16_no-cudagraphs_2/results.json) | 32 | decode/32 | 0.9844 | 0.05457 | False | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_2](qwen_bfloat16_no-cudagraphs_2/results.json) | 44 | prefill/0 | 0.2812 | 0.04987 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_2](qwen_bfloat16_no-cudagraphs_2/results.json) | 44 | decode/8 | 0.375 | 0.0445 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_2](qwen_bfloat16_no-cudagraphs_2/results.json) | 44 | decode/32 | 0.375 | 0.0454 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_2](qwen_bfloat16_no-cudagraphs_2/results.json) | 128 | prefill/0 | 0.375 | 0.05864 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_2](qwen_bfloat16_no-cudagraphs_2/results.json) | 128 | decode/8 | 0.4102 | 0.05052 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_2](qwen_bfloat16_no-cudagraphs_2/results.json) | 128 | decode/32 | 0.4102 | 0.04252 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_2](qwen_bfloat16_no-cudagraphs_2/results.json) | 512 | prefill/0 | 0.4375 | 0.04256 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_2](qwen_bfloat16_no-cudagraphs_2/results.json) | 512 | decode/8 | 0.25 | 0.03769 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_no-cudagraphs_2](qwen_bfloat16_no-cudagraphs_2/results.json) | 512 | decode/32 | 0.5156 | 0.04665 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_2](qwen_bfloat16_reduce-overhead_2/results.json) | 32 | prefill/0 | 0.2383 | 0.04292 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_2](qwen_bfloat16_reduce-overhead_2/results.json) | 32 | decode/8 | 0.25 | 0.03444 | False | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_2](qwen_bfloat16_reduce-overhead_2/results.json) | 32 | decode/32 | 0.9844 | 0.05457 | False | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_2](qwen_bfloat16_reduce-overhead_2/results.json) | 44 | prefill/0 | 0.2812 | 0.04987 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_2](qwen_bfloat16_reduce-overhead_2/results.json) | 44 | decode/8 | 0.375 | 0.0445 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_2](qwen_bfloat16_reduce-overhead_2/results.json) | 44 | decode/32 | 0.375 | 0.0454 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_2](qwen_bfloat16_reduce-overhead_2/results.json) | 128 | prefill/0 | 0.375 | 0.05864 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_2](qwen_bfloat16_reduce-overhead_2/results.json) | 128 | decode/8 | 0.4102 | 0.05052 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_2](qwen_bfloat16_reduce-overhead_2/results.json) | 128 | decode/32 | 0.4102 | 0.04252 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_2](qwen_bfloat16_reduce-overhead_2/results.json) | 512 | prefill/0 | 0.4375 | 0.04256 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_2](qwen_bfloat16_reduce-overhead_2/results.json) | 512 | decode/8 | 0.25 | 0.03769 | True | atol=0.02, rtol=0.02 |
| [qwen_bfloat16_reduce-overhead_2](qwen_bfloat16_reduce-overhead_2/results.json) | 512 | decode/32 | 0.5156 | 0.04665 | True | atol=0.02, rtol=0.02 |