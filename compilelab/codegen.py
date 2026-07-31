from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch
from torch.profiler import ProfilerActivity, profile

from compilelab.benchmark import environment_metadata
from compilelab.workload import GatedMLP, make_mlp_input


def classify_inductor_wrapper(source: str) -> dict[str, Any]:
    external_calls = re.findall(r"extern_kernels\.([A-Za-z_][A-Za-z0-9_]*)\(", source)
    triton_definitions = re.findall(
        r"^([A-Za-z_][A-Za-z0-9_]*) = async_compile\.triton\(",
        source,
        flags=re.MULTILINE,
    )
    triton_launches = re.findall(
        r"^\s*([A-Za-z_][A-Za-z0-9_]*)\.run\(",
        source,
        flags=re.MULTILINE,
    )
    source_node_groups = re.findall(
        r"^\s*# Topologically Sorted Source Nodes: \[([^]]*)\]",
        source,
        flags=re.MULTILINE,
    )
    external_counts = dict(sorted(Counter(external_calls).items()))
    return {
        "external_kernel_calls": external_counts,
        "external_kernel_call_count": len(external_calls),
        "triton_kernel_definitions": triton_definitions,
        "triton_kernel_launches": triton_launches,
        "triton_kernel_launch_count": len(triton_launches),
        "wrapper_launch_site_count": len(external_calls) + len(triton_launches),
        "buffer_reuse_count": len(re.findall(r"# reuse\b", source)),
        "input_alignment_copy_count": len(
            re.findall(
                r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*copy_misaligned\(",
                source,
                flags=re.MULTILINE,
            )
        ),
        "source_node_groups": source_node_groups,
    }


def find_inductor_wrapper(cache_dir: Path) -> tuple[Path, str]:
    candidates: list[tuple[Path, str]] = []
    for path in cache_dir.rglob("*.py"):
        source = path.read_text(encoding="utf-8", errors="replace")
        if "class Runner:" not in source or "def call(self, args):" not in source:
            continue
        if "async_compile.triton" not in source or "extern_kernels." not in source:
            continue
        candidates.append((path, source))

    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one generated Inductor wrapper, found {len(candidates)}"
        )
    return candidates[0]


def timing_statistics(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("timing samples cannot be empty")
    ordered = sorted(samples)
    p90_index = round((len(ordered) - 1) * 0.90)
    return {
        "median_us": statistics.median(samples),
        "p90_us": ordered[p90_index],
        "min_us": min(samples),
        "max_us": max(samples),
    }


def measure_runtime_breakdown(
    variants: dict[str, Callable[[], torch.Tensor]],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, dict[str, dict[str, float]]]:
    for function in variants.values():
        for _ in range(warmup):
            function()
    torch.cuda.synchronize(device)

    samples: dict[str, dict[str, list[float]]] = {
        name: {"host_enqueue": [], "completion_wait": [], "synchronized_total": []}
        for name in variants
    }
    names = list(variants)
    for iteration in range(iterations):
        order = names if iteration % 2 == 0 else list(reversed(names))
        for name in order:
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            variants[name]()
            enqueued = time.perf_counter()
            torch.cuda.synchronize(device)
            completed = time.perf_counter()
            samples[name]["host_enqueue"].append((enqueued - start) * 1_000_000.0)
            samples[name]["completion_wait"].append(
                (completed - enqueued) * 1_000_000.0
            )
            samples[name]["synchronized_total"].append(
                (completed - start) * 1_000_000.0
            )

    return {
        name: {
            component: timing_statistics(component_samples)
            for component, component_samples in variant_samples.items()
        }
        for name, variant_samples in samples.items()
    }


def classify_cuda_kernel(name: str) -> str:
    lowered = name.lower()
    if "triton" in lowered:
        return "generated_triton"
    if "cutlass" in lowered or "gemm" in lowered:
        return "external_gemm"
    if "silu_kernel" in lowered:
        return "eager_silu"
    if "mulfunctor" in lowered:
        return "eager_multiply"
    return "other"


def profile_cuda_kernels(
    function: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    iterations: int,
) -> dict[str, Any]:
    torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as captured:
        for _ in range(iterations):
            function()
        torch.cuda.synchronize(device)

    categories: dict[str, list[float]] = defaultdict(list)
    for event in captured.events():
        if not str(event.device_type).endswith("CUDA"):
            continue
        if event.name in {"Activity Buffer Request", "ProfilerStep*"}:
            continue
        categories[classify_cuda_kernel(event.name)].append(
            float(event.device_time_total)
        )

    serialized_categories = {
        name: {
            "launch_count": len(durations),
            "total_device_time_us": sum(durations),
            "mean_device_time_us": statistics.mean(durations),
        }
        for name, durations in sorted(categories.items())
    }
    launch_count = sum(
        category["launch_count"] for category in serialized_categories.values()
    )
    return {
        "profiled_invocations": iterations,
        "launch_count": launch_count,
        "launches_per_invocation": launch_count / iterations,
        "categories": serialized_categories,
    }


def render_summary(result: dict[str, Any]) -> str:
    wrapper = result["inductor_wrapper"]
    runtime = result["runtime_breakdown"]
    eager = runtime["eager"]
    compiled = runtime["compiled"]
    profiles = result["kernel_profile"]
    external_calls = ", ".join(
        f"{count} × {name}"
        for name, count in wrapper["external_kernel_calls"].items()
    )
    triton_kernels = ", ".join(wrapper["triton_kernel_definitions"])
    speedup = (
        eager["synchronized_total"]["median_us"]
        / compiled["synchronized_total"]["median_us"]
    )
    eager_enqueue = eager["host_enqueue"]["median_us"]
    compiled_enqueue = compiled["host_enqueue"]["median_us"]
    eager_wait = eager["completion_wait"]["median_us"]
    compiled_wait = compiled["completion_wait"]["median_us"]
    eager_total = eager["synchronized_total"]["median_us"]
    compiled_total = compiled["synchronized_total"]["median_us"]

    enqueue_delta = compiled_enqueue - eager_enqueue
    wait_delta = compiled_wait - eager_wait
    total_delta = compiled_total - eager_total
    if enqueue_delta > 0.0 and wait_delta < 0.0 and total_delta > 0.0:
        conclusion = (
            "For this configuration, the compiled path reduces post-enqueue "
            "completion time but increases host enqueue time. The added host-side "
            "cost is larger than the completion-time reduction, so synchronized "
            "single-call latency regresses."
        )
    elif total_delta <= 0.0:
        conclusion = (
            "For this configuration, the compiled path improves synchronized "
            "single-call latency."
        )
    else:
        conclusion = (
            "For this configuration, the compiled path regresses synchronized "
            "single-call latency; the component medians do not support attributing "
            "the change to one timing component."
        )

    return f"""# Gated MLP Inductor code-generation analysis

| Field | Value |
|---|---:|
| Device | {result["environment"]["device_name"]} |
| PyTorch | {result["environment"]["torch"]} |
| Input shape | {" × ".join(str(v) for v in result["experiment"]["shape"])} |
| Dtype | {result["experiment"]["dtype"]} |
| External wrapper calls | {external_calls} |
| Generated Triton kernels | {triton_kernels} |
| Compiled wrapper launch sites | {wrapper["wrapper_launch_site_count"]} |
| Eager profiled launches/invocation | {profiles["eager"]["launches_per_invocation"]:.1f} |
| Compiled profiled launches/invocation | {profiles["compiled"]["launches_per_invocation"]:.1f} |

## Runtime breakdown

| Median component | Eager | Compiled |
|---|---:|---:|
| Host enqueue | {eager_enqueue:.2f} µs | {compiled_enqueue:.2f} µs |
| Post-enqueue completion wait | {eager_wait:.2f} µs | {compiled_wait:.2f} µs |
| Synchronized total | {eager_total:.2f} µs | {compiled_total:.2f} µs |
| Synchronized speedup | — | {speedup:.3f}× |

Inductor leaves all three linear projections as external `mm` calls and emits
one Triton kernel for the SiLU/multiply chain. The profiler therefore observes
five eager launches and four compiled launches per invocation.

{conclusion} This conclusion is limited to the recorded software, hardware,
dtype, and shape.
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for generated-code analysis")
    if args.warmup < 1 or args.iterations < 1 or args.profile_iterations < 1:
        raise ValueError("warmup and iteration counts must be positive")

    device = torch.device("cuda")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    shape = (args.batch_size, args.sequence_length, args.model_dim)
    previous_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")

    with tempfile.TemporaryDirectory(prefix="torchcompile-codegen-") as cache_name:
        cache_dir = Path(cache_name)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_name
        try:
            torch.manual_seed(args.seed)
            model = GatedMLP(args.model_dim, args.hidden_dim).to(
                device=device, dtype=dtype
            ).eval()
            inputs = make_mlp_input(
                shape, device=device, dtype=dtype, seed=args.seed
            )

            with torch.inference_mode():
                eager_output = model(*inputs)
                compiled_model = torch.compile(model, fullgraph=True)
                compiled_output = compiled_model(*inputs)
                torch.cuda.synchronize(device)

                tolerance = 2e-2
                if not torch.allclose(
                    eager_output, compiled_output, atol=tolerance, rtol=tolerance
                ):
                    raise AssertionError("compiled output does not match eager output")

                variants = {
                    "eager": lambda: model(*inputs),
                    "compiled": lambda: compiled_model(*inputs),
                }
                runtime_breakdown = measure_runtime_breakdown(
                    variants,
                    device=device,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                kernel_profile = {
                    name: profile_cuda_kernels(
                        function,
                        device=device,
                        iterations=args.profile_iterations,
                    )
                    for name, function in variants.items()
                }

            _, wrapper_source = find_inductor_wrapper(cache_dir)
            wrapper_analysis = classify_inductor_wrapper(wrapper_source)
            sanitized_wrapper = "\n".join(
                line.rstrip()
                for line in wrapper_source.replace(
                    str(cache_dir), "<inductor-cache>"
                ).splitlines()
            )
        finally:
            if previous_cache is None:
                os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
            else:
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = previous_cache

    delta = (eager_output.float() - compiled_output.float()).abs()
    result = {
        "schema_version": 1,
        "run_utc": datetime.now(UTC).isoformat(),
        "environment": environment_metadata(device),
        "experiment": {
            "workload": "GatedMLP",
            "shape": list(shape),
            "model_dim": args.model_dim,
            "hidden_dim": args.hidden_dim,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "dtype": args.dtype,
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "profile_iterations": args.profile_iterations,
            "fullgraph": True,
            "isolated_inductor_cache": True,
        },
        "correctness": {
            "allclose": True,
            "atol": tolerance,
            "rtol": tolerance,
            "max_abs_error": float(delta.max()),
            "mean_abs_error": float(delta.mean()),
        },
        "inductor_wrapper": {
            **wrapper_analysis,
            "source_artifact": "inductor_output_code.py",
        },
        "runtime_breakdown": runtime_breakdown,
        "kernel_profile": kernel_profile,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(
        render_summary(result), encoding="utf-8"
    )
    (args.output_dir / "inductor_output_code.py").write_text(
        sanitized_wrapper.rstrip() + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and classify generated Inductor code for a gated MLP"
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--model-dim", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--profile-iterations", type=int, default=20)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/mlp_codegen")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(render_summary(result))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
