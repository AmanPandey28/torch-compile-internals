from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from compilelab.benchmark import (
    build_workload,
    environment_metadata,
    parse_shape,
    resolve_device,
    resolve_dtype,
    synchronize,
)
from compilelab.trace_analysis import summarize_trace

COMPILE_MODES = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)


def resolve_profile_dtype(
    requested: str,
    *,
    workload: str,
    device: torch.device,
) -> tuple[str, torch.dtype]:
    if requested == "auto":
        dtype_name = (
            "float16"
            if workload == "gated-mlp" and device.type == "cuda"
            else "float32"
        )
    else:
        dtype_name = requested
    return dtype_name, resolve_dtype(dtype_name, device)


def summarize_device_events(trace_path: Path) -> dict[str, Any]:
    return summarize_trace(trace_path)


def capture_trace(
    *,
    name: str,
    function: Callable[[], torch.Tensor],
    device: torch.device,
    iterations: int,
    output_dir: Path,
    record_shapes: bool,
    profile_memory: bool,
    with_stack: bool,
    row_limit: int,
) -> dict[str, Any]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    synchronize(device)
    with profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        acc_events=True,
    ) as captured:
        for _ in range(iterations):
            with record_function(f"{name}_iteration"):
                function()
        synchronize(device)

    trace_path = output_dir / f"{name}.json"
    table_path = output_dir / f"{name}_table.txt"
    captured.export_chrome_trace(str(trace_path))

    sort_key = (
        "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    )
    table = captured.key_averages(group_by_input_shape=record_shapes).table(
        sort_by=sort_key,
        row_limit=row_limit,
    )
    table_path.write_text(table + "\n", encoding="utf-8")

    device_events = summarize_device_events(trace_path)
    print(f"\n{name.upper()} PROFILER TABLE")
    print(table)
    print(f"Trace: {trace_path.resolve()}")
    print(f"Table: {table_path.resolve()}")

    return {
        "trace": trace_path.name,
        "table": table_path.name,
        "profiled_iterations": iterations,
        "device_launch_count": device_events["launch_count"],
        "device_launches_per_iteration": (
            device_events["launch_count"] / iterations
            if device.type == "cuda"
            else None
        ),
        "device_kernels": device_events["kernels"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.warmup < 1 or args.iterations < 1 or args.row_limit < 1:
        raise ValueError("warmup, iterations, and row limit must be positive")

    device = resolve_device(args.device)
    dtype_name, dtype = resolve_profile_dtype(
        args.dtype,
        workload=args.workload,
        device=device,
    )
    args.dtype = dtype_name
    model, inputs, workload_metadata = build_workload(
        args,
        device=device,
        dtype=dtype,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    compile_mode = None if args.compile_mode == "default" else args.compile_mode
    first_call_ms: float | None = None
    correctness: dict[str, Any] | None = None
    variants: dict[str, Callable[[], torch.Tensor]] = {}

    with torch.inference_mode():
        eager_output = model(*inputs)
        if args.variant in {"eager", "both"}:
            variants["eager"] = lambda: model(*inputs)

        if args.variant in {"compiled", "both"}:
            compiled_model = torch.compile(
                model,
                fullgraph=True,
                mode=compile_mode,
            )
            synchronize(device)
            start = time.perf_counter()
            compiled_output = compiled_model(*inputs)
            synchronize(device)
            first_call_ms = (time.perf_counter() - start) * 1_000.0

            tolerance = 1e-4 if dtype == torch.float32 else 2e-2
            delta = (eager_output.float() - compiled_output.float()).abs()
            allclose = bool(
                torch.allclose(
                    eager_output,
                    compiled_output,
                    atol=tolerance,
                    rtol=tolerance,
                )
            )
            if not allclose:
                raise AssertionError("compiled output does not match eager output")
            correctness = {
                "allclose": True,
                "atol": tolerance,
                "rtol": tolerance,
                "max_abs_error": float(delta.max()),
                "mean_abs_error": float(delta.mean()),
            }
            variants["compiled"] = lambda: compiled_model(*inputs)

        for function in variants.values():
            for _ in range(args.warmup):
                function()
        synchronize(device)

        traces = {
            name: capture_trace(
                name=f"{name}_{args.workload.replace('-', '_')}",
                function=function,
                device=device,
                iterations=args.iterations,
                output_dir=args.output_dir,
                record_shapes=args.record_shapes,
                profile_memory=args.profile_memory,
                with_stack=args.with_stack,
                row_limit=args.row_limit,
            )
            for name, function in variants.items()
        }

    result = {
        "schema_version": 1,
        "run_utc": datetime.now(UTC).isoformat(),
        "environment": environment_metadata(device),
        "experiment": {
            **workload_metadata,
            "variant": args.variant,
            "dtype": dtype_name,
            "seed": args.seed,
            "warmup": args.warmup,
            "profiled_iterations": args.iterations,
            "fullgraph": True if args.variant != "eager" else None,
            "compile_mode": args.compile_mode,
            "record_shapes": args.record_shapes,
            "profile_memory": args.profile_memory,
            "with_stack": args.with_stack,
        },
        "correctness": correctness,
        "compiled_first_call_ms": first_call_ms,
        "profiles": traces,
        "notes": [
            "Compilation and warm-up occurred before profiler capture.",
            (
                "Profiler timings include instrumentation overhead and are not "
                "benchmark latencies."
            ),
            (
                "Use the Chrome trace for execution attribution and the benchmark "
                "harness for latency conclusions."
            ),
        ],
    }
    metadata_path = args.output_dir / "profile_metadata.json"
    metadata_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Metadata: {metadata_path.resolve()}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export eager and compiled Torch Profiler traces"
    )
    parser.add_argument(
        "--workload",
        choices=("pointwise", "gated-mlp"),
        default="gated-mlp",
    )
    parser.add_argument(
        "--variant",
        choices=("eager", "compiled", "both"),
        default="both",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="auto selects float16 for the CUDA MLP and float32 otherwise",
    )
    parser.add_argument("--shape", type=parse_shape, default=(1024, 4096))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--model-dim", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="number of invocations included in each profiler trace",
    )
    parser.add_argument(
        "--compile-mode",
        choices=COMPILE_MODES,
        default="default",
    )
    parser.add_argument(
        "--record-shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--profile-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/profiler_trace"),
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
