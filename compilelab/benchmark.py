from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import statistics
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch

from compilelab.workload import GatedMLP, PointwiseReduction, make_inputs, make_mlp_input


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(samples: list[float], quantile: float) -> float:
    if not samples:
        raise ValueError("samples cannot be empty")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def time_once(function: Callable[[], torch.Tensor], device: torch.device) -> float:
    synchronize(device)
    start = time.perf_counter()
    function()
    synchronize(device)
    return (time.perf_counter() - start) * 1_000.0


def measure(
    function: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    synchronize(device)

    samples = [time_once(function, device) for _ in range(iterations)]
    return {
        "median_ms": statistics.median(samples),
        "p90_ms": percentile(samples, 0.90),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def capture_dynamo_graph(
    model: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
) -> tuple[list[str], int]:
    graph_code: list[str] = []

    def capture_backend(
        graph_module: torch.fx.GraphModule,
        _example_inputs: list[torch.Tensor],
    ) -> Callable[..., Any]:
        graph_code.append(graph_module.code)
        return graph_module.forward

    captured = torch.compile(model, backend=capture_backend, fullgraph=True)
    captured(*args)
    return graph_code, len(graph_code)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 is not supported by this CPU baseline")
    return dtype


def parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be comma-separated integers") from exc
    if len(shape) < 2 or any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError(
            "shape must contain at least two positive dimensions"
        )
    return shape


def break_even_calls(
    first_call_ms: float,
    eager_median_ms: float,
    compiled_median_ms: float,
) -> int | None:
    saving_per_call = eager_median_ms - compiled_median_ms
    compile_overhead = max(0.0, first_call_ms - compiled_median_ms)
    if saving_per_call <= 0.0:
        return None
    return math.ceil(compile_overhead / saving_per_call)


def environment_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        metadata["device_name"] = torch.cuda.get_device_name(device)
        metadata["compute_capability"] = list(
            torch.cuda.get_device_capability(device)
        )
    else:
        metadata["device_name"] = platform.processor() or "CPU"
        metadata["cpu_threads"] = torch.get_num_threads()
    return metadata


def build_workload(
    args: argparse.Namespace,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...], dict[str, Any]]:
    torch.manual_seed(args.seed)
    if args.workload == "pointwise":
        model = PointwiseReduction()
        inputs = make_inputs(
            args.shape, device=device, dtype=dtype, seed=args.seed
        )
        metadata = {
            "workload": "PointwiseReduction",
            "shape": list(args.shape),
        }
    else:
        shape = (args.batch_size, args.sequence_length, args.model_dim)
        model = GatedMLP(args.model_dim, args.hidden_dim)
        inputs = make_mlp_input(
            shape, device=device, dtype=dtype, seed=args.seed
        )
        metadata = {
            "workload": "GatedMLP",
            "shape": list(shape),
            "model_dim": args.model_dim,
            "hidden_dim": args.hidden_dim,
        }

    model = model.to(device=device, dtype=dtype).eval()
    metadata["parameter_count"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    return model, inputs, metadata


def render_summary(result: dict[str, Any]) -> str:
    eager = result["timing"]["eager"]
    compiled = result["timing"]["compiled"]
    speedup = result["timing"]["steady_state_speedup"]
    break_even = result["timing"]["break_even_calls"]
    isolated_cache = result["experiment"]["isolated_inductor_cache"]
    break_even_text = (
        f"~{break_even} calls"
        if break_even is not None
        else "not reached (compiled was not faster)"
    )
    cache_text = "fresh temporary cache" if isolated_cache else "existing cache"
    shape = " × ".join(str(value) for value in result["experiment"]["shape"])

    return f"""# Benchmark result

This result applies only to the environment and shape recorded below.

| Field | Value |
|---|---:|
| Workload | {result["experiment"]["workload"]} |
| Device | {result["environment"]["device_name"]} |
| PyTorch | {result["environment"]["torch"]} |
| Shape | {shape} |
| Parameters | {result["experiment"]["parameter_count"]:,} |
| Dtype | {result["experiment"]["dtype"]} |
| Inductor cache | {cache_text} |
| Dynamo graphs captured | {result["graph_capture"]["graph_count"]} |
| Maximum absolute error | {result["correctness"]["max_abs_error"]:.3e} |
| Eager median | {eager["median_ms"]:.4f} ms |
| Compiled first call | {compiled["first_call_ms"]:.4f} ms |
| Compiled steady-state median | {compiled["median_ms"]:.4f} ms |
| Steady-state speedup | {speedup:.3f}× |
| Estimated break-even | {break_even_text} |

The first call was measured with a {cache_text}; it can include tracing, code
generation, compilation, and initial allocations. The speedup compares
steady-state medians after warm-up. The break-even value is an estimate, not a
production SLA.
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model, inputs, workload_metadata = build_workload(
        args, device=device, dtype=dtype
    )

    with torch.inference_mode():
        eager_output = model(*inputs)
        eager_timing = measure(
            lambda: model(*inputs),
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
        )

        compiled_model = torch.compile(model, fullgraph=True)
        compiled_first_call_ms = time_once(
            lambda: compiled_model(*inputs), device
        )
        compiled_output = compiled_model(*inputs)

        tolerance = 1e-4 if dtype == torch.float32 else 2e-2
        if not torch.allclose(
            eager_output, compiled_output, atol=tolerance, rtol=tolerance
        ):
            raise AssertionError("compiled output does not match eager output")

        compiled_timing = measure(
            lambda: compiled_model(*inputs),
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        graph_code, graph_count = capture_dynamo_graph(model, inputs)

    delta = (eager_output.float() - compiled_output.float()).abs()
    eager_median = eager_timing["median_ms"]
    compiled_median = compiled_timing["median_ms"]
    result = {
        "schema_version": 1,
        "run_utc": datetime.now(UTC).isoformat(),
        "environment": environment_metadata(device),
        "experiment": {
            **workload_metadata,
            "dtype": str(dtype).removeprefix("torch."),
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "fullgraph": True,
            "isolated_inductor_cache": args.isolated_cache,
        },
        "correctness": {
            "allclose": True,
            "atol": tolerance,
            "rtol": tolerance,
            "max_abs_error": float(delta.max()),
            "mean_abs_error": float(delta.mean()),
        },
        "graph_capture": {
            "graph_count": graph_count,
            "backend": "custom eager backend used only for inspection",
        },
        "timing": {
            "eager": eager_timing,
            "compiled": {
                "first_call_ms": compiled_first_call_ms,
                **compiled_timing,
            },
            "steady_state_speedup": eager_median / compiled_median,
            "break_even_calls": break_even_calls(
                compiled_first_call_ms, eager_median, compiled_median
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(
        render_summary(result), encoding="utf-8"
    )
    serialized_graphs = "\n\n# ---- captured graph ----\n\n".join(
        code.strip() for code in graph_code
    )
    (args.output_dir / "dynamo_fx_graph.py").write_text(
        serialized_graphs + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark eager execution against torch.compile"
    )
    parser.add_argument(
        "--workload",
        choices=("pointwise", "gated-mlp"),
        default="pointwise",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--shape", type=parse_shape, default=(1024, 4096))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--model-dim", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--isolated-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use a fresh temporary Inductor cache (default: enabled)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/latest")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    previous_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    cache_context: Any
    if args.isolated_cache:
        cache_context = tempfile.TemporaryDirectory(prefix="torchcompile-lab-")
    else:
        cache_context = contextlib.nullcontext(None)

    with cache_context as temporary_cache:
        if temporary_cache is not None:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = temporary_cache
        try:
            result = run(args)
        finally:
            if previous_cache is None:
                os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
            else:
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = previous_cache
    print(render_summary(result))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
