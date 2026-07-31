from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch

from compilelab.benchmark import capture_dynamo_graph, environment_metadata
from compilelab.codegen import (
    classify_inductor_wrapper,
    find_inductor_wrapper,
    measure_runtime_breakdown,
    profile_cuda_kernels,
)
from compilelab.workload import GatedMLP, PackedGatedMLP, make_mlp_input


CORRECTNESS_ATOL = 2e-2
CORRECTNESS_RTOL = 2e-2


def parse_cases(value: str) -> tuple[tuple[int, int], ...]:
    cases: list[tuple[int, int]] = []
    try:
        for raw_case in value.split(","):
            batch, sequence = raw_case.strip().lower().split("x", maxsplit=1)
            cases.append((int(batch), int(sequence)))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "cases must use comma-separated batchxsequence values"
        ) from exc
    if not cases or any(batch <= 0 or sequence <= 0 for batch, sequence in cases):
        raise argparse.ArgumentTypeError("case dimensions must be positive")
    return tuple(cases)


def error_statistics(
    reference: torch.Tensor, actual: torch.Tensor
) -> dict[str, float | bool]:
    delta = (reference.float() - actual.float()).abs()
    return {
        "allclose": bool(
            torch.allclose(
                reference,
                actual,
                atol=CORRECTNESS_ATOL,
                rtol=CORRECTNESS_RTOL,
            )
        ),
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
    }


def compile_and_capture_wrapper(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    *,
    cache_prefix: str,
    device: torch.device,
) -> tuple[Callable[..., torch.Tensor], torch.Tensor, dict[str, Any], str]:
    previous_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    with tempfile.TemporaryDirectory(prefix=cache_prefix) as cache_name:
        cache_dir = Path(cache_name)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_name
        try:
            compiled = torch.compile(model, fullgraph=True)
            output = compiled(*inputs)
            torch.cuda.synchronize(device)

            _, wrapper_source = find_inductor_wrapper(cache_dir)
            wrapper_analysis = classify_inductor_wrapper(wrapper_source)
            sanitized_wrapper = "\n".join(
                line.rstrip()
                for line in wrapper_source.replace(
                    str(cache_dir), "<inductor-cache>"
                ).splitlines()
            ).rstrip() + "\n"
        finally:
            if previous_cache is None:
                os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
            else:
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = previous_cache

    return compiled, output, wrapper_analysis, sanitized_wrapper


def capture_transformation_graphs(
    *,
    batch_size: int,
    sequence_length: int,
    model_dim: int,
    hidden_dim: int,
    seed: int,
) -> str:
    torch.manual_seed(seed)
    original = GatedMLP(model_dim, hidden_dim).eval()
    packed = PackedGatedMLP.from_unpacked(original).eval()
    inputs = make_mlp_input(
        (batch_size, sequence_length, model_dim),
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=seed,
    )
    with torch.inference_mode():
        original_graphs, original_count = capture_dynamo_graph(original, inputs)
        packed_graphs, packed_count = capture_dynamo_graph(packed, inputs)
    torch.compiler.reset()
    if original_count != 1 or packed_count != 1:
        raise RuntimeError("expected one full Dynamo graph for each workload")
    return (
        "# ---- original GatedMLP ----\n"
        f"{original_graphs[0].strip()}\n\n"
        "# ---- packed GatedMLP ----\n"
        f"{packed_graphs[0].strip()}\n"
    )


def run_case(
    *,
    batch_size: int,
    sequence_length: int,
    model_dim: int,
    hidden_dim: int,
    dtype: torch.dtype,
    dtype_name: str,
    seed: int,
    warmup: int,
    iterations: int,
    profile_iterations: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, str]]:
    torch.compiler.reset()
    torch.manual_seed(seed)
    original = GatedMLP(model_dim, hidden_dim).eval()
    packed = PackedGatedMLP.from_unpacked(original).eval()
    original = original.to(device=device, dtype=dtype)
    packed = packed.to(device=device, dtype=dtype)
    shape = (batch_size, sequence_length, model_dim)
    inputs = make_mlp_input(shape, device=device, dtype=dtype, seed=seed)

    original_parameter_count = sum(
        parameter.numel() for parameter in original.parameters()
    )
    packed_parameter_count = sum(
        parameter.numel() for parameter in packed.parameters()
    )
    if original_parameter_count != packed_parameter_count:
        raise AssertionError("projection packing changed the parameter count")

    with torch.inference_mode():
        reference = original(*inputs)
        packed_eager_output = packed(*inputs)
        (
            original_compiled,
            original_compiled_output,
            original_wrapper,
            original_wrapper_source,
        ) = compile_and_capture_wrapper(
            original,
            inputs,
            cache_prefix="torchcompile-packing-original-",
            device=device,
        )
        (
            packed_compiled,
            packed_compiled_output,
            packed_wrapper,
            packed_wrapper_source,
        ) = compile_and_capture_wrapper(
            packed,
            inputs,
            cache_prefix="torchcompile-packing-packed-",
            device=device,
        )

        correctness = {
            "packed_eager_vs_original_eager": error_statistics(
                reference, packed_eager_output
            ),
            "original_compiled_vs_original_eager": error_statistics(
                reference, original_compiled_output
            ),
            "packed_compiled_vs_original_eager": error_statistics(
                reference, packed_compiled_output
            ),
        }
        if not all(check["allclose"] for check in correctness.values()):
            raise AssertionError("a packed or compiled output did not match eager")

        variants = {
            "original_eager": lambda: original(*inputs),
            "packed_eager": lambda: packed(*inputs),
            "original_compiled": lambda: original_compiled(*inputs),
            "packed_compiled": lambda: packed_compiled(*inputs),
        }
        runtime = measure_runtime_breakdown(
            variants,
            device=device,
            warmup=warmup,
            iterations=iterations,
        )
        profiles = {
            name: profile_cuda_kernels(
                function,
                device=device,
                iterations=profile_iterations,
            )
            for name, function in variants.items()
        }

    original_compiled_us = runtime["original_compiled"][
        "synchronized_total"
    ]["median_us"]
    packed_compiled_us = runtime["packed_compiled"]["synchronized_total"][
        "median_us"
    ]
    original_eager_us = runtime["original_eager"]["synchronized_total"][
        "median_us"
    ]
    packed_eager_us = runtime["packed_eager"]["synchronized_total"][
        "median_us"
    ]
    result = {
        "shape": list(shape),
        "token_count": batch_size * sequence_length,
        "dtype": dtype_name,
        "parameter_count": original_parameter_count,
        "correctness": correctness,
        "inductor_wrapper": {
            "original": original_wrapper,
            "packed": packed_wrapper,
        },
        "runtime_breakdown": runtime,
        "kernel_profile": profiles,
        "comparison": {
            "eager_speedup": original_eager_us / packed_eager_us,
            "compiled_speedup": original_compiled_us / packed_compiled_us,
            "compiled_median_delta_us": packed_compiled_us - original_compiled_us,
            "compiled_launch_reduction": (
                profiles["original_compiled"]["launches_per_invocation"]
                - profiles["packed_compiled"]["launches_per_invocation"]
            ),
        },
    }
    return result, {
        "original": original_wrapper_source,
        "packed": packed_wrapper_source,
    }


def render_summary(result: dict[str, Any]) -> str:
    cases = result["cases"]
    representative = cases[result["representative_case_index"]]
    original_wrapper = representative["inductor_wrapper"]["original"]
    packed_wrapper = representative["inductor_wrapper"]["packed"]
    original_profile = representative["kernel_profile"]
    original_external_calls = original_wrapper["external_kernel_call_count"]
    packed_external_calls = packed_wrapper["external_kernel_call_count"]
    original_triton_launches = original_wrapper["triton_kernel_launch_count"]
    packed_triton_launches = packed_wrapper["triton_kernel_launch_count"]
    original_compiled_launches = original_profile["original_compiled"][
        "launches_per_invocation"
    ]
    packed_compiled_launches = original_profile["packed_compiled"][
        "launches_per_invocation"
    ]
    original_eager_launches = original_profile["original_eager"][
        "launches_per_invocation"
    ]
    packed_eager_launches = original_profile["packed_eager"][
        "launches_per_invocation"
    ]
    structural_rows = [
        f"| External GEMM calls in Inductor wrapper | {original_external_calls} | "
        f"{packed_external_calls} |",
        f"| Triton kernel launches in wrapper | {original_triton_launches} | "
        f"{packed_triton_launches} |",
        f"| Compiled CUDA launches per invocation | "
        f"{original_compiled_launches:.1f} | {packed_compiled_launches:.1f} |",
        f"| Eager CUDA launches per invocation | {original_eager_launches:.1f} | "
        f"{packed_eager_launches:.1f} |",
    ]
    latency_rows = []
    for case in cases:
        original_us = case["runtime_breakdown"]["original_compiled"][
            "synchronized_total"
        ]["median_us"]
        packed_us = case["runtime_breakdown"]["packed_compiled"][
            "synchronized_total"
        ]["median_us"]
        latency_rows.append(
            f"| {' × '.join(str(value) for value in case['shape'])} | "
            f"{original_us:.2f} µs | {packed_us:.2f} µs | "
            f"{case['comparison']['compiled_speedup']:.3f}× |"
        )

    wins = [case for case in cases if case["comparison"]["compiled_speedup"] > 1.0]
    strongest = max(cases, key=lambda case: case["comparison"]["compiled_speedup"])
    strongest_shape = " × ".join(str(value) for value in strongest["shape"])
    strongest_speedup = strongest["comparison"]["compiled_speedup"]
    if wins:
        performance_statement = (
            f"Packed compiled execution was faster in {len(wins)} of "
            f"{len(cases)} measured shapes. The largest observed speedup was "
            f"{strongest_speedup:.3f}× at shape `{strongest_shape}`."
        )
    else:
        performance_statement = (
            "Projection packing reduced the generated call structure but did not "
            "improve compiled latency in the measured shapes."
        )

    eager_wins = [
        case for case in cases if case["comparison"]["eager_speedup"] > 1.0
    ]
    packed_compiled_beats_eager = [
        case
        for case in cases
        if case["runtime_breakdown"]["original_eager"]["synchronized_total"][
            "median_us"
        ]
        > case["runtime_breakdown"]["packed_compiled"]["synchronized_total"][
            "median_us"
        ]
    ]
    if not eager_wins and not packed_compiled_beats_eager:
        scope_statement = (
            "Packing did not improve eager latency, and the packed compiled path "
            "did not beat original eager execution in these cases. The measured "
            "win is specifically packed compiled versus original compiled."
        )
    else:
        scope_statement = (
            f"Packing improved eager latency in {len(eager_wins)} cases, and packed "
            f"compiled execution beat original eager in "
            f"{len(packed_compiled_beats_eager)} cases."
        )
    maximum_error = max(
        check["max_abs_error"]
        for case in cases
        for check in case["correctness"].values()
    )
    representative_runtime = representative["runtime_breakdown"]
    original_enqueue = representative_runtime["original_compiled"][
        "host_enqueue"
    ]["median_us"]
    packed_enqueue = representative_runtime["packed_compiled"]["host_enqueue"][
        "median_us"
    ]
    original_wait = representative_runtime["original_compiled"][
        "completion_wait"
    ]["median_us"]
    packed_wait = representative_runtime["packed_compiled"]["completion_wait"][
        "median_us"
    ]
    if packed_enqueue < original_enqueue and packed_wait > original_wait:
        component_statement = (
            "The lower host enqueue time outweighed the longer GPU completion "
            "wait for this shape."
        )
    else:
        component_statement = (
            "The component medians show how host submission and GPU completion "
            "contributed to the total change."
        )

    return f"""# Gated MLP projection-packing analysis

The transformation concatenates the gate and up projection weights once during
model conversion. The packed model computes both projections with one linear
operation, splits the result into views, and preserves the original down
projection. Parameter count remains {representative['parameter_count']:,}.

## Generated execution

| Structural metric | Original | Packed |
|---|---:|---:|
{chr(10).join(structural_rows)}

The representative generated wrapper changes from three external GEMM calls to
two while retaining one fused Triton SiLU/multiply kernel. The split is a view
and does not introduce another profiled CUDA launch.

## Compiled latency

| Input shape | Original | Packed | Speedup |
|---|---:|---:|---:|
{chr(10).join(latency_rows)}

{performance_statement}

{scope_statement}

For the representative shape, median host enqueue changed from
{original_enqueue:.2f} µs to {packed_enqueue:.2f} µs and post-enqueue completion
wait changed from {original_wait:.2f} µs to {packed_wait:.2f} µs.
{component_statement}

All eager and compiled outputs passed {result['experiment']['dtype']} tolerance
against the original model; the maximum observed absolute error was
{maximum_error:.3e}. The result is specific to the recorded shapes, software
stack, and RTX 5050 Laptop GPU.
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for projection-packing analysis")
    if args.warmup < 1 or args.iterations < 1 or args.profile_iterations < 1:
        raise ValueError("warmup and iteration counts must be positive")
    if args.model_dim <= 0 or args.hidden_dim <= 0:
        raise ValueError("model and hidden dimensions must be positive")

    device = torch.device("cuda")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    representative_batch, representative_sequence = args.cases[-1]
    graph_artifact = capture_transformation_graphs(
        batch_size=representative_batch,
        sequence_length=representative_sequence,
        model_dim=args.model_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )

    cases = []
    representative_sources: dict[str, str] | None = None
    for case_index, (batch_size, sequence_length) in enumerate(args.cases):
        case, wrapper_sources = run_case(
            batch_size=batch_size,
            sequence_length=sequence_length,
            model_dim=args.model_dim,
            hidden_dim=args.hidden_dim,
            dtype=dtype,
            dtype_name=args.dtype,
            seed=args.seed,
            warmup=args.warmup,
            iterations=args.iterations,
            profile_iterations=args.profile_iterations,
            device=device,
        )
        cases.append(case)
        if case_index == len(args.cases) - 1:
            representative_sources = wrapper_sources

    if representative_sources is None:
        raise AssertionError("no representative wrapper was captured")
    result = {
        "schema_version": 1,
        "run_utc": datetime.now(UTC).isoformat(),
        "environment": environment_metadata(device),
        "experiment": {
            "transformation": "pack gate_proj and up_proj",
            "model_dim": args.model_dim,
            "hidden_dim": args.hidden_dim,
            "dtype": args.dtype,
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "profile_iterations": args.profile_iterations,
            "correctness_atol": CORRECTNESS_ATOL,
            "correctness_rtol": CORRECTNESS_RTOL,
            "fullgraph": True,
            "isolated_inductor_cache_per_variant": True,
            "measurement_order": "alternating forward and reverse",
        },
        "representative_case_index": len(cases) - 1,
        "cases": cases,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(
        render_summary(result), encoding="utf-8"
    )
    (args.output_dir / "dynamo_graphs.py").write_text(
        graph_artifact, encoding="utf-8"
    )
    (args.output_dir / "original_inductor.py").write_text(
        representative_sources["original"], encoding="utf-8"
    )
    (args.output_dir / "packed_inductor.py").write_text(
        representative_sources["packed"], encoding="utf-8"
    )
    torch.compiler.reset()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure gated-MLP projection packing on CUDA"
    )
    parser.add_argument(
        "--cases",
        type=parse_cases,
        default=((1, 1), (1, 128), (4, 128)),
        help="comma-separated batchxsequence cases",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--model-dim", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--profile-iterations", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/mlp_projection_packing"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(render_summary(result))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
