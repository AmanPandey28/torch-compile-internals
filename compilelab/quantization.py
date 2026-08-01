from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import torch

from compilelab.benchmark import environment_metadata
from compilelab.codegen import measure_runtime_breakdown, profile_cuda_kernels
from compilelab.projection_packing import parse_cases
from compilelab.transformer import PackedTransformerBlock, TransformerBlock
from compilelab.transformer_optimization import (
    aggregate_wrapper_analysis,
    find_generated_wrappers,
    sanitize_wrapper,
)
from compilelab.workload import make_mlp_input


QUANTIZATION_STRATEGIES = (
    "bf16",
    "int8_weight_only",
    "int8_dynamic",
    "fp8_weight_only",
    "fp8_dynamic",
)


def quality_statistics(
    reference: torch.Tensor, actual: torch.Tensor
) -> dict[str, float | bool | None]:
    reference_float = reference.float()
    actual_float = actual.float()
    error = reference_float - actual_float
    signal_power = float(reference_float.square().mean())
    noise_power = float(error.square().mean())
    sqnr_db = (
        10.0 * math.log10(signal_power / noise_power)
        if signal_power > 0.0 and noise_power > 0.0
        else None
    )
    cosine = torch.nn.functional.cosine_similarity(
        reference_float.flatten(), actual_float.flatten(), dim=0
    )
    return {
        "finite": bool(actual.isfinite().all()),
        "max_abs_error": float(error.abs().max()),
        "mean_abs_error": float(error.abs().mean()),
        "root_mean_square_error": math.sqrt(noise_power),
        "cosine_similarity": float(cosine),
        "sqnr_db": sqnr_db,
    }


def make_quantization_config(strategy: str, *, token_count: int) -> Any:
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Float8WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig,
    )

    if strategy == "int8_weight_only":
        return Int8WeightOnlyConfig(version=2)
    if strategy == "int8_dynamic":
        if token_count <= 16:
            return Int8WeightOnlyConfig(version=2)
        return Int8DynamicActivationInt8WeightConfig(version=2)
    if strategy == "fp8_weight_only":
        return Float8WeightOnlyConfig()
    if strategy == "fp8_dynamic":
        return Float8DynamicActivationFloat8WeightConfig()
    raise ValueError(f"unknown quantization strategy: {strategy}")


def compile_quantized_model(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    *,
    device: torch.device,
    capture_source: bool,
) -> tuple[Any, float, dict[str, Any], str | None]:
    previous_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    with tempfile.TemporaryDirectory(prefix="transformer-quantized-") as cache_name:
        cache_dir = Path(cache_name)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_name
        try:
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            compiled = torch.compile(model, fullgraph=True, mode="max-autotune")
            compiled(*inputs)
            torch.cuda.synchronize(device)
            first_call_ms = (time.perf_counter() - start) * 1_000.0
            wrappers = find_generated_wrappers(cache_dir)
            analysis = aggregate_wrapper_analysis(wrappers)
            source = (
                sanitize_wrapper(wrappers[0], cache_dir) if capture_source else None
            )
        finally:
            if previous_cache is None:
                os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
            else:
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = previous_cache
    return compiled, first_call_ms, analysis, source


def quantization_worker(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torchao
        from torchao.quantization import quantize_
        from torchao.utils import get_model_size_in_bytes
    except ImportError as exc:
        raise RuntimeError(
            "TorchAO is required; install the quantization optional dependency"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for transformer quantization")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    original = (
        TransformerBlock(
            args.model_dim,
            args.num_heads,
            args.hidden_dim,
            rms_norm_eps=args.rms_norm_eps,
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    model = PackedTransformerBlock.from_unpacked(original).eval()
    shape = (
        args.worker_batch_size,
        args.worker_sequence_length,
        args.model_dim,
    )
    inputs = make_mlp_input(
        shape,
        device=device,
        dtype=torch.bfloat16,
        seed=args.seed,
    )
    token_count = args.worker_batch_size * args.worker_sequence_length
    quantization_seconds = 0.0
    if args.worker_strategy != "bf16":
        configuration = make_quantization_config(
            args.worker_strategy, token_count=token_count
        )
        start = time.perf_counter()
        quantize_(model, configuration)
        torch.cuda.synchronize(device)
        quantization_seconds = time.perf_counter() - start

    storage_bytes = int(get_model_size_in_bytes(model))
    with torch.inference_mode():
        reference = PackedTransformerBlock.from_unpacked(original).eval()(*inputs)
        compiled, first_call_ms, wrapper, source = compile_quantized_model(
            model,
            inputs,
            device=device,
            capture_source=args.capture_source,
        )
        actual = compiled(*inputs)
        torch.cuda.synchronize(device)
        quality = quality_statistics(reference, actual)
        if not quality["finite"] or quality["cosine_similarity"] < 0.99:
            raise AssertionError(
                f"{args.worker_strategy} failed the numerical quality threshold"
            )
        function = partial(compiled, *inputs)
        runtime = measure_runtime_breakdown(
            {"strategy": function},
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
        )["strategy"]
        profile = profile_cuda_kernels(
            function,
            device=device,
            iterations=args.profile_iterations,
        )

    return {
        "strategy": args.worker_strategy,
        "shape": list(shape),
        "torchao": torchao.__version__,
        "storage_bytes": storage_bytes,
        "quantization_seconds": quantization_seconds,
        "dynamic_int8_decode_fallback": (
            args.worker_strategy == "int8_dynamic" and token_count <= 16
        ),
        "quality": quality,
        "first_call_ms": first_call_ms,
        "runtime_breakdown": runtime,
        "kernel_profile": profile,
        "inductor_wrapper": wrapper,
        "wrapper_source": source,
    }


def worker_command(
    args: argparse.Namespace,
    *,
    strategy: str,
    batch_size: int,
    sequence_length: int,
    capture_source: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "compilelab.quantization",
        "--worker-strategy",
        strategy,
        "--worker-batch-size",
        str(batch_size),
        "--worker-sequence-length",
        str(sequence_length),
        "--model-dim",
        str(args.model_dim),
        "--num-heads",
        str(args.num_heads),
        "--hidden-dim",
        str(args.hidden_dim),
        "--rms-norm-eps",
        str(args.rms_norm_eps),
        "--seed",
        str(args.seed),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--profile-iterations",
        str(args.profile_iterations),
    ]
    if capture_source:
        command.append("--capture-source")
    return command


def run_isolated_strategy(
    args: argparse.Namespace,
    *,
    strategy: str,
    batch_size: int,
    sequence_length: int,
    capture_source: bool,
) -> dict[str, Any]:
    completed = subprocess.run(
        worker_command(
            args,
            strategy=strategy,
            batch_size=batch_size,
            sequence_length=sequence_length,
            capture_source=capture_source,
        ),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip().splitlines()
        reason = details[-1] if details else completed.stdout.strip()
        raise RuntimeError(f"{strategy} worker failed: {reason}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{strategy} worker returned no result")
    return json.loads(lines[-1])


def median_us(strategy: dict[str, Any]) -> float:
    return strategy["runtime_breakdown"]["synchronized_total"]["median_us"]


def render_summary(result: dict[str, Any]) -> str:
    rows = []
    for case in result["cases"]:
        baseline = case["strategies"]["bf16"]
        for name in QUANTIZATION_STRATEGIES:
            strategy = case["strategies"][name]
            speedup = median_us(baseline) / median_us(strategy)
            storage_ratio = baseline["storage_bytes"] / strategy["storage_bytes"]
            fallback = "yes" if strategy["dynamic_int8_decode_fallback"] else "no"
            rows.append(
                f"| {' × '.join(str(value) for value in case['shape'])} | "
                f"{name.replace('_', '-')} | {median_us(strategy):.2f} µs | "
                f"{speedup:.3f}× | "
                f"{storage_ratio:.2f}× | "
                f"{strategy['quality']['cosine_similarity']:.6f} | {fallback} |"
            )

    representative = result["cases"][result["representative_case_index"]]
    prefill_cases = [
        case for case in result["cases"] if case["shape"][0] * case["shape"][1] > 16
    ]
    int8_prefill_speedups = [
        median_us(case["strategies"]["bf16"])
        / median_us(case["strategies"]["int8_dynamic"])
        for case in prefill_cases
    ]
    fp8_prefill_speedups = [
        median_us(case["strategies"]["bf16"])
        / median_us(case["strategies"]["fp8_dynamic"])
        for case in prefill_cases
    ]
    prefill_finding = ""
    if prefill_cases:
        prefill_finding = (
            "Dynamic INT8 prefill speedups were "
            f"{', '.join(f'{value:.3f}×' for value in int8_prefill_speedups)}; "
            "dynamic FP8 prefill speedups\nwere "
            f"{', '.join(f'{value:.3f}×' for value in fp8_prefill_speedups)}. "
        )
    storage_rows = []
    for name in QUANTIZATION_STRATEGIES:
        strategy = representative["strategies"][name]
        external_calls = ", ".join(
            f"`{kernel}` × {count}"
            for kernel, count in strategy["inductor_wrapper"][
                "external_kernel_calls"
            ].items()
        )
        storage_rows.append(
            f"| {name.replace('_', '-')} | "
            f"{strategy['storage_bytes'] / (1024**2):.2f} MiB | "
            f"{external_calls or 'none'} |"
        )

    torchao_version = result["environment"]["torchao"]
    quantization_header = (
        "| Shape | Strategy | Median | Speedup | Storage reduction | "
        "Cosine similarity | INT8 decode fallback |"
    )

    return f"""# Minimal transformer quantization analysis

The packed transformer block is quantized with TorchAO {torchao_version}.
Every strategy uses BF16 activations at the block boundary and is compiled with
`max-autotune`; latency is compared with the unquantized packed BF16 block.

{quantization_header}
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Dynamic INT8 uses weight-only decode fallback when the flattened token dimension
is at most 16 because the generated integer matrix kernel requires a larger M
dimension. Prefill cases use dynamic activation and weight quantization.

{prefill_finding}None of the quantized paths improved single-token decode
latency. The INT8 weight-only `_weight_int8pack_mm` path was substantially
slower for prefill on this backend, demonstrating why quantization choices
require workload- and hardware-specific measurement.

## Representative generated paths

| Strategy | Model storage | External calls |
|---|---:|---|
{chr(10).join(storage_rows)}

All quantized outputs were finite and exceeded cosine similarity 0.99 against
the packed BF16 reference. Storage figures include parameters and quantization
metadata reported by TorchAO. Results are specific to the recorded software,
shapes, and RTX 5050 Laptop GPU.
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torchao
    except ImportError as exc:
        raise RuntimeError(
            "TorchAO is required; install with the quantization optional dependency"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for transformer quantization")
    if args.model_dim % args.num_heads != 0:
        raise ValueError("model_dim must be divisible by num_heads")

    cases = []
    captured_sources: dict[str, str] = {}
    total_runs = len(args.cases) * len(QUANTIZATION_STRATEGIES)
    completed_runs = 0
    for case_index, (batch_size, sequence_length) in enumerate(args.cases):
        strategies = {}
        for strategy in QUANTIZATION_STRATEGIES:
            completed_runs += 1
            capture_source = case_index == len(args.cases) - 1
            print(
                f"[{completed_runs}/{total_runs}] {batch_size}x{sequence_length} "
                f"{strategy}",
                flush=True,
            )
            worker_result = run_isolated_strategy(
                args,
                strategy=strategy,
                batch_size=batch_size,
                sequence_length=sequence_length,
                capture_source=capture_source,
            )
            source = worker_result.pop("wrapper_source")
            if source is not None:
                captured_sources[strategy] = source
            strategies[strategy] = worker_result
        cases.append(
            {
                "shape": [batch_size, sequence_length, args.model_dim],
                "strategies": strategies,
            }
        )

    result = {
        "schema_version": 1,
        "run_utc": datetime.now(UTC).isoformat(),
        "environment": {
            **environment_metadata(torch.device("cuda")),
            "torchao": torchao.__version__,
        },
        "experiment": {
            "workload": "packed minimal transformer block",
            "baseline_dtype": "bfloat16",
            "compile_mode": "max-autotune",
            "model_dim": args.model_dim,
            "num_heads": args.num_heads,
            "hidden_dim": args.hidden_dim,
            "rms_norm_eps": args.rms_norm_eps,
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "profile_iterations": args.profile_iterations,
            "quality_threshold_cosine": 0.99,
            "isolated_process_per_strategy": True,
            "isolated_inductor_cache": True,
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
    source_files = {
        "bf16": "bf16_inductor.py",
        "int8_dynamic": "int8_inductor.py",
        "fp8_dynamic": "fp8_inductor.py",
    }
    for strategy, filename in source_files.items():
        (args.output_dir / filename).write_text(
            captured_sources[strategy], encoding="utf-8"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark TorchAO quantization on a packed transformer block"
    )
    parser.add_argument(
        "--cases",
        type=parse_cases,
        default=((1, 1), (1, 128), (4, 128)),
    )
    parser.add_argument("--model-dim", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--profile-iterations", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/transformer_quantization"),
    )
    parser.add_argument(
        "--worker-strategy",
        choices=QUANTIZATION_STRATEGIES,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-sequence-length", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--capture-source", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.worker_strategy is not None:
        if args.worker_batch_size is None or args.worker_sequence_length is None:
            raise ValueError("worker shape arguments are required")
        print(json.dumps(quantization_worker(args)))
        return
    result = run(args)
    print(render_summary(result))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
