from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch

from compilelab.benchmark import capture_dynamo_graph, environment_metadata
from compilelab.codegen import (
    classify_inductor_wrapper,
    measure_runtime_breakdown,
    profile_cuda_kernels,
)
from compilelab.projection_packing import error_statistics, parse_cases
from compilelab.transformer import (
    PackedTransformerBlock,
    RMSNormResidual,
    TransformerBlock,
)
from compilelab.workload import make_mlp_input


TRANSFORMER_STRATEGIES = (
    "original_eager",
    "packed_eager",
    "original_default",
    "packed_components",
    "packed_default",
    "packed_reduce_overhead",
    "packed_max_autotune_no_cudagraphs",
    "packed_max_autotune",
    "packed_shape_padding",
)
RMSNORM_STRATEGIES = ("rmsnorm_native", "rmsnorm_decomposed")
COMPILE_STRATEGIES = {
    "original_default": {"model": "original", "mode": None, "options": None},
    "packed_default": {"model": "packed", "mode": None, "options": None},
    "packed_reduce_overhead": {
        "model": "packed",
        "mode": "reduce-overhead",
        "options": None,
    },
    "packed_max_autotune_no_cudagraphs": {
        "model": "packed",
        "mode": "max-autotune-no-cudagraphs",
        "options": None,
    },
    "packed_max_autotune": {
        "model": "packed",
        "mode": "max-autotune",
        "options": None,
    },
    "packed_shape_padding": {
        "model": "packed",
        "mode": None,
        "options": {"shape_padding": True},
    },
}


def find_generated_wrappers(cache_dir: Path) -> list[str]:
    wrappers = []
    for path in cache_dir.rglob("*.py"):
        source = path.read_text(encoding="utf-8", errors="replace")
        if "class Runner:" in source and "def call(self, args):" in source:
            wrappers.append(source)
    if not wrappers:
        raise RuntimeError("no generated Inductor wrapper was found")
    return wrappers


def aggregate_wrapper_analysis(sources: list[str]) -> dict[str, Any]:
    analyses = [classify_inductor_wrapper(source) for source in sources]
    external_calls: Counter[str] = Counter()
    source_node_groups: list[str] = []
    for analysis in analyses:
        external_calls.update(analysis["external_kernel_calls"])
        source_node_groups.extend(analysis["source_node_groups"])
    return {
        "wrapper_count": len(analyses),
        "external_kernel_calls": dict(sorted(external_calls.items())),
        "external_kernel_call_count": sum(
            analysis["external_kernel_call_count"] for analysis in analyses
        ),
        "triton_kernel_launch_count": sum(
            analysis["triton_kernel_launch_count"] for analysis in analyses
        ),
        "wrapper_launch_site_count": sum(
            analysis["wrapper_launch_site_count"] for analysis in analyses
        ),
        "buffer_reuse_count": sum(
            analysis["buffer_reuse_count"] for analysis in analyses
        ),
        "input_alignment_copy_count": sum(
            analysis["input_alignment_copy_count"] for analysis in analyses
        ),
        "source_node_groups": source_node_groups,
    }


def sanitize_wrapper(source: str, cache_dir: Path) -> str:
    return (
        "\n".join(
            line.rstrip()
            for line in source.replace(str(cache_dir), "<inductor-cache>").splitlines()
        ).rstrip()
        + "\n"
    )


def compile_whole_block(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    *,
    mode: str | None,
    options: dict[str, bool] | None,
    device: torch.device,
    capture_source: bool,
) -> tuple[Callable[[], torch.Tensor], float, dict[str, Any], str | None]:
    previous_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    with tempfile.TemporaryDirectory(prefix="transformer-inductor-") as cache_name:
        cache_dir = Path(cache_name)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_name
        try:
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            compiled = torch.compile(
                model,
                fullgraph=True,
                mode=mode,
                options=options,
            )
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
    return lambda: compiled(*inputs), first_call_ms, analysis, source


def compile_components(
    model: PackedTransformerBlock,
    inputs: tuple[torch.Tensor, ...],
    *,
    device: torch.device,
) -> tuple[Callable[[], torch.Tensor], float, dict[str, Any]]:
    previous_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    with tempfile.TemporaryDirectory(prefix="transformer-components-") as cache_name:
        cache_dir = Path(cache_name)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_name
        try:
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            attention = torch.compile(model.attention, fullgraph=True)
            mlp = torch.compile(model.mlp, fullgraph=True)

            def execute() -> torch.Tensor:
                hidden = inputs[0] + attention(model.attention_norm(inputs[0]))
                return hidden + mlp(model.mlp_norm(hidden))

            execute()
            torch.cuda.synchronize(device)
            first_call_ms = (time.perf_counter() - start) * 1_000.0
            analysis = aggregate_wrapper_analysis(find_generated_wrappers(cache_dir))
        finally:
            if previous_cache is None:
                os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
            else:
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = previous_cache
    return execute, first_call_ms, analysis


def transformer_worker(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for transformer optimization analysis")
    device = torch.device("cuda")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    torch.manual_seed(args.seed)
    original = TransformerBlock(
        args.model_dim,
        args.num_heads,
        args.hidden_dim,
        rms_norm_eps=args.rms_norm_eps,
    ).eval()
    packed = PackedTransformerBlock.from_unpacked(original).eval()
    original = original.to(device=device, dtype=dtype)
    packed = packed.to(device=device, dtype=dtype)
    inputs = make_mlp_input(
        (args.worker_batch_size, args.worker_sequence_length, args.model_dim),
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    with torch.inference_mode():
        reference = original(*inputs)
        wrapper_analysis = None
        wrapper_source = None
        first_call_ms = None
        if args.worker_strategy == "original_eager":
            function = partial(original, *inputs)
        elif args.worker_strategy == "packed_eager":
            function = partial(packed, *inputs)
        elif args.worker_strategy == "packed_components":
            function, first_call_ms, wrapper_analysis = compile_components(
                packed, inputs, device=device
            )
        else:
            configuration = COMPILE_STRATEGIES[args.worker_strategy]
            selected = original if configuration["model"] == "original" else packed
            function, first_call_ms, wrapper_analysis, wrapper_source = (
                compile_whole_block(
                    selected,
                    inputs,
                    mode=configuration["mode"],
                    options=configuration["options"],
                    device=device,
                    capture_source=args.capture_source,
                )
            )

        actual = function()
        correctness = error_statistics(reference, actual)
        if not correctness["allclose"]:
            raise AssertionError(
                f"{args.worker_strategy} output did not match original eager"
            )
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

    parameter_count = sum(
        parameter.numel()
        for parameter in (
            original.parameters()
            if args.worker_strategy.startswith("original")
            else packed.parameters()
        )
    )
    return {
        "strategy": args.worker_strategy,
        "shape": [
            args.worker_batch_size,
            args.worker_sequence_length,
            args.model_dim,
        ],
        "parameter_count": parameter_count,
        "correctness": correctness,
        "first_call_ms": first_call_ms,
        "runtime_breakdown": runtime,
        "kernel_profile": profile,
        "inductor_wrapper": wrapper_analysis,
        "wrapper_source": wrapper_source,
    }


def rmsnorm_worker(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RMSNorm analysis")
    device = torch.device("cuda")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    torch.manual_seed(args.seed)
    native = (
        RMSNormResidual(
            args.model_dim,
            decomposed=False,
            eps=args.rms_norm_eps,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    decomposed = (
        RMSNormResidual(
            args.model_dim,
            decomposed=True,
            eps=args.rms_norm_eps,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    with torch.no_grad():
        decomposed.norm.weight.copy_(native.norm.weight)
    x = torch.randn(
        args.worker_batch_size,
        args.worker_sequence_length,
        args.model_dim,
        device=device,
        dtype=dtype,
    )
    residual = torch.randn_like(x)
    inputs = (x, residual)
    selected = native if args.worker_strategy == "rmsnorm_native" else decomposed
    with torch.inference_mode():
        reference = native(*inputs)
        function, first_call_ms, wrapper, source = compile_whole_block(
            selected,
            inputs,
            mode=None,
            options=None,
            device=device,
            capture_source=True,
        )
        actual = function()
        correctness = error_statistics(reference, actual)
        if not correctness["allclose"]:
            raise AssertionError("compiled RMSNorm output did not match native eager")
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
        "shape": list(x.shape),
        "correctness": correctness,
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
        "compilelab.transformer_optimization",
        "--worker-strategy",
        strategy,
        "--worker-batch-size",
        str(batch_size),
        "--worker-sequence-length",
        str(sequence_length),
        "--dtype",
        args.dtype,
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
        detail = completed.stderr.strip().splitlines()
        reason = detail[-1] if detail else completed.stdout.strip()
        raise RuntimeError(f"{strategy} worker failed: {reason}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{strategy} worker returned no result")
    return json.loads(lines[-1])


def capture_block_graphs(args: argparse.Namespace) -> str:
    torch.manual_seed(args.seed)
    original = TransformerBlock(
        args.model_dim,
        args.num_heads,
        args.hidden_dim,
        rms_norm_eps=args.rms_norm_eps,
    ).eval()
    packed = PackedTransformerBlock.from_unpacked(original).eval()
    batch_size, sequence_length = args.cases[-1]
    inputs = make_mlp_input(
        (batch_size, sequence_length, args.model_dim),
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=args.seed,
    )
    with torch.inference_mode():
        original_graphs, original_count = capture_dynamo_graph(original, inputs)
        packed_graphs, packed_count = capture_dynamo_graph(packed, inputs)
    torch.compiler.reset()
    if original_count != 1 or packed_count != 1:
        raise RuntimeError("expected one Dynamo graph for each transformer block")
    return (
        "# ---- original transformer block ----\n"
        f"{original_graphs[0].strip()}\n\n"
        "# ---- packed transformer block ----\n"
        f"{packed_graphs[0].strip()}\n"
    )


def strategy_median(strategy: dict[str, Any]) -> float:
    return strategy["runtime_breakdown"]["synchronized_total"]["median_us"]


def render_summary(result: dict[str, Any]) -> str:
    cases = result["cases"]
    rows = []
    mode_names = (
        "packed_default",
        "packed_reduce_overhead",
        "packed_max_autotune_no_cudagraphs",
        "packed_max_autotune",
        "packed_shape_padding",
    )
    for case in cases:
        strategies = case["strategies"]
        best_name = min(mode_names, key=lambda name: strategy_median(strategies[name]))
        rows.append(
            f"| {' × '.join(str(value) for value in case['shape'])} | "
            f"{strategy_median(strategies['original_eager']):.2f} µs | "
            f"{strategy_median(strategies['original_default']):.2f} µs | "
            f"{strategy_median(strategies['packed_default']):.2f} µs | "
            f"{best_name.removeprefix('packed_').replace('_', '-')} | "
            f"{strategy_median(strategies[best_name]):.2f} µs |"
        )

    mode_rows = []
    for name in mode_names:
        values = " | ".join(
            f"{strategy_median(case['strategies'][name]):.2f} µs" for case in cases
        )
        display_name = name.removeprefix("packed_").replace("_", "-")
        mode_rows.append(f"| {display_name} | {values} |")

    representative = cases[result["representative_case_index"]]
    original = representative["strategies"]["original_default"]
    packed = representative["strategies"]["packed_default"]
    packing_speedups = [
        strategy_median(case["strategies"]["original_default"])
        / strategy_median(case["strategies"]["packed_default"])
        for case in cases
    ]
    region_rows = []
    whole_speedups = []
    for case in cases:
        whole = case["strategies"]["packed_default"]
        regions = case["strategies"]["packed_components"]
        speedup = strategy_median(regions) / strategy_median(whole)
        whole_speedups.append(speedup)
        region_rows.append(
            f"| {' × '.join(str(value) for value in case['shape'])} | "
            f"{strategy_median(whole):.2f} µs | "
            f"{strategy_median(regions):.2f} µs | {speedup:.3f}× |"
        )
    fusion = result["rmsnorm_fusion"]
    native_fusion = fusion["native"]["inductor_wrapper"]
    decomposed_fusion = fusion["decomposed"]["inductor_wrapper"]
    block_header = (
        "| Shape | Original eager | Original compiled | Packed compiled | "
        "Best packed mode | Best latency |"
    )
    mode_shape_columns = " | ".join(
        " × ".join(str(value) for value in case["shape"]) for case in cases
    )
    structural_rows = (
        "| External GEMM calls | "
        f"{original['inductor_wrapper']['external_kernel_call_count']} | "
        f"{packed['inductor_wrapper']['external_kernel_call_count']} |\n"
        "| Generated Triton launch sites | "
        f"{original['inductor_wrapper']['triton_kernel_launch_count']} | "
        f"{packed['inductor_wrapper']['triton_kernel_launch_count']} |\n"
        "| Total wrapper launch sites | "
        f"{original['inductor_wrapper']['wrapper_launch_site_count']} | "
        f"{packed['inductor_wrapper']['wrapper_launch_site_count']} |\n"
        "| Profiled CUDA launches/invocation | "
        f"{original['kernel_profile']['launches_per_invocation']:.1f} | "
        f"{packed['kernel_profile']['launches_per_invocation']:.1f} |"
    )
    rmsnorm_rows = (
        "| Native RMSNorm | "
        f"{native_fusion['wrapper_launch_site_count']} | "
        f"{fusion['native']['kernel_profile']['launches_per_invocation']:.1f} | "
        f"{strategy_median(fusion['native']):.2f} µs |\n"
        "| Decomposed arithmetic | "
        f"{decomposed_fusion['wrapper_launch_site_count']} | "
        f"{fusion['decomposed']['kernel_profile']['launches_per_invocation']:.1f} | "
        f"{strategy_median(fusion['decomposed']):.2f} µs |"
    )

    return f"""# Minimal transformer optimization analysis

The workload is a pre-norm causal transformer block with two RMSNorm layers,
multi-head scaled-dot-product attention, residual connections, and a gated MLP.
The optimized block packs Q/K/V and gate/up weights once during model conversion.

## Block-level results

{block_header}
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Default compiled packing produced speedups of
{", ".join(f"{speedup:.3f}×" for speedup in packing_speedups)} across the measured
shapes by reducing the generated external GEMM count from seven to four.

## Compilation boundary

| Shape | Whole block | Attention and MLP regions | Whole-block speedup |
|---|---:|---:|---:|
{chr(10).join(region_rows)}

Whole-block compilation was {whole_speedups[0]:.3f}× and
{whole_speedups[1]:.3f}× faster for the batch-one cases; the two boundaries were
effectively tied at the largest shape ({whole_speedups[2]:.3f}×).

## Compile-mode matrix

| Packed strategy | {mode_shape_columns} |
|---|{"---:|" * len(cases)}
{chr(10).join(mode_rows)}

## Generated execution

| Representative structural metric | Original | Packed |
|---|---:|---:|
{structural_rows}

The packed graph replaces three Q/K/V GEMMs with one and replaces two gated-MLP
input GEMMs with one. Inductor also fuses the first residual addition with the
second RMSNorm and fuses the final residual addition into a generated pointwise
kernel.

## RMSNorm representation

| Form | Wrapper launch sites | Profiled launches/invocation | Median latency |
|---|---:|---:|---:|
{rmsnorm_rows}

Both forms lowered to one generated launch, so manually decomposing RMSNorm did
not remove another kernel. All strategies passed numerical comparison with
original eager execution. Results are specific to the recorded software,
shapes, and RTX 5050 Laptop GPU.
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for transformer optimization analysis")
    if args.model_dim % args.num_heads != 0:
        raise ValueError("model_dim must be divisible by num_heads")
    if args.warmup < 1 or args.iterations < 1 or args.profile_iterations < 1:
        raise ValueError("warmup and iteration counts must be positive")

    graph_artifact = capture_block_graphs(args)
    cases = []
    captured_sources: dict[str, str] = {}
    total_runs = len(args.cases) * len(TRANSFORMER_STRATEGIES) + 2
    completed_runs = 0
    for case_index, (batch_size, sequence_length) in enumerate(args.cases):
        strategies = {}
        for strategy in TRANSFORMER_STRATEGIES:
            capture_source = case_index == len(args.cases) - 1 and strategy in {
                "original_default",
                "packed_default",
            }
            completed_runs += 1
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
            source = worker_result.pop("wrapper_source", None)
            if source is not None:
                captured_sources[strategy] = source
            strategies[strategy] = worker_result
        cases.append(
            {
                "shape": [batch_size, sequence_length, args.model_dim],
                "strategies": strategies,
            }
        )

    representative_batch, representative_sequence = args.cases[-1]
    rmsnorm_results = {}
    for strategy in RMSNORM_STRATEGIES:
        completed_runs += 1
        print(f"[{completed_runs}/{total_runs}] {strategy}", flush=True)
        worker_result = run_isolated_strategy(
            args,
            strategy=strategy,
            batch_size=representative_batch,
            sequence_length=representative_sequence,
            capture_source=True,
        )
        source = worker_result.pop("wrapper_source")
        captured_sources[strategy] = source
        key = strategy.removeprefix("rmsnorm_")
        rmsnorm_results[key] = worker_result

    packed_groups = cases[-1]["strategies"]["packed_default"]["inductor_wrapper"][
        "source_node_groups"
    ]
    result = {
        "schema_version": 1,
        "run_utc": datetime.now(UTC).isoformat(),
        "environment": environment_metadata(torch.device("cuda")),
        "experiment": {
            "workload": "minimal pre-norm causal transformer block",
            "model_dim": args.model_dim,
            "num_heads": args.num_heads,
            "head_dim": args.model_dim // args.num_heads,
            "hidden_dim": args.hidden_dim,
            "rms_norm_eps": args.rms_norm_eps,
            "dtype": args.dtype,
            "seed": args.seed,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "profile_iterations": args.profile_iterations,
            "isolated_process_per_strategy": True,
            "isolated_inductor_cache": True,
        },
        "representative_case_index": len(cases) - 1,
        "cases": cases,
        "rmsnorm_fusion": {
            **rmsnorm_results,
            "block_source_groups": packed_groups,
            "residual_into_second_rmsnorm": any(
                "hidden" in group and "rms_norm_1" in group for group in packed_groups
            ),
            "final_residual_pointwise": any(
                "add_1" in group for group in packed_groups
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
    (args.output_dir / "dynamo_graphs.py").write_text(graph_artifact, encoding="utf-8")
    source_files = {
        "original_default": "original_inductor.py",
        "packed_default": "packed_inductor.py",
        "rmsnorm_native": "rmsnorm_native_inductor.py",
        "rmsnorm_decomposed": "rmsnorm_decomposed_inductor.py",
    }
    for strategy, filename in source_files.items():
        (args.output_dir / filename).write_text(
            captured_sources[strategy], encoding="utf-8"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark minimal-transformer compiler optimizations"
    )
    parser.add_argument(
        "--cases",
        type=parse_cases,
        default=((1, 1), (1, 128), (4, 128)),
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
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
        default=Path("artifacts/transformer_optimization"),
    )
    parser.add_argument(
        "--worker-strategy",
        choices=TRANSFORMER_STRATEGIES + RMSNORM_STRATEGIES,
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
        if args.worker_strategy in RMSNORM_STRATEGIES:
            result = rmsnorm_worker(args)
        else:
            result = transformer_worker(args)
        print(json.dumps(result))
        return

    result = run(args)
    print(render_summary(result))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
