"""Controlled Qwen prefill/decode comparison with fresh request caches.

Extends the local 03_qwen.py study with length/dtype sweeps, first-call costs,
FX capture, recompilation accounting, and machine-readable trace summaries.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from compilelab.benchmark import environment_metadata, percentile
from compilelab.codegen import classify_inductor_wrapper
from compilelab.trace_analysis import summarize_trace

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROMPT = "Explain in simple terms why a KV cache makes language model inference faster."


def positive_integers(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use comma-separated positive integers"
        ) from exc
    if not values or any(number < 1 for number in values):
        raise argparse.ArgumentTypeError("values must be positive")
    return tuple(dict.fromkeys(values))


def prompt_ids(base: torch.Tensor, length: int) -> torch.Tensor:
    """Use native chat token IDs, repeating or truncating to the requested length."""
    repeats = (length + base.shape[1] - 1) // base.shape[1]
    return base.repeat(1, repeats)[:, :length].contiguous()


def prepare_work(
    phase: str,
    forward: Callable[..., Any],
    input_ids: torch.Tensor,
    tokens: list[torch.Tensor],
) -> Callable[[], list[torch.Tensor]]:
    if phase == "prefill":
        return lambda: [forward(input_ids, use_cache=True, logits_to_keep=1).logits]
    # This prefill is request setup, outside decode timing and profiler capture.
    cache = forward(input_ids, use_cache=True, logits_to_keep=1).past_key_values

    def work() -> list[torch.Tensor]:
        nonlocal cache
        logits = []
        for token in tokens:
            output = forward(
                token, past_key_values=cache, use_cache=True, logits_to_keep=1
            )
            cache = output.past_key_values
            logits.append(output.logits)
        return logits

    return work


def preserve_graph_outputs(forward: Callable[..., Any]) -> Callable[..., Any]:
    """Own state across CUDA Graph iterations; apply equally to both variants.

    DynamicCache entries and logits can otherwise reference replay-owned buffers.
    These copies are included in practical-mode timing and kernel counts.
    """

    def call(*values: Any, **kwargs: Any) -> Any:
        torch.compiler.cudagraph_mark_step_begin()
        output = forward(*values, **kwargs)
        output.logits = output.logits.clone()
        for layer in output.past_key_values.layers:
            layer.keys = layer.keys.clone()
            layer.values = layer.values.clone()
        return output

    return call


def timed_work(work: Callable[[], Any]) -> tuple[Any, dict[str, float]]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = work()
    enqueued = time.perf_counter()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return output, {
        "total_ms": (end - start) * 1000,
        "host_enqueue_ms": (enqueued - start) * 1000,
        "completion_wait_ms": (end - enqueued) * 1000,
    }


def compare_logits(
    reference: list[torch.Tensor],
    actual: list[torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if len(reference) != len(actual):
        raise AssertionError("different numbers of logit tensors")
    errors = [
        (left.float() - right.float()).abs() for left, right in zip(reference, actual)
    ]
    result = {
        "passed": all(
            bool(torch.isfinite(right).all())
            and torch.allclose(left, right, atol=atol, rtol=rtol)
            for left, right in zip(reference, actual)
        ),
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": max(float(error.max()) for error in errors),
        "mean_abs_error": statistics.mean(float(error.mean()) for error in errors),
        "greedy_ids_equal": all(
            torch.equal(left.argmax(-1), right.argmax(-1))
            for left, right in zip(reference, actual)
        ),
        "scope": "Every vocabulary logit for prefill's final position or each replayed decode step",
    }
    return result


def archive_generated_code(output_dir: Path) -> list[dict[str, Any]]:
    cache_root = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    generated = []
    for source_path in sorted(cache_root.rglob("*.py")):
        source = source_path.read_text(encoding="utf-8", errors="replace")
        if "async_compile.triton" not in source:
            continue
        destination = output_dir / "generated" / f"inductor_{len(generated) + 1}.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            source.replace(str(cache_root), "<inductor-cache>"), encoding="utf-8"
        )
        generated.append(
            {
                "artifact": str(destination.relative_to(output_dir)),
                **classify_inductor_wrapper(source),
            }
        )
    return generated


def run(args: argparse.Namespace) -> dict[str, Any]:
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen GPU profiling requires CUDA; no CPU fallback is used")
    if min(args.warmup, args.iterations) < 1:
        raise ValueError("warmup and iterations must be positive")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by the selected CUDA device")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_path = args.model_path or Path(
        snapshot_download(
            MODEL_ID, revision=args.revision, local_files_only=not args.allow_download
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, attn_implementation="sdpa", local_files_only=True
        )
        .cuda()
        .eval()
    )
    base_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    # Transformers releases can return a BatchEncoding instead of a tensor.
    if not isinstance(base_ids, torch.Tensor):
        base_ids = base_ids["input_ids"]
    base_ids = base_ids.cuda()
    compile_options = (
        {"triton.cudagraphs": False}
        if args.configuration == "no-cudagraphs"
        else dict(torch._inductor.list_mode_options("reduce-overhead"))
    )
    preserve_casts = args.precision_casts == "preserve" or (
        args.precision_casts == "auto" and dtype == torch.bfloat16
    )
    compile_options["emulate_precision_casts"] = preserve_casts
    graph_records: list[dict[str, Any]] = []

    def recording_backend(graph: torch.fx.GraphModule, examples: list[Any]) -> Callable:
        index = len(graph_records) + 1
        path = args.output_dir / f"dynamo_graph_{index}.py"
        path.write_text(graph.code, encoding="utf-8")
        graph_records.append(
            {
                "index": index,
                "artifact": path.name,
                "node_count": len(list(graph.graph.nodes)),
                "input_signature": [
                    {
                        "kind": "Tensor",
                        "shape": [str(n) for n in value.shape],
                        "dtype": str(value.dtype),
                    }
                    if isinstance(value, torch.Tensor)
                    else {"kind": type(value).__name__, "value": str(value)}
                    for value in examples
                ],
            }
        )
        return torch._inductor.compile(graph, examples, options=compile_options)

    compiled = torch.compile(
        model.forward, backend=recording_backend, fullgraph=True, dynamic=True
    )
    forwards = {"eager": model.forward, "compiled": compiled}
    if args.configuration == "reduce-overhead":
        forwards = {
            name: preserve_graph_outputs(forward) for name, forward in forwards.items()
        }
    atol = (
        args.atol
        if args.atol is not None
        else (1e-3 if dtype == torch.float32 else 2e-2)
    )
    rtol = (
        args.rtol
        if args.rtol is not None
        else (1e-3 if dtype == torch.float32 else 2e-2)
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_utc": datetime.now(UTC).isoformat(),
        "environment": {
            **environment_metadata(torch.device("cuda")),
            "transformers": transformers.__version__,
        },
        "experiment": {
            "model": MODEL_ID,
            "revision": model_path.name,
            "model_requested_revision": args.revision,
            "dtype": args.dtype,
            "tf32": False,
            "batch_size": 1,
            "prompt_lengths": args.prompt_lengths,
            "decode_steps": args.decode_steps,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
            "configuration": args.configuration,
            "compile_options": compile_options,
            "attention_backend_policy": args.attention_backend,
            "precision_cast_policy": "preserve" if preserve_casts else "default",
            "output_lifetime_policy": (
                "clone logits and DynamicCache K/V outside compile for both variants; included in timing"
                if args.configuration == "reduce-overhead"
                else "ordinary outputs; graph replay disabled"
            ),
            "fullgraph": True,
            "dynamic": True,
            "cache": "DynamicCache",
            "compile_backend": "recording backend delegating to torch._inductor.compile",
            "cache_policy": "fresh process; isolated Inductor and Triton directories from harness",
            "input_policy": "native chat prompt, repeated/truncated to exact lengths; IDs archived",
            "generation_policy": "replay greedy eager reference tokens; fresh cache for every trial",
            "timing_scope": "synchronized forward/phase time, excludes loading, tokenization, sampling, decode prefill setup, profiler",
            "first_call_scope": "first observed call for each phase/length; only the first compiled phase has a cold process",
        },
        "measurements": [],
        "graphs": graph_records,
        "study_status": "running",
        "correctness_failures": 0,
    }
    checkpoint = args.output_dir / "results.json"

    from torch.nn.attention import SDPBackend, sdpa_kernel

    attention_context = (
        sdpa_kernel(SDPBackend.MATH)
        if args.attention_backend == "math"
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), attention_context:
        for length in args.prompt_lengths:
            ids = prompt_ids(base_ids, length)
            output = model(ids, use_cache=True, logits_to_keep=1)
            tokens: list[torch.Tensor] = []
            for _ in range(max(args.decode_steps)):
                token = output.logits[:, -1].argmax(-1, keepdim=True)
                tokens.append(token)
                output = model(
                    token,
                    past_key_values=output.past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
            del output
            (args.output_dir / f"tokens_{length}.json").write_text(
                json.dumps(
                    {
                        "input_ids": ids.cpu().tolist(),
                        "continuation_ids": [token.cpu().tolist() for token in tokens],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            phases = [("prefill", 0)] + [
                ("decode", steps) for steps in args.decode_steps
            ]
            for phase, steps in phases:
                selected_tokens = tokens[:steps]
                label = f"s{length}_{phase}_{steps}"
                print(f"{args.dtype}/{args.configuration}: {label}", flush=True)
                first_calls = {}
                for name, forward in forwards.items():
                    before = len(graph_records)
                    output, first = timed_work(
                        prepare_work(phase, forward, ids, selected_tokens)
                    )
                    del output
                    first_calls[name] = {
                        **first,
                        "new_backend_compilations": len(graph_records) - before,
                    }
                    for _ in range(args.warmup):
                        prepare_work(phase, forward, ids, selected_tokens)()
                    torch.cuda.synchronize()
                expected = prepare_work(
                    phase, forwards["eager"], ids, selected_tokens
                )()
                actual = prepare_work(
                    phase, forwards["compiled"], ids, selected_tokens
                )()
                correctness = compare_logits(expected, actual, atol=atol, rtol=rtol)
                del expected, actual
                if not correctness["passed"]:
                    result["measurements"].append(
                        {
                            "prompt_tokens": length,
                            "phase": phase,
                            "decode_steps": steps,
                            "correctness": correctness,
                            "status": "correctness_failed",
                            "first_calls": first_calls,
                            "performance_accepted": False,
                            "reason": "steady-state timing and profiling skipped after numerical rejection",
                        }
                    )
                    result["correctness_failures"] += 1
                    result["study_status"] = "rejected"
                    checkpoint.write_text(
                        json.dumps(result, indent=2) + "\n", encoding="utf-8"
                    )
                    if args.continue_on_mismatch:
                        print(f"  Rejected numerically: {correctness}", flush=True)
                        continue
                    raise AssertionError(
                        f"logit tolerance failed for {label}: {correctness}"
                    )
                samples: dict[str, list[dict[str, float]]] = {
                    name: [] for name in forwards
                }
                compiled_before = len(graph_records)
                memory = {}
                for repeat in range(args.iterations):
                    order = (
                        list(forwards) if repeat % 2 == 0 else list(reversed(forwards))
                    )
                    for name in order:
                        work = prepare_work(phase, forwards[name], ids, selected_tokens)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        allocated_before = torch.cuda.memory_allocated()
                        output, sample = timed_work(work)
                        samples[name].append(sample)
                        memory[name] = {
                            "allocated_before_bytes": allocated_before,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                            "scope": "last timed trial; decode prefill cache exists at baseline",
                        }
                        del output, work
                steady_compilations = len(graph_records) - compiled_before
                if steady_compilations:
                    raise RuntimeError(
                        f"{steady_compilations} backend compilations during steady timing"
                    )
                for name, forward in forwards.items():
                    values = [sample["total_ms"] for sample in samples[name]]
                    row = {
                        "status": "passed",
                        "mode": name,
                        "prompt_tokens": length,
                        "phase": phase,
                        "decode_steps": steps,
                        "correctness": correctness,
                        "first_call": first_calls[name],
                        "samples": samples[name],
                        "median_ms": statistics.median(values),
                        "p90_ms": percentile(values, 0.9),
                        "min_ms": min(values),
                        "max_ms": max(values),
                        "median_ms_per_forward": statistics.median(values)
                        / (steps or 1),
                        "steady_backend_compilations": steady_compilations,
                        "backend_compilations_so_far": len(graph_records),
                        "memory": memory[name],
                    }
                    work = prepare_work(phase, forward, ids, selected_tokens)
                    torch.cuda.synchronize()
                    count_before = len(graph_records)
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=True,
                        profile_memory=True,
                    ) as captured:
                        with record_function(f"{name}/{label}"):
                            output = work()
                        torch.cuda.synchronize()
                    del output, work
                    trace = args.output_dir / f"{name}_{label}.json"
                    captured.export_chrome_trace(str(trace))
                    summary = summarize_trace(trace)
                    if not summary["launch_count"]:
                        raise RuntimeError(
                            "No GPU kernels captured; check CUPTI access"
                        )
                    if (
                        summary["compilation_events"]
                        or len(graph_records) != count_before
                    ):
                        raise RuntimeError(
                            f"Compilation inside profiler capture: {label}"
                        )
                    if (
                        args.configuration == "no-cudagraphs"
                        and summary["cuda_graph_replay_calls"]
                    ):
                        raise RuntimeError(
                            "Unexpected CUDA Graph replay in no-cudagraphs configuration"
                        )
                    row.update(
                        {
                            "trace": trace.name,
                            "trace_sha256": hashlib.sha256(
                                trace.read_bytes()
                            ).hexdigest(),
                            "trace_summary": summary,
                            "gpu_kernels_per_forward": summary["launch_count"]
                            / (steps or 1),
                        }
                    )
                    result["measurements"].append(row)
                    checkpoint.write_text(
                        json.dumps(result, indent=2) + "\n", encoding="utf-8"
                    )
                    print(
                        f"  {name}: {row['median_ms']:.3f} ms, {summary['launch_count']} kernels",
                        flush=True,
                    )

    # Preserve the actual generated kernels before the harness destroys its cache.
    result["generated_wrappers"] = archive_generated_code(args.output_dir)
    result["backend_compilations"] = len(graph_records)
    result["study_status"] = "rejected" if result["correctness_failures"] else "passed"
    checkpoint.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen fixed-token prefill/decode GPU benchmark"
    )
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument(
        "--configuration",
        choices=("no-cudagraphs", "reduce-overhead"),
        default="no-cudagraphs",
    )
    parser.add_argument(
        "--prompt-lengths", type=positive_integers, default=(32, 44, 128, 512)
    )
    parser.add_argument("--decode-steps", type=positive_integers, default=(8, 32))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument(
        "--precision-casts",
        choices=("auto", "default", "preserve"),
        default="auto",
        help="auto preserves intermediate casts for BF16; failures from default policy are retained",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("default", "math"),
        default="default",
        help="pin math SDPA equally for both variants to study low-precision agreement",
    )
    parser.add_argument(
        "--continue-on-mismatch",
        action="store_true",
        help="complete accuracy coverage, skipping performance for rejected cases",
    )
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="torchcompile-qwen-") as cache:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache
            try:
                run(args)
            finally:
                archive_generated_code(args.output_dir)
                os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
    else:
        try:
            run(args)
        finally:
            archive_generated_code(args.output_dir)


if __name__ == "__main__":
    main()
