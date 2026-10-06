"""Export measured compiler and inference results with portable replay commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_text_artifact(source: Path, destination: Path) -> None:
    """Normalize generated text without modifying measured result files."""
    text = "\n".join(
        line.rstrip() for line in source.read_text(encoding="utf-8").splitlines()
    ).rstrip()
    destination.write_text(text + "\n" if text else "", encoding="utf-8")


def replay_argv(entry: dict[str, Any]) -> list[str]:
    """Preserve measured flags while normalizing machine-specific output paths."""
    argv = list(entry["argv"])
    argv[0] = "python"
    output_flag = argv.index("--output-dir")
    argv[output_flag + 1] = str(Path("artifacts/benchmark_runs") / entry["name"])
    return argv


def assemble(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    from compilelab.benchmark import break_even_calls
    from compilelab.dynamic_shapes import parse_recompile_events

    if run_dir.resolve() == output_dir.resolve():
        raise ValueError("Run and export directories must be different")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, Any] = {
        "schema_version": 2,
        "manifest_sha256": digest(manifest_path),
        "git_revision": manifest.get("git_revision"),
        "command_scope": "Portable replay commands preserve measured workload flags; the Python executable and output directory are normalized. Source hashes identify the measured worker files.",
        "runs": [],
    }
    lines = [
        "# Compiler and Qwen inference evidence",
        "",
        "Measurements apply to the recorded hardware, software, and inputs. Raw result files are retained byte-for-byte; replay commands and measured source hashes are in [evidence.json](evidence.json).",
        "",
        "## Fresh-process compiler measurements",
        "",
        "| Case | Eager median ms | Compiled median ms | Ratio | First call s | Break-even calls | Evidence |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    qwen_rows = []
    latest = {entry["name"]: entry for entry in manifest["runs"]}
    for name, entry in latest.items():
        destination = output_dir / name
        destination.mkdir(exist_ok=True)
        source = run_dir / name / "results.json"
        record = {
            key: entry[key]
            for key in (
                "name",
                "source_file",
                "source_sha256",
                "returncode",
                "start_utc",
                "end_utc",
                "gpu_before",
                "gpu_after",
            )
            if key in entry
        }
        record["argv"] = replay_argv(entry)
        record["command"] = shlex.join(record["argv"])
        record["status"] = "passed" if entry["returncode"] == 0 else "failed"
        if not source.exists():
            evidence["runs"].append(record)
            continue
        record["result_sha256"] = digest(source)
        if (
            entry.get("result_sha256", record["result_sha256"])
            != record["result_sha256"]
        ):
            raise AssertionError(f"Result hash mismatch: {name}")
        shutil.copyfile(source, destination / "results.json")
        raw = json.loads(source.read_text(encoding="utf-8"))
        record["result"] = f"{name}/results.json"
        record["environment"] = raw.get("environment", {})
        record["study_status"] = raw.get("study_status", record["status"])
        if entry["returncode"] == 0 and record["study_status"] == "rejected":
            record["status"] = "rejected"
        if entry["returncode"] != 0:
            record["diagnostics"] = (
                "Worker stdout/stderr are retained in the run directory"
            )
            evidence["runs"].append(record)
            continue
        if name.startswith(("pointwise_", "mlp_")) and "timing" in raw:
            timing = raw["timing"]
            eager = timing["eager"]["median_ms"]
            compiled = timing["compiled"]["median_ms"]
            first = timing["compiled"]["first_call_ms"]
            amortization = break_even_calls(first, eager, compiled)
            if amortization != timing["break_even_calls"]:
                raise AssertionError(f"Break-even mismatch: {name}")
            record["verified_break_even_calls"] = amortization
            lines.append(
                f"| {name} | {eager:.4f} | {compiled:.4f} | {eager / compiled:.3f}× | {first / 1000:.3f} | {amortization if amortization is not None else 'none'} | [JSON]({name}/results.json) |"
            )
        if name.startswith("qwen_"):
            record["backend_compilations"] = raw.get("backend_compilations")
            stderr = (run_dir / name / "stderr.txt").read_text(encoding="utf-8")
            recompiles = parse_recompile_events(stderr)
            record["recompile_events"] = recompiles
            (destination / "recompiles.json").write_text(
                json.dumps(recompiles, indent=2) + "\n", encoding="utf-8"
            )
            for row in raw["measurements"]:
                if row.get("mode") != "eager" or row.get("status") != "passed":
                    continue
                matches = [
                    other
                    for other in raw["measurements"]
                    if other.get("mode") == "compiled"
                    and other["prompt_tokens"] == row["prompt_tokens"]
                    and other["phase"] == row["phase"]
                    and other["decode_steps"] == row["decode_steps"]
                ]
                if len(matches) != 1:
                    raise AssertionError(f"Ambiguous Qwen comparison: {name}")
                compiled = matches[0]
                if (
                    compiled.get("status") != "passed"
                    or not compiled["correctness"]["passed"]
                ):
                    raise AssertionError(f"Cannot export rejected performance: {name}")
                qwen_rows.append((name, row, compiled))
            if name == "qwen_float32_no-cudagraphs_1":
                for pattern in ("dynamo_graph_*.py", "tokens_*.json"):
                    for artifact in (run_dir / name).glob(pattern):
                        if artifact.suffix == ".py":
                            copy_text_artifact(artifact, destination / artifact.name)
                        else:
                            shutil.copyfile(artifact, destination / artifact.name)
                for artifact in (run_dir / name / "generated").glob("*.py"):
                    (destination / "generated").mkdir(exist_ok=True)
                    copy_text_artifact(
                        artifact, destination / "generated" / artifact.name
                    )
        for artifact in (
            "summary.md",
            "dynamo_fx_graph.py",
            "python_if_graphs.py",
            "torch_cond_graph.py",
            "python_if_explain.txt",
            "torch_cond_explain.txt",
            "inductor_output_code.py",
            "recompiles.txt",
            "automatic_graphs.py",
        ):
            artifact_path = run_dir / name / artifact
            if artifact_path.exists():
                copy_text_artifact(artifact_path, destination / artifact)
        evidence["runs"].append(record)
    lines += [
        "",
        "Break-even uses `ceil(max(0, first_call_ms - compiled_median_ms) / (eager_median_ms - compiled_median_ms))`. It is undefined when compiled execution is slower. First-call time includes capture, lowering, compilation, and initial execution; it is not pure compiler CPU time. Across-trial variation is empirical variation, not a confidence interval.",
        "",
        "## Qwen prefill and decode",
        "",
        "Times are whole measured phases, outside the profiler. Decode excludes its prefill setup and token selection; per-forward decode values are averages, not token-latency percentiles.",
        "",
        "| Run | Prompt | Phase/steps | Eager ms | Compiled ms | Ratio | GPU kernels eager → compiled | Max logit error | Greedy IDs equal | Graph replay calls |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for name, eager, compiled in qwen_rows:
        lines.append(
            f"| [{name}]({name}/results.json) | {eager['prompt_tokens']} | {eager['phase']}/{eager['decode_steps']} | {eager['median_ms']:.3f} | {compiled['median_ms']:.3f} | {eager['median_ms'] / compiled['median_ms']:.3f}× | {eager['trace_summary']['launch_count']} → {compiled['trace_summary']['launch_count']} | {compiled['correctness']['max_abs_error']:.3g} | {compiled['correctness']['greedy_ids_equal']} | {sum(compiled['trace_summary']['cuda_graph_replay_calls'].values())} |"
        )
    lines += [
        "",
        "GPU kernel durations include profiling overhead. Kernel interval coverage is a trace-derived activity fraction, not measured GPU occupancy/utilization or bandwidth. CUDA Graph settings are requested policy; actual replay calls are reported separately. A backend graph count records compilations across tested shapes, rather than claiming one graph supports every input.",
        "",
    ]
    lines += [
        "## Numerical rejections",
        "",
        "These cases failed the fixed logit tolerance. Steady-state timing and profiling were skipped; they are not accepted performance improvements.",
        "",
        "| Run | Prompt | Phase/steps | Maximum error | Mean error | Greedy IDs equal | Tolerance |",
        "|---|---:|---|---:|---:|---|---|",
    ]
    for record in evidence["runs"]:
        path = output_dir / record["name"] / "results.json"
        if not record["name"].startswith("qwen_") or not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        for row in raw["measurements"]:
            if row.get("status") != "correctness_failed":
                continue
            quality = row["correctness"]
            lines.append(
                f"| [{record['name']}]({record['name']}/results.json) | {row['prompt_tokens']} | {row['phase']}/{row['decode_steps']} | {quality['max_abs_error']:.4g} | {quality['mean_abs_error']:.4g} | {quality['greedy_ids_equal']} | atol={quality['atol']}, rtol={quality['rtol']} |"
            )
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export compiler and Qwen benchmark results"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("artifacts/benchmark_runs")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/benchmark_results")
    )
    args = parser.parse_args()
    assemble(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
