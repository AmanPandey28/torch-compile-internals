"""Run serial GPU experiments in fresh processes with isolated compiler caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def gpu_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return {
            "fields": "name,driver_version,pstate,temperature_c,power_w,power_limit_w,sm_clock_mhz,memory_clock_mhz",
            "value": result.stdout.strip(),
            "clocks_locked": False,
            "assumption": "laptop GPU; display active; no power/clock settings modified",
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": str(exc), "clocks_locked": False}


def jobs(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    result: list[tuple[str, list[str]]] = []
    if args.suite in {"micro", "all"}:
        mlp_dtype = "float16" if args.device == "cuda" else "float32"
        common = [
            "--device",
            args.device,
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
        ]
        for trial in range(1, args.trials + 1):
            result.append(
                (
                    f"pointwise_primary_{trial}",
                    ["-m", "compilelab", *common, "--shape", "1024,4096"],
                )
            )
            result.append(
                (
                    f"mlp_primary_{trial}",
                    [
                        "-m",
                        "compilelab",
                        *common,
                        "--workload",
                        "gated-mlp",
                        "--dtype",
                        mlp_dtype,
                    ],
                )
            )
        for shape in ("64,257", "128,1024"):
            result.append(
                (
                    f"pointwise_{shape.replace(',', 'x')}",
                    ["-m", "compilelab", *common, "--shape", shape],
                )
            )
        for batch, length in ((1, 1), (1, 128)):
            result.append(
                (
                    f"mlp_{batch}x{length}",
                    [
                        "-m",
                        "compilelab",
                        *common,
                        "--workload",
                        "gated-mlp",
                        "--dtype",
                        mlp_dtype,
                        "--batch-size",
                        str(batch),
                        "--sequence-length",
                        str(length),
                    ],
                )
            )
        result.append(
            ("graph_break", ["-m", "compilelab.graph_break", "--device", args.device])
        )
        result.append(
            (
                "dynamic_shapes",
                [
                    "-m",
                    "compilelab.dynamic_shapes",
                    "--sequence-lengths",
                    "32,64,128,32,64",
                ],
            )
        )
        if args.device == "cuda":
            result.append(
                (
                    "mlp_codegen",
                    [
                        "-m",
                        "compilelab.codegen",
                        "--warmup",
                        "50",
                        "--iterations",
                        "500",
                        "--profile-iterations",
                        "20",
                    ],
                )
            )
    if args.suite in {"qwen", "all"}:
        for trial in range(1, args.qwen_trials + 1):
            for dtype in ("float32", "bfloat16"):
                for configuration in ("no-cudagraphs", "reduce-overhead"):
                    arguments = [
                        "-m",
                        "compilelab.qwen",
                        "--dtype",
                        dtype,
                        "--configuration",
                        configuration,
                        "--prompt-lengths",
                        args.prompt_lengths,
                        "--decode-steps",
                        args.decode_steps,
                        "--warmup",
                        str(args.qwen_warmup),
                        "--iterations",
                        str(args.qwen_iterations),
                    ]
                    if args.model_path:
                        arguments += ["--model-path", str(args.model_path)]
                    if dtype == "bfloat16":
                        arguments += [
                            "--attention-backend",
                            "math",
                            "--continue-on-mismatch",
                        ]
                    result.append((f"qwen_{dtype}_{configuration}_{trial}", arguments))
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        min(
            args.trials,
            args.qwen_trials,
            args.warmup,
            args.iterations,
            args.qwen_warmup,
            args.qwen_iterations,
        )
        < 1
    ):
        raise ValueError("trial, warmup, and iteration counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(
            f"{manifest_path} exists; use --resume or choose a new output directory"
        )
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        manifest = {
            "schema_version": 1,
            "run_utc": datetime.now(UTC).isoformat(),
            "git_revision": git.stdout.strip(),
            "dirty_worktree": bool(dirty.stdout.strip()),
            "cache_policy": "fresh worker process, private Inductor + Triton caches; remote graph cache disabled",
            "execution": "serial; no benchmark workers run concurrently",
            "runs": [],
        }
    successful = {
        entry["name"] for entry in manifest["runs"] if entry["returncode"] == 0
    }
    for name, arguments in jobs(args):
        if name in successful:
            print(f"Already complete: {name}", flush=True)
            continue
        destination = args.output_dir / name
        destination.mkdir(exist_ok=True)
        command = [sys.executable, *arguments, "--output-dir", str(destination)]
        source_module = (
            "compilelab.benchmark" if arguments[1] == "compilelab" else arguments[1]
        )
        source = Path(__file__).parent / (source_module.rsplit(".", 1)[-1] + ".py")
        (destination / "source.py").write_bytes(source.read_bytes())
        entry: dict[str, Any] = {
            "name": name,
            "command": shlex.join(command),
            "argv": command,
            "source_file": str(source.relative_to(Path.cwd())),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "gpu_before": gpu_snapshot(),
            "start_utc": datetime.now(UTC).isoformat(),
        }
        print(f"Running {name}: {shlex.join(command)}", flush=True)
        with tempfile.TemporaryDirectory(prefix="torchcompile-experiment-") as cache:
            env = os.environ.copy()
            env.update(
                {
                    "TORCHINDUCTOR_CACHE_DIR": str(Path(cache) / "inductor"),
                    "TRITON_CACHE_DIR": str(Path(cache) / "triton"),
                    "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE": "0",
                    "TORCH_LOGS": "recompiles,graph_breaks",
                }
            )
            process = subprocess.run(
                command, env=env, capture_output=True, text=True, check=False
            )
            for stream, value in (
                ("stdout", process.stdout),
                ("stderr", process.stderr),
            ):
                (destination / f"{stream}.txt").write_text(
                    value.replace(cache, "<compiler-cache>").replace(
                        str(Path.cwd()), "<project>"
                    ),
                    encoding="utf-8",
                )
        entry.update(
            {
                "returncode": process.returncode,
                "end_utc": datetime.now(UTC).isoformat(),
                "gpu_after": gpu_snapshot(),
                "result": str(
                    destination.relative_to(args.output_dir) / "results.json"
                ),
            }
        )
        result_path = destination / "results.json"
        if result_path.exists():
            entry["result_sha256"] = hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest()
            entry["study_status"] = json.loads(
                result_path.read_text(encoding="utf-8")
            ).get("study_status", "passed" if process.returncode == 0 else "failed")
        manifest["runs"].append(entry)
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Finished {name}: exit {process.returncode}", flush=True)
        if process.returncode and not args.keep_going:
            raise RuntimeError(f"{name} failed; inspect {destination / 'stderr.txt'}")
    if any(
        entry["returncode"] != 0
        and entry["name"]
        not in {row["name"] for row in manifest["runs"] if row["returncode"] == 0}
        for entry in manifest["runs"]
    ):
        raise RuntimeError(
            "Some experiments failed; manifest and per-worker logs record the failures"
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fresh-process compiler and Qwen experiment suite"
    )
    parser.add_argument("--suite", choices=("micro", "qwen", "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--qwen-trials", type=int, default=2)
    parser.add_argument("--qwen-warmup", type=int, default=3)
    parser.add_argument("--qwen-iterations", type=int, default=10)
    parser.add_argument("--prompt-lengths", default="32,44,128,512")
    parser.add_argument("--decode-steps", default="8,32")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/benchmark_runs")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
