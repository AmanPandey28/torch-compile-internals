from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from compilelab.workload import GatedMLP

MODE_VALUES: dict[str, bool | None] = {
    "static": False,
    "automatic": None,
    "dynamic": True,
}
MODE_LABELS = {
    "static": "Static",
    "automatic": "Automatic",
    "dynamic": "Upfront dynamic",
}


@dataclass(frozen=True)
class ExperimentConfig:
    sequence_lengths: tuple[int, ...]
    batch_size: int
    model_dim: int
    hidden_dim: int
    seed: int


def parse_sequence_lengths(value: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "sequence lengths must be comma-separated integers"
        ) from exc
    if len(lengths) < 2 or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError(
            "provide at least two positive sequence lengths"
        )
    return lengths


def describe_example_input(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": [str(dimension) for dimension in value.shape],
            "dtype": str(value.dtype).removeprefix("torch."),
        }
    return {"kind": type(value).__name__, "value": str(value)}


def execute_mode(mode: str, config: ExperimentConfig) -> dict[str, Any]:
    if mode not in MODE_VALUES:
        raise ValueError(f"unknown compilation mode: {mode}")

    torch.manual_seed(config.seed)
    model = GatedMLP(config.model_dim, config.hidden_dim).eval()
    graph_records: list[dict[str, Any]] = []

    def capture_backend(
        graph_module: torch.fx.GraphModule,
        example_inputs: list[Any],
    ) -> Callable[..., Any]:
        signature = [describe_example_input(value) for value in example_inputs]
        graph_records.append(
            {
                "graph_number": len(graph_records) + 1,
                "symbolic_integer_inputs": sum(
                    value["kind"] == "SymInt" for value in signature
                ),
                "example_inputs": signature,
                "code": graph_module.code.strip() + "\n",
            }
        )
        return graph_module.forward

    compiled_model = torch.compile(
        model,
        backend=capture_backend,
        fullgraph=True,
        dynamic=MODE_VALUES[mode],
    )
    calls: list[dict[str, Any]] = []

    with torch.inference_mode():
        for call_index, sequence_length in enumerate(config.sequence_lengths):
            generator = torch.Generator().manual_seed(config.seed + call_index + 1)
            inputs = torch.randn(
                config.batch_size,
                sequence_length,
                config.model_dim,
                generator=generator,
            )
            reference = model(inputs)
            graphs_before = len(graph_records)
            actual = compiled_model(inputs)
            graphs_after = len(graph_records)
            correct = torch.allclose(reference, actual, atol=1e-5, rtol=1e-5)
            if not correct:
                raise AssertionError(
                    f"compiled output did not match eager output for length "
                    f"{sequence_length} in {mode} mode"
                )
            calls.append(
                {
                    "call": call_index + 1,
                    "sequence_length": sequence_length,
                    "new_graph_compiled": graphs_after > graphs_before,
                    "backend_compilations_after_call": graphs_after,
                    "correct": True,
                }
            )

    return {
        "mode": mode,
        "torch_compile_dynamic": MODE_VALUES[mode],
        "backend_compilations": len(graph_records),
        "calls": calls,
        "graphs": graph_records,
    }


def parse_recompile_events(log: str) -> list[dict[str, Any]]:
    recompile_pattern = re.compile(
        r"\[__recompiles\]\s+Recompiling function\s+(.+?)(?:\s+in\s+.*)?$"
    )
    failure_pattern = re.compile(r"\[__recompiles\]\s+-\s+\d+/\d+:\s+(.+)$")
    events: list[dict[str, Any]] = []

    for line in log.splitlines():
        recompile_match = recompile_pattern.search(line)
        if recompile_match:
            events.append(
                {
                    "function": recompile_match.group(1).strip(),
                    "guard_failures": [],
                }
            )
            continue
        failure_match = failure_pattern.search(line)
        if failure_match and events:
            events[-1]["guard_failures"].append(failure_match.group(1).strip())

    return events


def worker_command(mode: str, config: ExperimentConfig) -> list[str]:
    return [
        sys.executable,
        "-m",
        "compilelab.dynamic_shapes",
        "--worker-mode",
        mode,
        "--sequence-lengths",
        ",".join(str(length) for length in config.sequence_lengths),
        "--model-dim",
        str(config.model_dim),
        "--hidden-dim",
        str(config.hidden_dim),
        "--seed",
        str(config.seed),
    ]


def run_isolated_mode(mode: str, config: ExperimentConfig) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["TORCH_LOGS"] = "recompiles"
    completed = subprocess.run(
        worker_command(mode, config),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        reason = detail[-1] if detail else "worker exited without an error message"
        raise RuntimeError(f"{mode} worker failed: {reason}")

    result = json.loads(completed.stdout)
    result["recompile_events"] = parse_recompile_events(completed.stderr)
    return result


def render_recompile_log(modes: dict[str, dict[str, Any]]) -> str:
    lines = ["Dynamo guard-triggered recompilations", ""]
    for mode in MODE_VALUES:
        events = modes[mode]["recompile_events"]
        lines.append(f"[{mode}]")
        if not events:
            lines.append("No recompilations recorded.")
        for event_index, event in enumerate(events, start=1):
            lines.append(f"Recompile {event_index}: function {event['function']}")
            for failure in event["guard_failures"]:
                lines.append(f"- {failure}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_automatic_graphs(mode: dict[str, Any]) -> str:
    sections = [
        "# FX graphs captured with torch.compile(dynamic=None).",
        "# Graph 1 starts specialized; later graphs may introduce symbolic inputs.",
    ]
    for graph in mode["graphs"]:
        sections.extend(
            [
                "",
                f"# ---- graph {graph['graph_number']} ----",
                graph["code"].rstrip(),
            ]
        )
    return "\n".join(sections).rstrip() + "\n"


def render_call_state(call: dict[str, Any]) -> str:
    graph = call["backend_compilations_after_call"]
    if call["new_graph_compiled"]:
        return f"compile G{graph}"
    return "cache reuse"


def render_summary(result: dict[str, Any]) -> str:
    modes = result["modes"]
    sequence = result["experiment"]["sequence_lengths"]
    mode_rows = []
    dynamic_labels = {"static": "False", "automatic": "None", "dynamic": "True"}
    for mode in MODE_VALUES:
        entry = modes[mode]
        mode_rows.append(
            f"| {MODE_LABELS[mode]} | `{dynamic_labels[mode]}` | "
            f"{entry['backend_compilations']} | {len(entry['recompile_events'])} |"
        )

    call_rows = []
    for index, length in enumerate(sequence):
        suffix = " (repeat)" if length in sequence[:index] else ""
        states = [
            render_call_state(modes[mode]["calls"][index]) for mode in MODE_VALUES
        ]
        call_rows.append(
            f"| {index + 1} | {length}{suffix} | {states[0]} | "
            f"{states[1]} | {states[2]} |"
        )

    unique_lengths = len(set(sequence))
    static = modes["static"]
    automatic = modes["automatic"]
    dynamic = modes["dynamic"]
    static_observation = (
        f"Static mode compiled {static['backend_compilations']} graphs for "
        f"{unique_lengths} unique sequence lengths and logged "
        f"{len(static['recompile_events'])} guard-triggered recompiles."
    )
    automatic_symbolic = automatic["graphs"][-1]["symbolic_integer_inputs"]
    symbolic_input_label = "input" if automatic_symbolic == 1 else "inputs"
    automatic_observation = (
        f"Automatic mode compiled {automatic['backend_compilations']} graphs; "
        f"its final graph signature contains {automatic_symbolic} symbolic "
        f"integer {symbolic_input_label}."
    )
    if dynamic["backend_compilations"] == 1:
        dynamic_observation = (
            "Upfront dynamic mode introduced the symbolic dimension on its first "
            "capture and handled the full sequence with one graph."
        )
    else:
        dynamic_observation = (
            f"Upfront dynamic mode still compiled {dynamic['backend_compilations']} "
            "graphs, demonstrating that `dynamic=True` does not prevent every "
            "possible specialization."
        )

    return f"""# Dynamic-shape specialization analysis

This experiment varies only the sequence dimension of a gated transformer MLP
and runs each `torch.compile` policy in a separate process. A pass-through
backend counts Dynamo graph compilations without adding Inductor code-generation
effects.

| Mode | `dynamic` argument | Backend compilations | Guard-triggered recompiles |
|---|---:|---:|---:|
{chr(10).join(mode_rows)}

| Call | Sequence length | Static | Automatic | Upfront dynamic |
|---:|---:|---:|---:|---:|
{chr(10).join(call_rows)}

{static_observation} {automatic_observation} {dynamic_observation}

Repeated lengths reuse cached graphs rather than compiling again.

All outputs matched eager execution. These counts describe Dynamo capture for
this workload and input sequence; they do not establish that a more dynamic
graph is faster. Dynamic kernels can trade fewer compilations for broader guards
or less specialized generated code.
"""


def run(config: ExperimentConfig, output_dir: Path) -> dict[str, Any]:
    if config.batch_size != 1:
        raise ValueError("this controlled experiment requires batch size 1")
    if config.model_dim <= 0 or config.hidden_dim <= 0:
        raise ValueError("model and hidden dimensions must be positive")

    captured_modes = {mode: run_isolated_mode(mode, config) for mode in MODE_VALUES}
    modes: dict[str, dict[str, Any]] = {}
    for mode, captured in captured_modes.items():
        graph_metadata = [
            {key: value for key, value in graph.items() if key != "code"}
            for graph in captured["graphs"]
        ]
        modes[mode] = {
            **{key: value for key, value in captured.items() if key != "graphs"},
            "graphs": graph_metadata,
        }
    result = {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "execution_device": "cpu",
        },
        "experiment": {
            "workload": "GatedMLP",
            "sequence_lengths": list(config.sequence_lengths),
            "batch_size": config.batch_size,
            "model_dim": config.model_dim,
            "hidden_dim": config.hidden_dim,
            "dtype": "float32",
            "seed": config.seed,
            "fullgraph": True,
            "backend": "pass-through graph capture",
            "isolated_process_per_mode": True,
        },
        "modes": modes,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(render_summary(result), encoding="utf-8")
    (output_dir / "recompiles.txt").write_text(
        render_recompile_log(modes), encoding="utf-8"
    )
    (output_dir / "automatic_graphs.py").write_text(
        render_automatic_graphs(captured_modes["automatic"]), encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare torch.compile dynamic-shape specialization policies"
    )
    parser.add_argument(
        "--sequence-lengths",
        type=parse_sequence_lengths,
        default=(32, 64, 128, 32, 64),
    )
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/dynamic_shapes")
    )
    parser.add_argument(
        "--worker-mode",
        choices=tuple(MODE_VALUES),
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        sequence_lengths=args.sequence_lengths,
        batch_size=1,
        model_dim=args.model_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    if args.worker_mode is not None:
        print(json.dumps(execute_mode(args.worker_mode, config)))
        return

    result = run(config, args.output_dir)
    print(render_summary(result))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
