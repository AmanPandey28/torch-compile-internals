from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn


class PythonDataDependentBranch(nn.Module):
    """A Python branch whose direction depends on a tensor value."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        if x.sum() > 0:
            return torch.sin(y)
        return torch.cos(y)


class CondDataDependentBranch(nn.Module):
    """The same behavior expressed as capturable tensor control flow."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        return torch.cond(
            x.sum() > 0,
            lambda operand: torch.sin(operand),
            lambda operand: torch.cos(operand),
            (y,),
        )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def capture_regions(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    *,
    fullgraph: bool,
) -> tuple[list[str], list[torch.Tensor]]:
    graph_code: list[str] = []

    def backend(
        graph_module: torch.fx.GraphModule,
        _example_inputs: list[torch.Tensor],
    ) -> Callable[..., Any]:
        graph_code.append(graph_module.code)
        return graph_module.forward

    compiled = torch.compile(model, backend=backend, fullgraph=fullgraph)
    outputs = [compiled(value) for value in inputs]
    return graph_code, outputs


def fullgraph_failure(model: nn.Module, example: torch.Tensor) -> dict[str, str]:
    try:
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        compiled(example)
    except Exception as exc:
        first_line = str(exc).strip().splitlines()[0]
        return {"exception_type": type(exc).__name__, "reason": first_line}
    raise AssertionError("the Python data-dependent branch unexpectedly compiled")


def joined_graphs(graphs: list[str]) -> str:
    sections = []
    for index, graph in enumerate(graphs, start=1):
        sections.append(f"# ---- graph {index} ----\n{graph.strip()}\n")
    return "\n\n".join(sections)


def render_summary(result: dict[str, Any]) -> str:
    original = result["original"]
    fixed = result["fixed"]
    return f"""# Data-dependent graph-break case

## Question

What happens when a Python branch depends on a runtime tensor value, and how
does expressing that branch with `torch.cond` change Dynamo capture?

| Check | Python `if` | `torch.cond` |
|---|---:|---:|
| Correct for positive input | {original["positive_correct"]} | {fixed["positive_correct"]} |
| Correct for negative input | {original["negative_correct"]} | {fixed["negative_correct"]} |
| Captured graphs across both paths | {original["graph_count"]} | {fixed["graph_count"]} |
| Accepted by `fullgraph=True` | no | yes |

The Python form fails full-graph capture with
`{original["fullgraph_failure"]["reason"]}`. Dynamo captures a predicate graph
and specializes separate continuation graphs for the two Python branches. The
`torch.cond` form represents the branch inside one captured graph.

This is a structural result, not a performance claim. `torch.cond` avoids the
graph break, but its runtime benefit still depends on backend support, branch
cost, shapes, and hardware.
"""


def run(device: torch.device, output_dir: Path) -> dict[str, Any]:
    positive = torch.ones(16, device=device)
    negative = -torch.ones(16, device=device)
    inputs = (positive, negative)

    original_model = PythonDataDependentBranch().to(device).eval()
    fixed_model = CondDataDependentBranch().to(device).eval()

    with torch.inference_mode():
        references = [original_model(value) for value in inputs]
        original_graphs, original_outputs = capture_regions(
            original_model, inputs, fullgraph=False
        )
        failure = fullgraph_failure(original_model, positive)
        fixed_graphs, fixed_outputs = capture_regions(
            fixed_model, inputs, fullgraph=True
        )

    original_correct = [
        torch.allclose(reference, actual)
        for reference, actual in zip(references, original_outputs)
    ]
    fixed_correct = [
        torch.allclose(reference, actual)
        for reference, actual in zip(references, fixed_outputs)
    ]
    if not all(original_correct + fixed_correct):
        raise AssertionError("a compiled branch output did not match eager")

    result = {
        "schema_version": 1,
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU"
            ),
        },
        "case": "data_dependent_python_branch",
        "original": {
            "implementation": "PythonDataDependentBranch",
            "positive_correct": original_correct[0],
            "negative_correct": original_correct[1],
            "graph_count": len(original_graphs),
            "fullgraph_failure": failure,
        },
        "fixed": {
            "implementation": "CondDataDependentBranch",
            "positive_correct": fixed_correct[0],
            "negative_correct": fixed_correct[1],
            "graph_count": len(fixed_graphs),
            "fullgraph": True,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        render_summary(result), encoding="utf-8"
    )
    (output_dir / "python_if_graphs.py").write_text(
        joined_graphs(original_graphs), encoding="utf-8"
    )
    (output_dir / "torch_cond_graph.py").write_text(
        joined_graphs(fixed_graphs), encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a data-dependent Python branch with torch.cond"
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/graph_break_case"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(resolve_device(args.device), args.output_dir)
    print(render_summary(result))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
