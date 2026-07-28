from __future__ import annotations

import torch

from compilelab.benchmark import capture_dynamo_graph, parse_shape
from compilelab.graph_break import (
    CondDataDependentBranch,
    PythonDataDependentBranch,
    capture_regions,
    fullgraph_failure,
)
from compilelab.workload import GatedMLP, PointwiseReduction, make_inputs


def test_workload_shape_and_values_are_compilable() -> None:
    model = PointwiseReduction().eval()
    inputs = make_inputs(
        (4, 16), device=torch.device("cpu"), dtype=torch.float32, seed=7
    )

    with torch.inference_mode():
        eager = model(*inputs)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        actual = compiled(*inputs)

    assert eager.shape == (4,)
    torch.testing.assert_close(actual, eager)


def test_custom_backend_captures_one_graph() -> None:
    model = PointwiseReduction().eval()
    inputs = make_inputs(
        (4, 16), device=torch.device("cpu"), dtype=torch.float32, seed=7
    )

    with torch.inference_mode():
        graph_code, graph_count = capture_dynamo_graph(model, inputs)

    assert graph_count == 1
    assert len(graph_code) == 1
    assert "torch.nn.functional.silu" in graph_code[0]
    assert "mean" in graph_code[0]


def test_parse_shape() -> None:
    assert parse_shape("1, 128, 1024") == (1, 128, 1024)


def test_data_dependent_branch_is_represented_by_torch_cond() -> None:
    positive = torch.ones(4)
    negative = -torch.ones(4)
    inputs = (positive, negative)
    reference_model = PythonDataDependentBranch().eval()
    fixed_model = CondDataDependentBranch().eval()

    with torch.inference_mode():
        references = [reference_model(value) for value in inputs]
        graphs, outputs = capture_regions(fixed_model, inputs, fullgraph=True)

    assert len(graphs) == 1
    assert "torch.ops.higher_order.cond" in graphs[0]
    for reference, actual in zip(references, outputs):
        torch.testing.assert_close(actual, reference)


def test_python_data_dependent_branch_fails_fullgraph() -> None:
    failure = fullgraph_failure(PythonDataDependentBranch(), torch.ones(4))

    assert failure["exception_type"] == "Unsupported"
    assert "Data-dependent branching" in failure["reason"]


def test_gated_mlp_shape_and_fullgraph_capture() -> None:
    torch.manual_seed(7)
    model = GatedMLP(model_dim=16, hidden_dim=32).eval()
    inputs = (torch.randn(2, 4, 16),)

    with torch.inference_mode():
        reference = model(*inputs)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        actual = compiled(*inputs)
        graph_code, graph_count = capture_dynamo_graph(model, inputs)

    assert reference.shape == (2, 4, 16)
    assert graph_count == 1
    assert "torch._C._nn.linear" in graph_code[0]
    assert "torch.nn.functional.silu" in graph_code[0]
    torch.testing.assert_close(actual, reference)
