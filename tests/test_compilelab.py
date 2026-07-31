from __future__ import annotations

import torch

from compilelab.benchmark import capture_dynamo_graph, parse_shape
from compilelab.codegen import classify_cuda_kernel, classify_inductor_wrapper
from compilelab.dynamic_shapes import (
    ExperimentConfig,
    execute_mode,
    parse_recompile_events,
    parse_sequence_lengths,
)
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


def test_inductor_wrapper_classification() -> None:
    source = """
fused_silu_mul = async_compile.triton('fused_silu_mul', '...')

class Runner:
    def call(self, args):
        extern_kernels.mm(args[0], args[1], out=args[2])
        extern_kernels.mm(args[0], args[3], out=args[4])
        fused_silu_mul.run(args[2], args[4], 16)
        extern_kernels.mm(args[2], args[5], out=args[6])
"""

    result = classify_inductor_wrapper(source)

    assert result["external_kernel_calls"] == {"mm": 3}
    assert result["triton_kernel_definitions"] == ["fused_silu_mul"]
    assert result["triton_kernel_launches"] == ["fused_silu_mul"]
    assert result["wrapper_launch_site_count"] == 4


def test_cuda_kernel_name_classification() -> None:
    assert classify_cuda_kernel("cutlass_80_tensorop_gemm") == "external_gemm"
    assert classify_cuda_kernel("triton_poi_fused_silu_mul") == "generated_triton"
    assert classify_cuda_kernel("vectorized_silu_kernel") == "eager_silu"
    assert classify_cuda_kernel("BinaryFunctor<MulFunctor<float>>") == "eager_multiply"


def test_parse_sequence_lengths() -> None:
    assert parse_sequence_lengths("64, 128,256") == (64, 128, 256)


def test_recompile_log_parsing_removes_runtime_prefixes() -> None:
    prefix = "V123 10:00:00.000000 42 guards.py:1] [0/1] [__recompiles]"
    log = "\n".join(
        (
            f"{prefix} Recompiling function forward in /private/workload.py:26",
            f"{prefix} triggered by the following guard failure(s):",
            f"{prefix} - 0/0: tensor 'x' size mismatch at index 1. "
            "expected 64, actual 128",
        )
    )

    assert parse_recompile_events(log) == [
        {
            "function": "forward",
            "guard_failures": [
                "tensor 'x' size mismatch at index 1. expected 64, actual 128"
            ],
        }
    ]


def test_static_shape_mode_compiles_once_per_unique_length() -> None:
    torch.compiler.reset()
    try:
        result = execute_mode(
            "static",
            ExperimentConfig(
                sequence_lengths=(4, 8, 4),
                batch_size=1,
                model_dim=8,
                hidden_dim=16,
                seed=7,
            ),
        )
    finally:
        torch.compiler.reset()

    assert result["backend_compilations"] == 2
    assert [call["new_graph_compiled"] for call in result["calls"]] == [
        True,
        True,
        False,
    ]
    assert all(call["correct"] for call in result["calls"])
