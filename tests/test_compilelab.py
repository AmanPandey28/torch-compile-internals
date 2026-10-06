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
from compilelab.projection_packing import parse_cases
from compilelab.quantization import quality_statistics
from compilelab.transformer import (
    DecomposedRMSNorm,
    PackedSelfAttention,
    PackedTransformerBlock,
    SelfAttention,
    TransformerBlock,
)
from compilelab.workload import (
    GatedMLP,
    PackedGatedMLP,
    PointwiseReduction,
    make_inputs,
)


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
            (
                f"{prefix} - 0/0: tensor 'x' size mismatch at index 1. "
                "expected 64, actual 128"
            ),
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


def test_parse_projection_packing_cases() -> None:
    assert parse_cases("1x1, 1x128,4X128") == (
        (1, 1),
        (1, 128),
        (4, 128),
    )


def test_packed_gated_mlp_preserves_state_and_gradients() -> None:
    torch.manual_seed(7)
    original = GatedMLP(model_dim=8, hidden_dim=16).eval()
    packed = PackedGatedMLP.from_unpacked(original)
    restored = packed.to_unpacked()

    torch.testing.assert_close(
        packed.gate_up_proj.weight,
        torch.cat((original.gate_proj.weight, original.up_proj.weight), dim=0),
        rtol=0,
        atol=0,
    )
    for name, parameter in original.named_parameters():
        torch.testing.assert_close(
            parameter, dict(restored.named_parameters())[name], rtol=0, atol=0
        )
    assert not packed.training
    assert not restored.training

    original_input = torch.randn(2, 3, 8, requires_grad=True)
    packed_input = original_input.detach().clone().requires_grad_(True)
    original_output = original(original_input)
    packed_output = packed(packed_input)
    torch.testing.assert_close(packed_output, original_output)

    original_output.square().mean().backward()
    packed_output.square().mean().backward()
    torch.testing.assert_close(packed_input.grad, original_input.grad)
    packed_gate_grad, packed_up_grad = packed.gate_up_proj.weight.grad.chunk(2)
    torch.testing.assert_close(packed_gate_grad, original.gate_proj.weight.grad)
    torch.testing.assert_close(packed_up_grad, original.up_proj.weight.grad)
    torch.testing.assert_close(
        packed.down_proj.weight.grad, original.down_proj.weight.grad
    )


def test_packed_gated_mlp_supports_fullgraph_compilation() -> None:
    torch.manual_seed(7)
    original = GatedMLP(model_dim=8, hidden_dim=16).eval()
    packed = PackedGatedMLP.from_unpacked(original)
    inputs = torch.randn(2, 3, 8)

    torch.compiler.reset()
    try:
        with torch.inference_mode():
            reference = original(inputs)
            compiled = torch.compile(packed, backend="eager", fullgraph=True)
            actual = compiled(inputs)
    finally:
        torch.compiler.reset()

    torch.testing.assert_close(actual, reference)


def test_packed_qkv_preserves_state_outputs_and_gradients() -> None:
    torch.manual_seed(7)
    original = SelfAttention(model_dim=8, num_heads=2).eval()
    packed = PackedSelfAttention.from_unpacked(original)
    restored = packed.to_unpacked()

    torch.testing.assert_close(
        packed.qkv_proj.weight,
        torch.cat(
            (
                original.q_proj.weight,
                original.k_proj.weight,
                original.v_proj.weight,
            ),
            dim=0,
        ),
        rtol=0,
        atol=0,
    )
    for name, parameter in original.named_parameters():
        torch.testing.assert_close(
            parameter, dict(restored.named_parameters())[name], rtol=0, atol=0
        )

    original_input = torch.randn(2, 4, 8, requires_grad=True)
    packed_input = original_input.detach().clone().requires_grad_(True)
    original_output = original(original_input)
    packed_output = packed(packed_input)
    torch.testing.assert_close(packed_output, original_output)

    original_output.square().mean().backward()
    packed_output.square().mean().backward()
    torch.testing.assert_close(packed_input.grad, original_input.grad)
    query_grad, key_grad, value_grad = packed.qkv_proj.weight.grad.chunk(3)
    torch.testing.assert_close(query_grad, original.q_proj.weight.grad)
    torch.testing.assert_close(key_grad, original.k_proj.weight.grad)
    torch.testing.assert_close(value_grad, original.v_proj.weight.grad)
    torch.testing.assert_close(
        packed.out_proj.weight.grad, original.out_proj.weight.grad
    )


def test_packed_transformer_block_round_trip_and_fullgraph() -> None:
    torch.manual_seed(7)
    original = TransformerBlock(model_dim=8, num_heads=2, hidden_dim=16).eval()
    packed = PackedTransformerBlock.from_unpacked(original)
    restored = packed.to_unpacked()
    inputs = torch.randn(2, 4, 8)

    for name, parameter in original.named_parameters():
        torch.testing.assert_close(
            parameter, dict(restored.named_parameters())[name], rtol=0, atol=0
        )

    torch.compiler.reset()
    try:
        with torch.inference_mode():
            reference = original(inputs)
            packed_eager = packed(inputs)
            compiled = torch.compile(packed, backend="eager", fullgraph=True)
            packed_compiled = compiled(inputs)
    finally:
        torch.compiler.reset()

    torch.testing.assert_close(packed_eager, reference)
    torch.testing.assert_close(packed_compiled, reference)


def test_decomposed_rms_norm_matches_native_rms_norm() -> None:
    torch.manual_seed(7)
    native = torch.nn.RMSNorm(8, eps=1e-6)
    decomposed = DecomposedRMSNorm(8, eps=1e-6)
    with torch.no_grad():
        decomposed.weight.copy_(native.weight)
    inputs = torch.randn(2, 4, 8)

    torch.testing.assert_close(decomposed(inputs), native(inputs))


def test_quantization_quality_statistics() -> None:
    reference = torch.tensor([1.0, 2.0])
    actual = torch.tensor([1.0, 1.0])

    result = quality_statistics(reference, actual)

    assert result["finite"]
    assert result["max_abs_error"] == 1.0
    assert result["mean_abs_error"] == 0.5
    assert result["root_mean_square_error"] == 2**-0.5
    assert result["cosine_similarity"] == torch.nn.functional.cosine_similarity(
        reference, actual, dim=0
    )
    assert result["sqnr_db"] is not None
