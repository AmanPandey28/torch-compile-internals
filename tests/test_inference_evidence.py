from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from compilelab.benchmark import break_even_calls
from compilelab.evidence import assemble, copy_text_artifact, digest, replay_argv
from compilelab.experiments import build_parser, jobs
from compilelab.graph_break import CondDataDependentBranch, PythonDataDependentBranch
from compilelab.qwen import (
    compare_logits,
    prepare_work,
    preserve_graph_outputs,
    prompt_ids,
)
from compilelab.trace_analysis import interval_union_us, summarize_trace


def test_trace_counts_kernels_without_annotations_or_memory_operations(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"cat": "kernel", "ph": "X", "name": "gemm", "ts": 10, "dur": 20},
                    {
                        "cat": "kernel",
                        "ph": "X",
                        "name": "pointwise",
                        "ts": 25,
                        "dur": 10,
                    },
                    {
                        "cat": "gpu_user_annotation",
                        "ph": "X",
                        "name": "iteration",
                        "ts": 0,
                        "dur": 50,
                    },
                    {"cat": "gpu_memcpy", "ph": "X", "name": "copy", "ts": 2, "dur": 4},
                    {"cat": "cpu_op", "ph": "X", "name": "aten::mm", "ts": 0, "dur": 7},
                    {
                        "cat": "cuda_runtime",
                        "ph": "X",
                        "name": "cudaGraphLaunch",
                        "ts": 0,
                        "dur": 1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    summary = summarize_trace(trace)
    assert summary["launch_count"] == 2
    assert summary["kernel_time_sum_us"] == 30
    assert summary["kernel_active_union_us"] == 25
    assert summary["kernel_active_fraction"] == 1
    assert summary["cuda_graph_replay_calls"] == {"cudaGraphLaunch": 1}
    assert summary["gpu_memcpy_events"] == 1


def test_interval_union_preserves_gaps_and_overlaps() -> None:
    assert interval_union_us([(0, 10), (5, 12), (20, 23)]) == 15
    assert interval_union_us([]) == 0


def test_each_decode_request_starts_from_a_fresh_cache() -> None:
    observed = []

    def forward(ids: torch.Tensor, past_key_values=None, **kwargs):
        cache = [] if past_key_values is None else past_key_values
        cache.append(int(ids.numel()))
        observed.append(tuple(cache))
        return SimpleNamespace(past_key_values=cache, logits=torch.tensor(cache[-1:]))

    ids = torch.ones(1, 4, dtype=torch.long)
    tokens = [torch.ones(1, 1, dtype=torch.long) for _ in range(2)]
    for _ in range(2):
        work = prepare_work("decode", forward, ids, tokens)
        work()
    assert observed == [(4,), (4, 1), (4, 1, 1)] * 2


def test_logit_validation_rejects_nan_and_reports_token_disagreement() -> None:
    reference = [torch.tensor([[1.0, 2.0]])]
    invalid = compare_logits(
        reference, [torch.tensor([[float("nan"), 2.0]])], atol=1e-3, rtol=1e-3
    )
    assert not invalid["passed"]
    changed = compare_logits(reference, [torch.tensor([[2.0, 1.0]])], atol=2, rtol=0)
    assert changed["passed"]
    assert not changed["greedy_ids_equal"]


def test_prompt_lengths_are_exact_and_native_ids_are_preserved() -> None:
    base = torch.tensor([[1, 2, 3]])
    assert torch.equal(prompt_ids(base, 3), base)
    assert prompt_ids(base, 8).tolist() == [[1, 2, 3, 1, 2, 3, 1, 2]]


def test_graph_output_ownership_survives_next_invocation(monkeypatch) -> None:
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: None)
    storage = torch.zeros(2)

    def forward(value: float):
        storage.fill_(value)
        layer = SimpleNamespace(keys=storage, values=storage)
        return SimpleNamespace(
            logits=storage, past_key_values=SimpleNamespace(layers=[layer])
        )

    safe_forward = preserve_graph_outputs(forward)
    first = safe_forward(1)
    safe_forward(2)
    assert first.logits.tolist() == [1, 1]
    assert first.past_key_values.layers[0].keys.tolist() == [1, 1]
    assert first.past_key_values.layers[0].values.tolist() == [1, 1]
    assert first.logits.data_ptr() != storage.data_ptr()


@pytest.mark.parametrize(
    "value",
    [
        torch.zeros(4),
        torch.tensor([1.0, -1.0, 2.0, -2.0]),
        torch.ones(8)[::2],
        torch.ones(1),
    ],
)
def test_cond_boundary_shapes_and_zero_predicate(value: torch.Tensor) -> None:
    with torch.inference_mode():
        actual = torch.compile(
            CondDataDependentBranch(), fullgraph=True, backend="eager"
        )(value)
        torch.testing.assert_close(actual, PythonDataDependentBranch()(value))


def test_break_even_uses_extra_first_call_cost_and_requires_savings() -> None:
    assert break_even_calls(10.1, 0.2, 0.1) == 100
    assert break_even_calls(10, 0.1, 0.2) is None


@pytest.mark.parametrize("device,dtype", [("cpu", "float32"), ("cuda", "float16")])
def test_micro_suite_uses_supported_mlp_dtype(device: str, dtype: str) -> None:
    args = build_parser().parse_args(["--suite", "micro", "--device", device])
    cases = jobs(args)
    for name, command in cases:
        if name.startswith("mlp_") and name != "mlp_codegen":
            assert command[command.index("--dtype") + 1] == dtype
    assert any(name == "mlp_codegen" for name, _ in cases) == (device == "cuda")


def export_fixture(run_dir: Path, name: str, result: dict) -> dict:
    destination = run_dir / name
    destination.mkdir(parents=True)
    raw_path = destination / "results.json"
    raw_path.write_text(json.dumps(result), encoding="utf-8")
    (destination / "stderr.txt").write_text("", encoding="utf-8")
    entry = {
        "name": name,
        "argv": [
            "/machine/python",
            "-m",
            "compilelab",
            "--shape",
            "1024,4096",
            "--output-dir",
            str(destination),
        ],
        "returncode": 0,
        "source_file": "compilelab/benchmark.py",
        "source_sha256": "a" * 64,
        "result_sha256": digest(raw_path),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps({"runs": [entry]}), encoding="utf-8"
    )
    return entry


def test_result_export_preserves_raw_bytes_and_normalizes_replay(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs"
    output_dir = tmp_path / "results"
    entry = export_fixture(
        run_dir,
        "pointwise_test",
        {
            "timing": {
                "eager": {"median_ms": 2},
                "compiled": {"median_ms": 1, "first_call_ms": 21},
                "break_even_calls": 20,
            }
        },
    )
    command = replay_argv(entry)
    assert command[0] == "python"
    assert command[-1] == "artifacts/benchmark_runs/pointwise_test"
    assert command[1:-1] == entry["argv"][1:-1]
    assert entry["argv"][0] == "/machine/python"
    evidence = assemble(run_dir, output_dir)
    record = evidence["runs"][0]
    assert record["source_sha256"] == entry["source_sha256"]
    assert digest(output_dir / record["result"]) == entry["result_sha256"]
    assert record["verified_break_even_calls"] == 20
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "evidence.json",
        "pointwise_test",
        "results.md",
    ]


def test_result_export_distinguishes_rejected_accuracy_from_success(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs"
    output_dir = tmp_path / "results"
    export_fixture(
        run_dir,
        "qwen_bfloat16_no-cudagraphs_1",
        {
            "study_status": "rejected",
            "measurements": [
                {
                    "status": "correctness_failed",
                    "performance_accepted": False,
                    "prompt_tokens": 32,
                    "phase": "prefill",
                    "decode_steps": 0,
                    "correctness": {
                        "passed": False,
                        "max_abs_error": 0.25,
                        "mean_abs_error": 0.04,
                        "greedy_ids_equal": False,
                        "atol": 0.02,
                        "rtol": 0.02,
                    },
                }
            ],
        },
    )
    evidence = assemble(run_dir, output_dir)
    assert evidence["runs"][0]["status"] == "rejected"
    report = (output_dir / "results.md").read_text(encoding="utf-8")
    assert "| 32 | prefill/0 | 0.25 | 0.04 | False |" in report
    assert "not accepted performance improvements" in report


def test_result_export_rejects_corrupt_input_before_copying(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs"
    output_dir = tmp_path / "results"
    export_fixture(run_dir, "pointwise_test", {})
    (run_dir / "pointwise_test" / "results.json").write_text(
        '{"changed": true}', encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="Result hash mismatch"):
        assemble(run_dir, output_dir)
    assert not (output_dir / "pointwise_test" / "results.json").exists()


def test_result_export_requires_separate_output_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be different"):
        assemble(tmp_path, tmp_path)


def test_text_artifact_export_trims_whitespace_and_preserves_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "graph.py"
    destination = tmp_path / "export.py"
    content = "def forward(x):   \n    return x \t\n\n\n"
    source.write_text(content, encoding="utf-8")
    copy_text_artifact(source, destination)
    assert source.read_text(encoding="utf-8") == content
    assert destination.read_text(encoding="utf-8") == "def forward(x):\n    return x\n"
