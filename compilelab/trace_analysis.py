"""Summarize actual CUDA execution slices in exported Chrome traces."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def interval_union_us(intervals: list[tuple[float, float]]) -> float:
    """Count overlapping kernel intervals once, across all streams."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0.0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def summarize_trace(path: Path) -> dict[str, Any]:
    events = json.loads(path.read_text(encoding="utf-8"))["traceEvents"]
    durations: dict[str, list[float]] = defaultdict(list)
    intervals: list[tuple[float, float]] = []
    launch_calls: Counter[str] = Counter()
    graph_calls: Counter[str] = Counter()
    compile_events: Counter[str] = Counter()
    cpu_ops: Counter[str] = Counter()
    for event in events:
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        category = event.get("cat", "")
        if category == "kernel":
            duration = float(event["dur"])
            timestamp = float(event["ts"])
            durations[name].append(duration)
            intervals.append((timestamp, timestamp + duration))
        elif category in {"cuda_runtime", "cuda_driver"}:
            if "LaunchKernel" in name:
                launch_calls[name] += 1
            if "GraphLaunch" in name:
                graph_calls[name] += 1
        if category == "cpu_op":
            cpu_ops[name] += 1
        if any(
            marker in name
            for marker in (
                "_compile.compile_inner",
                "compile_inner (dynamo_timed)",
                "GraphLowering.compile",
                "Scheduler.codegen",
            )
        ):
            compile_events[name] += 1
    kernels = [
        {
            "name": name,
            "launch_count": len(values),
            "total_device_time_us": sum(values),
            "mean_device_time_us": sum(values) / len(values),
        }
        for name, values in sorted(
            durations.items(), key=lambda item: sum(item[1]), reverse=True
        )
    ]
    window = (
        max(end for _, end in intervals) - min(start for start, _ in intervals)
        if intervals
        else 0.0
    )
    active = interval_union_us(intervals)
    return {
        "launch_count": sum(len(values) for values in durations.values()),
        "kernels": kernels,
        "kernel_time_sum_us": sum(sum(values) for values in durations.values()),
        "kernel_window_us": window,
        "kernel_active_union_us": active,
        "kernel_active_fraction": active / window if window else None,
        "activity_fraction_scope": (
            "Union of traced kernel intervals / first-to-last kernel window; "
            "not hardware utilization, occupancy, or memory bandwidth."
        ),
        "host_launch_api_calls": dict(launch_calls),
        "cuda_graph_replay_calls": dict(graph_calls),
        "compilation_events": dict(compile_events),
        "cpu_operator_counts": dict(cpu_ops),
        "gpu_memcpy_events": sum(
            event.get("cat") == "gpu_memcpy" and event.get("ph") == "X"
            for event in events
        ),
        "gpu_memset_events": sum(
            event.get("cat") == "gpu_memset" and event.get("ph") == "X"
            for event in events
        ),
    }
