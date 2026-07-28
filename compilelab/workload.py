from __future__ import annotations

import torch
from torch import nn


class PointwiseReduction(nn.Module):
    """A small workload with pointwise operations and a final reduction."""

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.silu(x + bias)
        return (y * y).mean(dim=-1)


class GatedMLP(nn.Module):
    """A Llama-style gated feed-forward block used in transformer decoders."""

    def __init__(self, model_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if model_dim <= 0 or hidden_dim <= 0:
            raise ValueError("model_dim and hidden_dim must be positive")
        self.gate_proj = nn.Linear(model_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(model_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(gated)


def make_inputs(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(shape) < 2 or any(dimension <= 0 for dimension in shape):
        raise ValueError("shape must contain at least two positive dimensions")

    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    bias = torch.randn(
        (shape[-1],), device=device, dtype=dtype, generator=generator
    )
    return x, bias


def make_mlp_input(
    shape: tuple[int, int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor]:
    if any(dimension <= 0 for dimension in shape):
        raise ValueError("shape dimensions must be positive")

    generator_device = device.type if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    return (x,)
