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


class PackedGatedMLP(nn.Module):
    """A gated MLP with gate and up projections packed along output features."""

    def __init__(self, model_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if model_dim <= 0 or hidden_dim <= 0:
            raise ValueError("model_dim and hidden_dim must be positive")
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.gate_up_proj = nn.Linear(
            model_dim, 2 * hidden_dim, bias=False
        )
        self.down_proj = nn.Linear(hidden_dim, model_dim, bias=False)

    @classmethod
    def from_unpacked(cls, source: GatedMLP) -> PackedGatedMLP:
        """Create a packed module with weights copied from an unpacked module."""

        model_dim = source.gate_proj.in_features
        hidden_dim = source.gate_proj.out_features
        if source.up_proj.in_features != model_dim:
            raise ValueError("gate and up projections must share their input size")
        if source.up_proj.out_features != hidden_dim:
            raise ValueError("gate and up projections must share their output size")
        if source.down_proj.in_features != hidden_dim:
            raise ValueError("down projection input must match the hidden size")
        if source.down_proj.out_features != model_dim:
            raise ValueError("down projection output must match the model size")

        weight = source.gate_proj.weight
        packed = cls(model_dim, hidden_dim).to(
            device=weight.device, dtype=weight.dtype
        )
        with torch.no_grad():
            packed.gate_up_proj.weight.copy_(
                torch.cat(
                    (source.gate_proj.weight, source.up_proj.weight), dim=0
                )
            )
            packed.down_proj.weight.copy_(source.down_proj.weight)
        packed.train(source.training)
        return packed

    def to_unpacked(self) -> GatedMLP:
        """Restore the original parameter layout without changing values."""

        weight = self.gate_up_proj.weight
        unpacked = GatedMLP(self.model_dim, self.hidden_dim).to(
            device=weight.device, dtype=weight.dtype
        )
        gate_weight, up_weight = weight.chunk(2, dim=0)
        with torch.no_grad():
            unpacked.gate_proj.weight.copy_(gate_weight)
            unpacked.up_proj.weight.copy_(up_weight)
            unpacked.down_proj.weight.copy_(self.down_proj.weight)
        unpacked.train(self.training)
        return unpacked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(torch.nn.functional.silu(gate) * up)


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
