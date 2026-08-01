from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from compilelab.workload import GatedMLP, PackedGatedMLP


class SelfAttention(nn.Module):
    """Causal multi-head self-attention with separate Q, K, and V weights."""

    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        if model_dim <= 0 or num_heads <= 0:
            raise ValueError("model_dim and num_heads must be positive")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = tensor.shape
        return tensor.view(batch, sequence, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
        )
        merged = (
            attended.transpose(1, 2).contiguous().view(batch, sequence, self.model_dim)
        )
        return self.out_proj(merged)


class PackedSelfAttention(nn.Module):
    """Causal self-attention with Q, K, and V packed into one projection."""

    def __init__(self, model_dim: int, num_heads: int) -> None:
        super().__init__()
        if model_dim <= 0 or num_heads <= 0:
            raise ValueError("model_dim and num_heads must be positive")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv_proj = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)

    @classmethod
    def from_unpacked(cls, source: SelfAttention) -> PackedSelfAttention:
        packed = cls(source.model_dim, source.num_heads).to(
            device=source.q_proj.weight.device,
            dtype=source.q_proj.weight.dtype,
        )
        with torch.no_grad():
            packed.qkv_proj.weight.copy_(
                torch.cat(
                    (
                        source.q_proj.weight,
                        source.k_proj.weight,
                        source.v_proj.weight,
                    ),
                    dim=0,
                )
            )
            packed.out_proj.weight.copy_(source.out_proj.weight)
        packed.train(source.training)
        return packed

    def to_unpacked(self) -> SelfAttention:
        unpacked = SelfAttention(self.model_dim, self.num_heads).to(
            device=self.qkv_proj.weight.device,
            dtype=self.qkv_proj.weight.dtype,
        )
        query, key, value = self.qkv_proj.weight.chunk(3, dim=0)
        with torch.no_grad():
            unpacked.q_proj.weight.copy_(query)
            unpacked.k_proj.weight.copy_(key)
            unpacked.v_proj.weight.copy_(value)
            unpacked.out_proj.weight.copy_(self.out_proj.weight)
        unpacked.train(self.training)
        return unpacked

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = tensor.shape
        return tensor.view(batch, sequence, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        query, key, value = self.qkv_proj(x).chunk(3, dim=-1)
        attended = F.scaled_dot_product_attention(
            self._split_heads(query),
            self._split_heads(key),
            self._split_heads(value),
            dropout_p=0.0,
            is_causal=True,
        )
        merged = (
            attended.transpose(1, 2).contiguous().view(batch, sequence, self.model_dim)
        )
        return self.out_proj(merged)


class TransformerBlock(nn.Module):
    """A minimal pre-norm transformer block with a gated feed-forward path."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        hidden_dim: int,
        *,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.rms_norm_eps = rms_norm_eps
        self.attention_norm = nn.RMSNorm(model_dim, eps=rms_norm_eps)
        self.attention = SelfAttention(model_dim, num_heads)
        self.mlp_norm = nn.RMSNorm(model_dim, eps=rms_norm_eps)
        self.mlp = GatedMLP(model_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x + self.attention(self.attention_norm(x))
        return hidden + self.mlp(self.mlp_norm(hidden))


class PackedTransformerBlock(nn.Module):
    """Transformer block with packed QKV and gated-MLP input projections."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        hidden_dim: int,
        *,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.rms_norm_eps = rms_norm_eps
        self.attention_norm = nn.RMSNorm(model_dim, eps=rms_norm_eps)
        self.attention = PackedSelfAttention(model_dim, num_heads)
        self.mlp_norm = nn.RMSNorm(model_dim, eps=rms_norm_eps)
        self.mlp = PackedGatedMLP(model_dim, hidden_dim)

    @classmethod
    def from_unpacked(cls, source: TransformerBlock) -> PackedTransformerBlock:
        packed = cls(
            source.model_dim,
            source.num_heads,
            source.hidden_dim,
            rms_norm_eps=source.rms_norm_eps,
        ).to(
            device=source.attention.q_proj.weight.device,
            dtype=source.attention.q_proj.weight.dtype,
        )
        packed.attention = PackedSelfAttention.from_unpacked(source.attention)
        packed.mlp = PackedGatedMLP.from_unpacked(source.mlp)
        with torch.no_grad():
            packed.attention_norm.weight.copy_(source.attention_norm.weight)
            packed.mlp_norm.weight.copy_(source.mlp_norm.weight)
        packed.train(source.training)
        return packed

    def to_unpacked(self) -> TransformerBlock:
        unpacked = TransformerBlock(
            self.model_dim,
            self.num_heads,
            self.hidden_dim,
            rms_norm_eps=self.rms_norm_eps,
        ).to(
            device=self.attention.qkv_proj.weight.device,
            dtype=self.attention.qkv_proj.weight.dtype,
        )
        unpacked.attention = self.attention.to_unpacked()
        unpacked.mlp = self.mlp.to_unpacked()
        with torch.no_grad():
            unpacked.attention_norm.weight.copy_(self.attention_norm.weight)
            unpacked.mlp_norm.weight.copy_(self.mlp_norm.weight)
        unpacked.train(self.training)
        return unpacked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x + self.attention(self.attention_norm(x))
        return hidden + self.mlp(self.mlp_norm(hidden))


class DecomposedRMSNorm(nn.Module):
    """RMSNorm expressed as primitive arithmetic for lowering comparison."""

    def __init__(self, model_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if model_dim <= 0 or eps <= 0:
            raise ValueError("model_dim and eps must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(model_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().square().mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return normalized * self.weight


class RMSNormResidual(nn.Module):
    """Focused workload for RMSNorm and residual-add code-generation analysis."""

    def __init__(
        self,
        model_dim: int,
        *,
        decomposed: bool,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        norm_type = DecomposedRMSNorm if decomposed else nn.RMSNorm
        self.norm = norm_type(model_dim, eps=eps)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return residual + self.norm(x)
