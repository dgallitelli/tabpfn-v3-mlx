"""Foundation layers: RMSNorm, MLP, RotaryEmbedding, TrainableOrthogonalEmbedding."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms).astype(dtype) * self.weight


class MLP(nn.Module):
    """Two-layer GELU feed-forward network (no bias, zero-init output)."""

    def __init__(self, emsize: int, dim_feedforward: int):
        super().__init__()
        self.linear1 = nn.Linear(emsize, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, emsize, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(nn.gelu(self.linear1(x)))


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (split-half, non-interleaved)."""

    def __init__(self, dim: int, theta: float = 10_000.0):
        super().__init__()
        assert dim % 2 == 0, f"RoPE head dim must be even, got {dim}"
        inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        self.inv_freq = inv_freq

    def rotate_queries_or_keys(self, t_BHSD: mx.array) -> mx.array:
        """Apply RoPE to tensor of shape (B, H, S, D) or (B, S, H, D).

        Expects input transposed to (B, H, S, D) convention.
        """
        return apply_rope(t_BHSD, self.inv_freq)


def apply_rope(t: mx.array, inv_freq: mx.array) -> mx.array:
    """Apply rotary positional embeddings (split-half pattern).

    Args:
        t: (..., S, D) where D is even.
        inv_freq: (D // 2,) inverse frequencies.
    """
    dtype = t.dtype
    seq_len = t.shape[-2]
    positions = mx.arange(seq_len).astype(mx.float32)
    freqs = positions[:, None] * inv_freq[None, :]  # (S, D/2)
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    # Split-half: concat cos/sin to full D
    cos = mx.concatenate([cos, cos], axis=-1)  # (S, D)
    sin = mx.concatenate([sin, sin], axis=-1)  # (S, D)
    half = t.shape[-1] // 2
    t_rotated = mx.concatenate([-t[..., half:], t[..., :half]], axis=-1)
    return (t.astype(mx.float32) * cos + t_rotated.astype(mx.float32) * sin).astype(dtype)


class TrainableOrthogonalEmbedding(nn.Module):
    """Embedding with orthogonal initialization for class labels."""

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.embedding(x.astype(mx.int32))
