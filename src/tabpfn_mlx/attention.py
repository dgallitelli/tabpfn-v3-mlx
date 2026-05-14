"""Attention variants: Attention, CrossAttention, ICLAttention, SoftmaxScalingMLP."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from tabpfn_mlx.layers import RotaryEmbedding, MLP, RMSNorm


def scaled_dot_product_attention(
    q: mx.array, k: mx.array, v: mx.array, scale: float | None = None
) -> mx.array:
    """Scaled dot-product attention.

    Args:
        q: (B, S, H, D) queries
        k: (B, J, Hkv, D) keys
        v: (B, J, Hkv, D) values
        scale: Optional scale factor (default: 1/sqrt(D))

    Returns:
        (B, S, H, D) attention output
    """
    D = q.shape[-1]
    if scale is None:
        scale = D ** -0.5

    num_q_heads = q.shape[2]
    num_kv_heads = k.shape[2]

    # Transpose to (B, H, S, D) for matmul
    q = q.transpose(0, 2, 1, 3)  # (B, H, S, D)
    k = k.transpose(0, 2, 1, 3)  # (B, Hkv, J, D)
    v = v.transpose(0, 2, 1, 3)  # (B, Hkv, J, D)

    # GQA: repeat KV heads to match Q heads
    if num_kv_heads < num_q_heads:
        repeat = num_q_heads // num_kv_heads
        k = mx.repeat(k, repeat, axis=1)
        v = mx.repeat(v, repeat, axis=1)

    # Compute attention
    scores = (q * scale) @ k.transpose(0, 1, 3, 2)  # (B, H, S, J)
    weights = mx.softmax(scores, axis=-1)
    out = weights @ v  # (B, H, S, D)

    # Transpose back to (B, S, H, D)
    return out.transpose(0, 2, 1, 3)


class SoftmaxScalingMLP(nn.Module):
    """Query-aware attention scaling using learned MLPs.

    Applies: q_scaled = q * base_mlp(log(n)) * (1 + tanh(query_mlp(q)))
    """

    def __init__(self, num_heads: int, head_dim: int, n_hidden: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        base_out_dim = num_heads * head_dim
        query_out_dim = head_dim

        self.base_linear1 = nn.Linear(1, n_hidden)
        self.base_linear2 = nn.Linear(n_hidden, base_out_dim)
        self.query_linear1 = nn.Linear(head_dim, n_hidden)
        self.query_linear2 = nn.Linear(n_hidden, query_out_dim)

    def __call__(self, q_BSHD: mx.array, n: int) -> mx.array:
        """Apply scaling to queries.

        Args:
            q_BSHD: (B, S, H, D) query tensor
            n: number of key elements for log-n scaling
        """
        logn = mx.log(mx.maximum(mx.array(float(n)), mx.array(1.0)))
        logn_input = logn.reshape(1, 1)
        base_scales = self.base_linear2(nn.gelu(self.base_linear1(logn_input)))
        base_scales = base_scales.reshape(1, 1, self.num_heads, self.head_dim)
        modulation = 1 + mx.tanh(self.query_linear2(nn.gelu(self.query_linear1(q_BSHD))))
        return q_BSHD * base_scales * modulation


class Attention(nn.Module):
    """Multi-head self-attention with optional RoPE."""

    def __init__(self, embedding_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_projection = nn.Linear(embedding_size, head_dim * num_heads, bias=False)
        self.k_projection = nn.Linear(embedding_size, head_dim * num_heads, bias=False)
        self.v_projection = nn.Linear(embedding_size, head_dim * num_heads, bias=False)
        self.out_projection = nn.Linear(head_dim * num_heads, embedding_size, bias=False)

    def __call__(
        self,
        x_BSE: mx.array,
        rope: RotaryEmbedding | None = None,
    ) -> mx.array:
        B, S, _ = x_BSE.shape
        q = self.q_projection(x_BSE).reshape(B, S, self.num_heads, self.head_dim)
        k = self.k_projection(x_BSE).reshape(B, S, self.num_heads, self.head_dim)
        v = self.v_projection(x_BSE).reshape(B, S, self.num_heads, self.head_dim)

        if rope is not None:
            # Transpose to (B, H, S, D) for RoPE, then back
            q = rope.rotate_queries_or_keys(q.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
            k = rope.rotate_queries_or_keys(k.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)

        out = scaled_dot_product_attention(q, k, v)
        return self.out_projection(out.reshape(B, S, self.head_dim * self.num_heads))


class CrossAttention(nn.Module):
    """Multi-head cross-attention (query attends to key/value sequence)."""

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        softmax_scaling_layer: SoftmaxScalingMLP | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.softmax_scaling_layer = softmax_scaling_layer

        self.q_projection = nn.Linear(embedding_size, head_dim * num_heads, bias=False)
        self.k_projection = nn.Linear(embedding_size, head_dim * num_heads, bias=False)
        self.v_projection = nn.Linear(embedding_size, head_dim * num_heads, bias=False)
        self.out_projection = nn.Linear(head_dim * num_heads, embedding_size, bias=False)

    def __call__(
        self,
        x_for_query_BQE: mx.array,
        x_for_key_and_value_BVE: mx.array,
    ) -> mx.array:
        B, Q, _ = x_for_query_BQE.shape
        V = x_for_key_and_value_BVE.shape[1]

        q = self.q_projection(x_for_query_BQE).reshape(B, Q, self.num_heads, self.head_dim)
        k = self.k_projection(x_for_key_and_value_BVE).reshape(B, V, self.num_heads, self.head_dim)
        v = self.v_projection(x_for_key_and_value_BVE).reshape(B, V, self.num_heads, self.head_dim)

        if self.softmax_scaling_layer is not None:
            q = self.softmax_scaling_layer(q, V)

        out = scaled_dot_product_attention(q, k, v)
        return self.out_projection(out.reshape(B, Q, self.head_dim * self.num_heads))


class ICLAttention(nn.Module):
    """ICL attention: all rows attend to train-only keys/values.

    Test rows cannot attend to each other — only to train rows.
    Supports GQA with different KV head counts for train vs test.
    """

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_dim: int,
        softmax_scaling_layer: SoftmaxScalingMLP | None = None,
        num_kv_heads: int | None = None,
        num_kv_heads_test: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.softmax_scaling_layer = softmax_scaling_layer
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_heads_test = num_kv_heads_test

        self.q_projection = nn.Linear(embedding_size, head_dim * num_heads, bias=False)
        self.out_projection = nn.Linear(head_dim * num_heads, embedding_size, bias=False)

        kv_dim = self.num_kv_heads * head_dim
        self.k_projection = nn.Linear(embedding_size, kv_dim, bias=False)
        self.v_projection = nn.Linear(embedding_size, kv_dim, bias=False)

    def __call__(
        self,
        x_BRE: mx.array,
        single_eval_pos: int,
        *,
        cached_kv: tuple[mx.array, mx.array] | None = None,
        return_kv: bool = False,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """ICL attention forward pass.

        Args:
            x_BRE: (B, R, E) all rows, or test-only when cached_kv is provided.
            single_eval_pos: Number of training rows.
            cached_kv: Pre-computed (K, V) from previous pass.
            return_kv: If True, also return (K, V) for caching.

        Returns:
            (output, kv_entry) where kv_entry is None unless return_kv is True.
        """
        B, R, _ = x_BRE.shape
        q = self.q_projection(x_BRE).reshape(B, R, self.num_heads, self.head_dim)

        if self.softmax_scaling_layer is not None:
            N = cached_kv[0].shape[1] if cached_kv is not None else (
                R if single_eval_pos is None else single_eval_pos
            )
            q = self.softmax_scaling_layer(q, N)

        if cached_kv is not None:
            k, v = cached_kv
            out = scaled_dot_product_attention(q, k, v)
        else:
            N = R if single_eval_pos is None else single_eval_pos
            x_train = x_BRE[:, :N]
            k = self.k_projection(x_train).reshape(B, N, self.num_kv_heads, self.head_dim)
            v = self.v_projection(x_train).reshape(B, N, self.num_kv_heads, self.head_dim)

            if (
                self.num_kv_heads_test is not None
                and single_eval_pos is not None
                and N < R
            ):
                # Train rows: full KV heads
                out_train = scaled_dot_product_attention(q[:, :N], k, v)
                # Test rows: fewer KV heads
                nh_test = self.num_kv_heads_test
                out_test = scaled_dot_product_attention(
                    q[:, N:], k[:, :, :nh_test], v[:, :, :nh_test]
                )
                out = mx.concatenate([out_train, out_test], axis=1)
            else:
                out = scaled_dot_product_attention(q, k, v)

        result = self.out_projection(out.reshape(B, R, self.head_dim * self.num_heads))

        kv_entry = None
        if return_kv:
            k_cache, v_cache = k, v
            if self.num_kv_heads_test is not None:
                nh_test = self.num_kv_heads_test
                k_cache = k_cache[:, :, :nh_test]
                v_cache = v_cache[:, :, :nh_test]
            kv_entry = (mx.stop_gradient(k_cache), mx.stop_gradient(v_cache))

        return result, kv_entry
