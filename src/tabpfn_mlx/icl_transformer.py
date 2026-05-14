"""ICL Transformer blocks (the 24-layer core of TabPFN v3)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from tabpfn_mlx.attention import ICLAttention, SoftmaxScalingMLP
from tabpfn_mlx.layers import MLP, RMSNorm


class ICLTransformerBlock(nn.Module):
    """ICL transformer block with train-only keys and optional softmax scaling."""

    def __init__(
        self,
        emsize: int,
        nhead: int,
        dim_feedforward: int,
        softmax_scaling_layer: SoftmaxScalingMLP | None = None,
        num_kv_heads: int | None = None,
        num_kv_heads_test: int | None = None,
    ):
        super().__init__()
        assert emsize % nhead == 0
        self.icl_attention = ICLAttention(
            embedding_size=emsize,
            num_heads=nhead,
            head_dim=emsize // nhead,
            softmax_scaling_layer=softmax_scaling_layer,
            num_kv_heads=num_kv_heads,
            num_kv_heads_test=num_kv_heads_test,
        )
        self.layernorm = RMSNorm(emsize)
        self.layernorm_mlp = RMSNorm(emsize)
        self.mlp = MLP(emsize, dim_feedforward)

    def __call__(
        self,
        x_BRE: mx.array,
        single_eval_pos: int,
        *,
        cached_kv: tuple[mx.array, mx.array] | None = None,
        return_kv: bool = False,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass with optional KV cache support.

        Args:
            x_BRE: (B, R, E) all rows, or test-only when cached_kv is set.
            single_eval_pos: Number of training rows.
            cached_kv: Pre-computed (K, V) for this layer.
            return_kv: If True, also return the (K, V) cache entry.

        Returns:
            (output, kv_entry) where kv_entry is None unless return_kv is True.
        """
        attn_out, kv_entry = self.icl_attention(
            self.layernorm(x_BRE),
            single_eval_pos=single_eval_pos,
            cached_kv=cached_kv,
            return_kv=return_kv,
        )
        x_BRE = x_BRE + attn_out

        mlp_out = self.mlp(self.layernorm_mlp(x_BRE))
        x_BRE = x_BRE + mlp_out

        return x_BRE, kv_entry
