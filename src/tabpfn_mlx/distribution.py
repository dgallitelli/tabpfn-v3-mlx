"""Feature Distribution Embedder (InducedSelfAttention blocks)."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from tabpfn_mlx.attention import CrossAttention, SoftmaxScalingMLP, scaled_dot_product_attention
from tabpfn_mlx.layers import MLP, RMSNorm


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with pre-norm and MLP."""

    def __init__(
        self,
        emsize: int,
        nhead: int,
        dim_feedforward: int,
        softmax_scaling_layer: SoftmaxScalingMLP | None = None,
    ):
        super().__init__()
        assert emsize % nhead == 0
        self.attn = CrossAttention(
            embedding_size=emsize,
            num_heads=nhead,
            head_dim=emsize // nhead,
            softmax_scaling_layer=softmax_scaling_layer,
        )
        self.mlp = MLP(emsize, dim_feedforward)
        self.layernorm_q = RMSNorm(emsize)
        self.layernorm_kv = RMSNorm(emsize)
        self.layernorm2 = RMSNorm(emsize)

    def __call__(
        self,
        x_BQE: mx.array,
        context_BVE: mx.array,
    ) -> mx.array:
        attn_out = self.attn(self.layernorm_q(x_BQE), self.layernorm_kv(context_BVE))
        x_BQE = x_BQE + attn_out
        mlp_out = self.mlp(self.layernorm2(x_BQE))
        return x_BQE + mlp_out


class InducedSelfAttentionBlock(nn.Module):
    """Induced self-attention (SetTransformer-style) for O(n) attention.

    Two-stage:
    1. Learnable inducing points attend to train rows.
    2. All rows attend to the inducing-point hidden states.
    """

    def __init__(
        self,
        emsize: int,
        nhead: int,
        num_inducing_points: int,
        dim_feedforward: int,
        softmax_scaling_layer: SoftmaxScalingMLP | None = None,
    ):
        super().__init__()
        self.cross_attn_block1 = CrossAttentionBlock(
            emsize=emsize,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            softmax_scaling_layer=softmax_scaling_layer,
        )
        self.cross_attn_block2 = CrossAttentionBlock(
            emsize=emsize,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.num_inducing_points = num_inducing_points
        self.inducing_vectors = mx.zeros((num_inducing_points, emsize))

    def __call__(
        self,
        x_BRCE: mx.array,
        single_eval_pos: int | None = None,
        *,
        cached_hidden: mx.array | None = None,
        return_hidden: bool = False,
    ) -> tuple[mx.array, mx.array | None]:
        """Forward with optional inducing hidden-state caching.

        Returns:
            (output, hidden) where hidden is None unless return_hidden is True.
        """
        B, R, C, E = x_BRCE.shape
        # Transpose to per-column processing: (B, C, R, E) → (B*C, R, E)
        x_BCRE = x_BRCE.transpose(0, 2, 1, 3)
        x_BcRE = x_BCRE.reshape(B * C, R, E)

        if cached_hidden is not None:
            hidden = cached_hidden
        else:
            Bc = B * C
            N = R if single_eval_pos is None else single_eval_pos
            ind = mx.broadcast_to(
                self.inducing_vectors[None, :, :], (Bc, self.num_inducing_points, E)
            )
            hidden = self.cross_attn_block1(ind, x_BcRE[:, :N])

        out_BcRE = self.cross_attn_block2(x_BcRE, hidden)

        out_BCRE = out_BcRE.reshape(B, C, R, E)
        result = out_BCRE.transpose(0, 2, 1, 3)

        if return_hidden:
            return result, mx.stop_gradient(hidden)
        return result, None


class FeatureDistributionEmbedder(nn.Module):
    """Stack of InducedSelfAttentionBlock layers applied per column."""

    def __init__(
        self,
        emsize: int,
        nhead: int,
        num_inducing_points: int,
        dim_feedforward: int,
        num_layers: int,
        softmax_scaling_mlp_hidden_dim: int = 64,
    ):
        super().__init__()
        self.layers = [
            InducedSelfAttentionBlock(
                emsize=emsize,
                nhead=nhead,
                num_inducing_points=num_inducing_points,
                dim_feedforward=dim_feedforward,
                softmax_scaling_layer=SoftmaxScalingMLP(
                    num_heads=nhead,
                    head_dim=emsize // nhead,
                    n_hidden=softmax_scaling_mlp_hidden_dim,
                ),
            )
            for _ in range(num_layers)
        ]

    def __call__(
        self,
        x_BRiCE: mx.array,
        num_train_rows: int | None = None,
        *,
        cached_hidden: list[mx.array] | None = None,
        return_hidden: bool = False,
    ) -> tuple[mx.array, list[mx.array] | None]:
        """Forward pass through all induced self-attention blocks."""
        hidden_states: list[mx.array] | None = [] if return_hidden else None

        for i, layer in enumerate(self.layers):
            layer_cached = cached_hidden[i] if cached_hidden is not None else None
            x_BRiCE, h = layer(
                x_BRiCE,
                single_eval_pos=num_train_rows,
                cached_hidden=layer_cached,
                return_hidden=return_hidden,
            )
            if hidden_states is not None and h is not None:
                hidden_states.append(h)

        return x_BRiCE, hidden_states
