"""Column Aggregator: cross-feature interaction with CLS readout."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from tabpfn_mlx.attention import Attention, scaled_dot_product_attention
from tabpfn_mlx.layers import MLP, RMSNorm, RotaryEmbedding


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block used in ColumnAggregator."""

    def __init__(self, emsize: int, nhead: int, dim_feedforward: int):
        super().__init__()
        assert emsize % nhead == 0
        self.attention = Attention(
            embedding_size=emsize,
            num_heads=nhead,
            head_dim=emsize // nhead,
        )
        self.layernorm = RMSNorm(emsize)
        self.layernorm_mlp = RMSNorm(emsize)
        self.mlp = MLP(emsize, dim_feedforward)

    def __call__(
        self,
        x_BRCE: mx.array,
        rope: RotaryEmbedding | None = None,
    ) -> mx.array:
        """Self-attention over features (dim 2) per row.

        Args:
            x_BRCE: (B, R, C, E) — features are the sequence dimension.
        """
        B, R, C, E = x_BRCE.shape
        # Fold B*R into batch for per-row feature attention
        x_flat = x_BRCE.reshape(B * R, C, E)
        attn_out = self.attention(self.layernorm(x_flat), rope=rope)
        x_flat = x_flat + attn_out
        x_flat = x_flat.reshape(B, R, C, E)

        # MLP
        mlp_out = self.mlp(self.layernorm_mlp(x_flat))
        return x_flat + mlp_out

    def forward_cross(
        self,
        query_BRQE: mx.array,
        context_BRCE: mx.array,
        rope: RotaryEmbedding | None = None,
    ) -> mx.array:
        """Cross-attention: query attends to context (CLS readout)."""
        B, R, Q, E = query_BRQE.shape
        _, _, V, _ = context_BRCE.shape

        norm_q = self.layernorm(query_BRQE)
        q_flat = norm_q.reshape(B * R, Q, E)
        c_flat = self.layernorm(context_BRCE).reshape(B * R, V, E)

        head_dim = self.attention.head_dim
        num_heads = self.attention.num_heads

        q_proj = self.attention.q_projection(q_flat).reshape(B * R, Q, num_heads, head_dim)
        k_flat = self.attention.k_projection(c_flat).reshape(B * R, V, num_heads, head_dim)
        v_flat = self.attention.v_projection(c_flat).reshape(B * R, V, num_heads, head_dim)

        if rope is not None:
            q_proj = rope.rotate_queries_or_keys(
                q_proj.transpose(0, 2, 1, 3)
            ).transpose(0, 2, 1, 3)
            k_flat = rope.rotate_queries_or_keys(
                k_flat.transpose(0, 2, 1, 3)
            ).transpose(0, 2, 1, 3)

        attn_out = scaled_dot_product_attention(q_proj, k_flat, v_flat)
        attn_out = attn_out.reshape(B * R, Q, head_dim * num_heads)
        attn_out = self.attention.out_projection(attn_out).reshape(B, R, Q, E)

        x_out = query_BRQE + attn_out
        mlp_out = self.mlp(self.layernorm_mlp(x_out))
        return x_out + mlp_out


class ColumnAggregator(nn.Module):
    """Cross-feature interaction that aggregates column information via CLS tokens.

    CLS tokens are prepended, pass through transformer blocks,
    and the last block performs CLS-only readout (q=CLS, k/v=all).
    """

    def __init__(
        self,
        emsize: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        num_cls_tokens: int,
        use_rope: bool = True,
        rope_base: float = 100_000,
    ):
        super().__init__()
        self.embed_dim = emsize
        self.num_cls_tokens = num_cls_tokens

        self.blocks = [
            TransformerBlock(emsize=emsize, nhead=nhead, dim_feedforward=dim_feedforward)
            for _ in range(num_layers)
        ]
        self.rope = (
            RotaryEmbedding(dim=emsize // nhead, theta=rope_base)
            if use_rope
            else None
        )
        self.cls_tokens = mx.zeros((num_cls_tokens, emsize))
        self.out_ln = RMSNorm(emsize)

    def __call__(self, x_BRiCE: mx.array) -> mx.array:
        """Transform feature embeddings into per-row CLS representations.

        Args:
            x_BRiCE: (B, Ri, C, E)

        Returns:
            (B, Ri, num_cls_tokens, E)
        """
        B, Ri, C, E = x_BRiCE.shape

        # Expand CLS tokens: (B, Ri, Cl, E)
        cls = mx.broadcast_to(
            self.cls_tokens[None, None, :, :],
            (B, Ri, self.num_cls_tokens, E),
        )
        # Prepend CLS tokens along column axis: (B, Ri, Cl+C, E)
        x = mx.concatenate([cls, x_BRiCE], axis=2)

        # All blocks except last: self-attention
        for block in self.blocks[:-1]:
            x = block(x, rope=self.rope)

        # Last block: CLS readout (cross-attention)
        cls_part = x[:, :, :self.num_cls_tokens, :]
        cls_out = self.blocks[-1].forward_cross(cls_part, x, self.rope)

        return self.out_ln(cls_out)
