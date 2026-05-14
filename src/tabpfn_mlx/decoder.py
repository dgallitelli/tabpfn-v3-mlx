"""Decoder modules: ManyClassDecoder (attention retrieval) + regression head."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from tabpfn_mlx.attention import SoftmaxScalingMLP, scaled_dot_product_attention


class ManyClassDecoder(nn.Module):
    """Attention-based retrieval decoder for many-class classification.

    Computes weighted (by attention score) average over one-hot encoded
    train targets, then takes the log to obtain logits.
    """

    def __init__(
        self,
        max_num_classes: int,
        input_size: int,
        head_dim: int = 64,
        num_heads: int = 6,
        softmax_scaling_layer: SoftmaxScalingMLP | None = None,
    ):
        super().__init__()
        self.max_num_classes = max_num_classes
        self.input_size = input_size
        self.attention_size = head_dim * num_heads
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.softmax_scaling_layer = softmax_scaling_layer

        self.q_projection = nn.Linear(input_size, self.attention_size)
        self.k_projection = nn.Linear(input_size, self.attention_size)

    def __call__(
        self,
        train_embeddings: mx.array,
        test_embeddings: mx.array,
        targets: mx.array,
    ) -> mx.array:
        """Forward pass.

        Args:
            train_embeddings: (B, N, E) train row embeddings
            test_embeddings: (B, M, E) test row embeddings
            targets: (B, N) class indices

        Returns:
            (M, B, T) logits
        """
        B, M, _ = test_embeddings.shape
        N = train_embeddings.shape[1]

        q_BME = self.q_projection(test_embeddings)
        k_BNE = self.k_projection(train_embeddings)

        if M == 0:
            return mx.zeros((0, B, self.max_num_classes))

        # One-hot encode targets: (B, N, T)
        targets_int = targets.astype(mx.int32)
        one_hot_targets = mx.zeros((B, N, self.max_num_classes))
        # Vectorized one-hot via indexing
        idx = mx.expand_dims(targets_int, axis=-1)  # (B, N, 1)
        one_hot_targets = mx.put_along_axis(one_hot_targets, idx, mx.array(1.0), axis=-1)

        # Reshape for multi-head attention
        q_BMHD = q_BME.reshape(B, M, self.num_heads, self.head_dim)
        k_BNHD = k_BNE.reshape(B, N, self.num_heads, self.head_dim)

        # Expand one-hot to all heads: (B, N, H, T)
        one_hot_BNHT = mx.broadcast_to(
            one_hot_targets[:, :, None, :],
            (B, N, self.num_heads, self.max_num_classes),
        )

        # Chunked class attention
        test_output_BMHT = _chunked_class_attention(
            q_BMHD, k_BNHD, one_hot_BNHT,
            softmax_scaling_layer=self.softmax_scaling_layer,
        )
        # Average over heads
        test_output_BMT = test_output_BMHT.mean(axis=2)

        # Transpose to (M, B, T) and convert to logits
        test_output_MBT = test_output_BMT.transpose(1, 0, 2)
        return mx.log(mx.clip(test_output_MBT, 1e-5, None) + 3e-5)


def _chunked_class_attention(
    q_BSHD: mx.array,
    k_BJHD: mx.array,
    v_BJHT: mx.array,
    softmax_scaling_layer: SoftmaxScalingMLP | None = None,
) -> mx.array:
    """Attention where value dimension T may exceed head_dim D.

    Splits V into head_dim-sized chunks, folds chunk index into batch,
    runs a single SDPA call, then unfolds.
    """
    B, S, H, D = q_BSHD.shape
    T = v_BJHT.shape[-1]
    num_chunks = math.ceil(T / D)

    # Pad V to multiple of D
    pad = num_chunks * D - T
    if pad > 0:
        v_BJHT = mx.pad(v_BJHT, [(0, 0), (0, 0), (0, 0), (0, pad)])

    J = v_BJHT.shape[1]

    # Fold chunk index into batch: (B, J, H, num_chunks, D) → (B*num_chunks, J, H, D)
    v_folded = v_BJHT.reshape(B, J, H, num_chunks, D)
    v_folded = v_folded.transpose(0, 3, 1, 2, 4).reshape(B * num_chunks, J, H, D)

    # Expand Q and K across chunks
    q_folded = mx.broadcast_to(
        q_BSHD[:, None, :, :, :], (B, num_chunks, S, H, D)
    ).reshape(B * num_chunks, S, H, D)

    k_folded = mx.broadcast_to(
        k_BJHD[:, None, :, :, :], (B, num_chunks, J, H, D)
    ).reshape(B * num_chunks, J, H, D)

    # Apply softmax scaling if provided
    if softmax_scaling_layer is not None:
        q_folded = softmax_scaling_layer(q_folded, J)

    # Single SDPA call
    out_folded = scaled_dot_product_attention(q_folded, k_folded, v_folded)

    # Unfold: (B*K, S, H, D) → (B, S, H, T)
    out = out_folded.reshape(B, num_chunks, S, H, D)
    out = out.transpose(0, 2, 3, 1, 4).reshape(B, S, H, num_chunks * D)
    return out[..., :T]
