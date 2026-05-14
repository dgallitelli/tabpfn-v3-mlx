"""Cell embedding and target encoders."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from tabpfn_mlx.layers import TrainableOrthogonalEmbedding


class CellEmbedding(nn.Module):
    """Linear projection from grouped features to embedding dimension."""

    def __init__(self, in_features: int, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_features, embed_dim)

    def __call__(self, x_BRiCG: mx.array) -> mx.array:
        """Project grouped features to embeddings.

        Args:
            x_BRiCG: (B, Ri, C, G) grouped feature values.

        Returns:
            (B, Ri, C, E) cell embeddings.
        """
        return self.linear(x_BRiCG)


class TargetEncoder(nn.Module):
    """Target encoder for column-level or ICL-level embedding.

    Uses orthogonal embedding for multiclass, linear for regression.
    """

    def __init__(self, embed_dim: int, task_type: str, max_num_classes: int = 10):
        super().__init__()
        self.task_type = task_type
        if task_type == "multiclass":
            self.encoder = TrainableOrthogonalEmbedding(max_num_classes, embed_dim)
        else:
            self.encoder = nn.Linear(1, embed_dim)

    def __call__(self, y: mx.array) -> mx.array:
        """Encode targets.

        Args:
            y: (B, N) class indices (multiclass) or (B, N) values (regression).

        Returns:
            (B, N, E) target embeddings.
        """
        if self.task_type == "multiclass":
            return self.encoder(y)
        return self.encoder(y[..., None])
