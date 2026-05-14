"""TabPFN v3 configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TabPFNV3Config:
    """Configuration for the TabPFN v3 architecture."""

    # Distribution embedder (per-column induced self-attention)
    embed_dim: int = 128
    dist_embed_num_blocks: int = 3
    dist_embed_num_heads: int = 8
    dist_embed_num_inducing_points: int = 128
    feature_group_size: int = 3

    # Feature aggregation (cross-feature interaction via CLS tokens)
    feat_agg_num_blocks: int = 3
    feat_agg_num_heads: int = 8
    feat_agg_num_cls_tokens: int = 4
    feat_agg_rope_base: float = 100_000

    use_rope: bool = True

    # ICL transformer
    nlayers: int = 24
    icl_num_heads: int = 8
    icl_num_kv_heads: int | None = None
    icl_num_kv_heads_test: int | None = None

    # Output decoder
    decoder_head_dim: int = 64
    decoder_num_heads: int = 6
    decoder_use_softmax_scaling: bool = False

    # Shared
    ff_factor: int = 2
    softmax_scaling_mlp_hidden_dim: int = 64

    # Norm
    layernorm_elementwise_affine: bool = True

    use_nan_indicators: bool = True

    # Task / output
    max_num_classes: int = 10
    num_buckets: int = 5000

    @property
    def icl_emsize(self) -> int:
        return self.embed_dim * self.feat_agg_num_cls_tokens

    def __post_init__(self) -> None:
        if self.icl_num_kv_heads is not None and (
            self.icl_num_heads % self.icl_num_kv_heads != 0
        ):
            raise ValueError(
                f"icl_num_heads ({self.icl_num_heads}) must be divisible by "
                f"icl_num_kv_heads ({self.icl_num_kv_heads})"
            )
        if self.icl_num_kv_heads_test is not None:
            if self.icl_num_heads % self.icl_num_kv_heads_test != 0:
                raise ValueError(
                    f"icl_num_heads ({self.icl_num_heads}) must be divisible by "
                    f"icl_num_kv_heads_test ({self.icl_num_kv_heads_test})"
                )
