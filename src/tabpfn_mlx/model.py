"""TabPFN v3 model: full architecture orchestrating all stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from tabpfn_mlx.aggregation import ColumnAggregator
from tabpfn_mlx.attention import SoftmaxScalingMLP
from tabpfn_mlx.config import TabPFNV3Config
from tabpfn_mlx.decoder import ManyClassDecoder
from tabpfn_mlx.distribution import FeatureDistributionEmbedder
from tabpfn_mlx.embedding import CellEmbedding, TargetEncoder
from tabpfn_mlx.icl_transformer import ICLTransformerBlock
from tabpfn_mlx.layers import RMSNorm
from tabpfn_mlx.preprocessing import (
    StandardScaler,
    generate_nan_and_inf_indicator,
    group_features,
    impute_nan_and_inf_with_mean,
)


@dataclass
class TabPFNV3Cache:
    """KV cache for efficient inference."""

    icl_cache: dict[int, tuple[mx.array, mx.array]] = field(default_factory=dict)
    train_embeddings: mx.array | None = None
    train_shape: tuple[int, int] = (0, 0)
    scaler_cache: dict[str, mx.array] | None = None
    inducing_hidden: list[mx.array] | None = None

    def is_empty(self) -> bool:
        return len(self.icl_cache) == 0


class TabPFNV3(nn.Module):
    """TabPFN v3 architecture — full multi-stage in-context learning model.

    Pipeline:
    1. Preprocessing: standard scaling + NaN encoding
    2. Feature grouping: circular shifts
    3. Cell embedding: feature_group_size values → embed_dim
    4. Target-aware column embedding
    5. Feature distribution embedder (InducedSelfAttention)
    6. Column aggregator (CLS token readout)
    7. ICL transformer (train-keys-only attention)
    8. Decoder (attention retrieval for multiclass, MLP for regression)
    """

    def __init__(self, config: TabPFNV3Config, task_type: str = "multiclass"):
        super().__init__()
        self.config = config
        self.task_type = task_type
        self.feature_group_size = config.feature_group_size
        self.use_nan_indicators = config.use_nan_indicators
        self.icl_emsize = config.icl_emsize

        # Determine cell embedding input size
        in_features = config.feature_group_size
        if self.use_nan_indicators:
            in_features *= 2

        # Stage 1: Cell embedding
        self.x_embed = CellEmbedding(in_features, config.embed_dim)

        # Stage 1: Target-aware column embedding
        self.col_y_encoder = TargetEncoder(
            config.embed_dim, task_type, config.max_num_classes
        )

        # Stage 2a: Distribution embedder
        self.feature_distribution_embedder = FeatureDistributionEmbedder(
            emsize=config.embed_dim,
            nhead=config.dist_embed_num_heads,
            num_inducing_points=config.dist_embed_num_inducing_points,
            dim_feedforward=config.embed_dim * config.ff_factor,
            num_layers=config.dist_embed_num_blocks,
            softmax_scaling_mlp_hidden_dim=config.softmax_scaling_mlp_hidden_dim,
        )

        # Stage 2b: Column aggregator
        self.column_aggregator = ColumnAggregator(
            emsize=config.embed_dim,
            nhead=config.feat_agg_num_heads,
            num_layers=config.feat_agg_num_blocks,
            num_cls_tokens=config.feat_agg_num_cls_tokens,
            dim_feedforward=config.embed_dim * config.ff_factor,
            use_rope=config.use_rope,
            rope_base=config.feat_agg_rope_base,
        )

        # Stage 3: ICL target encoder
        self.icl_y_encoder = TargetEncoder(
            self.icl_emsize, task_type, config.max_num_classes
        )

        # Stage 3: ICL transformer blocks
        self.icl_blocks = [
            ICLTransformerBlock(
                emsize=self.icl_emsize,
                nhead=config.icl_num_heads,
                dim_feedforward=self.icl_emsize * config.ff_factor,
                softmax_scaling_layer=SoftmaxScalingMLP(
                    num_heads=config.icl_num_heads,
                    head_dim=self.icl_emsize // config.icl_num_heads,
                    n_hidden=config.softmax_scaling_mlp_hidden_dim,
                ),
                num_kv_heads=config.icl_num_kv_heads,
                num_kv_heads_test=config.icl_num_kv_heads_test,
            )
            for _ in range(config.nlayers)
        ]

        # Stage 4: Output norm + decoder
        self.output_norm = RMSNorm(self.icl_emsize)

        if task_type == "multiclass":
            decoder_scaling = (
                SoftmaxScalingMLP(
                    num_heads=config.decoder_num_heads,
                    head_dim=config.decoder_head_dim,
                    n_hidden=config.softmax_scaling_mlp_hidden_dim,
                )
                if config.decoder_use_softmax_scaling
                else None
            )
            self.many_class_decoder = ManyClassDecoder(
                max_num_classes=config.max_num_classes,
                input_size=self.icl_emsize,
                head_dim=config.decoder_head_dim,
                num_heads=config.decoder_num_heads,
                softmax_scaling_layer=decoder_scaling,
            )
        else:
            n_out = config.num_buckets
            self.output_projection_linear1 = nn.Linear(
                self.icl_emsize, self.icl_emsize * config.ff_factor
            )
            self.output_projection_linear2 = nn.Linear(
                self.icl_emsize * config.ff_factor, n_out
            )

        self.standard_scaler = StandardScaler()

    def __call__(
        self,
        x: mx.array,
        y: mx.array,
        *,
        kv_cache: TabPFNV3Cache | None = None,
        return_kv_cache: bool = False,
        x_is_test_only: bool = False,
    ) -> mx.array | tuple[mx.array, TabPFNV3Cache | None]:
        """Main forward pass.

        Args:
            x: (Ri, B, C_raw) input features (rows-first convention).
            y: (N,) or (N, B) target labels for train rows.
            kv_cache: Pre-computed cache for inference.
            return_kv_cache: If True, also return the built cache.
            x_is_test_only: If True with kv_cache, x contains only test rows.

        Returns:
            logits (M, B, T) or (logits, cache) if return_kv_cache.
        """
        if y.ndim == 0:
            y = y.reshape(0)

        x_RiBC = x
        B = x_RiBC.shape[1]
        num_train = y.shape[0]

        # --- Stages 0-2 ---
        x_BRiClE, inducing_hidden = self._stages_0_to_2(
            x_RiBC, y, num_train, B,
            kv_cache=kv_cache,
            x_is_test_only=x_is_test_only,
            return_inducing_hidden=return_kv_cache,
        )

        # --- Stage 3: ICL transformer ---
        # Flatten CLS tokens: (B, Ri, Cl, E) → (B, Ri, Cl*E) = (B, Ri, D)
        x_BRiD = x_BRiClE.reshape(x_BRiClE.shape[0], x_BRiClE.shape[1], -1)

        icl_cache_out: dict[int, tuple[mx.array, mx.array]] = {}

        if kv_cache is not None and not kv_cache.is_empty():
            for layer_idx, block in enumerate(self.icl_blocks):
                x_BRiD, _ = block(
                    x_BRiD, 0,
                    cached_kv=kv_cache.icl_cache[layer_idx],
                )
        else:
            if num_train > 0:
                y_BN = self._prepare_y(y, num_train, B)
                y_icl_emb = self.icl_y_encoder(y_BN)
                # Add ICL target embedding to train rows
                train_part = x_BRiD[:, :num_train] + y_icl_emb
                test_part = x_BRiD[:, num_train:]
                x_BRiD = mx.concatenate([train_part, test_part], axis=1)

            if return_kv_cache:
                for layer_idx, block in enumerate(self.icl_blocks):
                    x_BRiD, kv_entry = block(
                        x_BRiD, num_train, return_kv=True
                    )
                    icl_cache_out[layer_idx] = kv_entry
            else:
                for block in self.icl_blocks:
                    x_BRiD, _ = block(x_BRiD, num_train)

        x_BRiD = self.output_norm(x_BRiD)

        # --- Split embeddings ---
        if kv_cache is not None and not kv_cache.is_empty():
            test_emb = x_BRiD
            train_emb = kv_cache.train_embeddings
        else:
            test_emb = x_BRiD[:, num_train:]
            train_emb = x_BRiD[:, :num_train]

        # --- Build cache ---
        built_cache = None
        if return_kv_cache:
            built_cache = TabPFNV3Cache(
                icl_cache=icl_cache_out,
                train_embeddings=mx.stop_gradient(train_emb),
                train_shape=(B, num_train),
                inducing_hidden=inducing_hidden,
            )

        # --- Decoder ---
        if self.task_type == "multiclass":
            y_BN = self._prepare_y(y, num_train, B)
            test_out = self.many_class_decoder(train_emb, test_emb, y_BN)
        else:
            test_out_BRD = self.output_projection_linear2(
                nn.gelu(self.output_projection_linear1(test_emb))
            )
            test_out = test_out_BRD.transpose(1, 0, 2)

        # NaN-safe output
        test_out = mx.where(mx.isnan(test_out), mx.array(0.0), test_out)

        if return_kv_cache:
            return test_out, built_cache
        return test_out

    def _stages_0_to_2(
        self,
        x_RiBC: mx.array,
        y: mx.array,
        num_train: int,
        B: int,
        *,
        kv_cache: TabPFNV3Cache | None = None,
        x_is_test_only: bool = False,
        return_inducing_hidden: bool = False,
    ) -> tuple[mx.array, list[mx.array] | None]:
        """Stages 0-2: preprocess, embed, distribute, aggregate."""

        if kv_cache is not None and not kv_cache.is_empty():
            rows_RiBC = x_RiBC if x_is_test_only else x_RiBC[num_train:]
            scaler_cache = kv_cache.scaler_cache
            precomputed_hidden = kv_cache.inducing_hidden
            effective_num_train = 0
        else:
            rows_RiBC = x_RiBC
            scaler_cache = None
            precomputed_hidden = None
            effective_num_train = num_train

        # Stage 0: Preprocess
        x_BRiC, nan_ind_BRiC = self._preprocess_raw(
            rows_RiBC, effective_num_train, scaler_cache
        )

        # Feature grouping
        x_grouped_BRiCG = group_features(
            x_BRiC, nan_ind_BRiC, self.feature_group_size
        )

        # Column-level y embedding
        y_col_emb_BNE = None
        if scaler_cache is None and num_train > 0:
            y_BN = self._prepare_y(y, num_train, B)
            y_col_emb_BNE = self.col_y_encoder(y_BN)

        # Stage 1: Cell embedding
        x_emb = self.x_embed(x_grouped_BRiCG)  # (B, Ri, C, E)

        # Add column y embedding to train rows
        if y_col_emb_BNE is not None and effective_num_train > 0:
            # y_col_emb_BNE: (B, N, E) → expand to (B, N, 1, E) for broadcast over C
            y_expanded = y_col_emb_BNE[:, :, None, :]  # (B, N, 1, E)
            train_emb = x_emb[:, :effective_num_train] + y_expanded
            rest_emb = x_emb[:, effective_num_train:]
            x_emb = mx.concatenate([train_emb, rest_emb], axis=1)

        # Stage 2a: Feature distribution embedding
        x_emb, chunk_hidden = self.feature_distribution_embedder(
            x_emb,
            num_train_rows=effective_num_train,
            cached_hidden=precomputed_hidden,
            return_hidden=return_inducing_hidden,
        )

        # Stage 2b: Column aggregation
        x_BRiClE = self.column_aggregator(x_emb)

        return x_BRiClE, chunk_hidden

    def _preprocess_raw(
        self,
        x_RiBC: mx.array,
        num_train: int,
        scaler_cache: dict[str, mx.array] | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        """Preprocess: NaN indicators → imputation → standardization → transpose."""
        nan_ind_BRiC = None
        if self.use_nan_indicators:
            nan_indicator_RiBC = generate_nan_and_inf_indicator(x_RiBC)
            nan_ind_BRiC = nan_indicator_RiBC.transpose(1, 0, 2)

        x_RiBC = impute_nan_and_inf_with_mean(x_RiBC, num_train, scaler_cache)

        if scaler_cache is not None:
            x_RiBC = self.standard_scaler.transform(x_RiBC, fitted_cache=scaler_cache)
        else:
            x_RiBC = self.standard_scaler(x=x_RiBC, num_train_rows=num_train)

        x_BRiC = x_RiBC.transpose(1, 0, 2)
        return x_BRiC, nan_ind_BRiC

    def _prepare_y(
        self,
        y: mx.array,
        num_train: int,
        batch_size: int,
    ) -> mx.array:
        """Prepare y_train: ensure shape (B, N).

        Handles both 1D (N,) and 2D (N, B) inputs.
        """
        if y.ndim == 1:
            return y[:num_train][None, :]  # (1, N)
        return y[:num_train].transpose(1, 0)  # (B, N)

    def predict_proba(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X_train: (N, C) training features.
            y_train: (N,) training labels.
            X_test: (M, C) test features.

        Returns:
            (M, n_classes) probability array.
        """
        x_combined = np.concatenate([X_train, X_test], axis=0)  # (R, C)
        # Convert to (R, 1, C) — rows-first, batch=1
        x_tensor = mx.array(x_combined[:, None, :].astype(np.float32))
        y_tensor = mx.array(y_train.astype(np.float32))

        logits = self(x_tensor, y_tensor)  # (M, 1, T)
        mx.eval(logits)

        # Convert logits to probabilities
        probs = mx.softmax(logits, axis=-1)
        mx.eval(probs)

        # Shape: (M, 1, T) → (M, T)
        result = np.array(probs[:, 0, :])
        return result

    def predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """Predict class labels (classification) or values (regression)."""
        if self.task_type == "regression":
            return self.predict_regression(X_train, y_train, X_test)
        probs = self.predict_proba(X_train, y_train, X_test)
        return probs.argmax(axis=1)

    def predict_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        borders: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict continuous values via bar distribution decoding.

        Args:
            X_train: (N, C) training features.
            y_train: (N,) training targets (continuous).
            X_test: (M, C) test features.
            borders: (n_buckets+1,) bin boundaries in z-normalized space.
                     If None, uses uniform [-128, 128] with 5000 bins.

        Returns:
            (M,) predicted values in original scale.
        """
        x_combined = np.concatenate([X_train, X_test], axis=0)
        x_tensor = mx.array(x_combined[:, None, :].astype(np.float32))

        # Z-normalize targets (matching official TabPFN preprocessing)
        y_mean = float(y_train.mean())
        y_std = float(y_train.std()) + 1e-8
        y_normalized = (y_train - y_mean) / y_std
        y_tensor = mx.array(y_normalized.astype(np.float32))

        logits = self(x_tensor, y_tensor)  # (M, 1, n_buckets)
        mx.eval(logits)

        probs = mx.softmax(logits[:, 0, :], axis=-1)
        mx.eval(probs)
        probs_np = np.array(probs)

        # Build z-space bin boundaries and midpoints
        if borders is None:
            borders = getattr(self, "regression_borders", None)
        if borders is None:
            n_buckets = probs_np.shape[1]
            borders = np.linspace(-128.0, 128.0, n_buckets + 1).astype(np.float32)
        midpoints = (borders[:-1] + borders[1:]) / 2.0

        # Expected value in z-normalized space
        predictions_znorm = (probs_np * midpoints[None, :]).sum(axis=1)

        # Map back to raw space: raw = znorm * std + mean
        predictions = predictions_znorm * y_std + y_mean
        return predictions
