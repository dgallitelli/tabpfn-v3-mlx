"""Preprocessing: StandardScaler, feature grouping, NaN indicators."""

from __future__ import annotations

import mlx.core as mx


_NAN_INDICATOR = -2.0
_INFINITY_INDICATOR = 2.0
_NEG_INFINITY_INDICATOR = 4.0


class StandardScaler:
    """Standard scaler for MLX arrays with NaN handling."""

    def fit(self, x: mx.array) -> dict[str, mx.array]:
        """Compute mean and std over the first dimension.

        Args:
            x: (T, ...) input tensor.

        Returns:
            Cache dict with 'mean' and 'std'.
        """
        is_finite = mx.isfinite(x)
        # nanmean: replace non-finite with 0, divide by count of finite values
        x_masked = mx.where(is_finite, x, mx.array(0.0))
        count = mx.sum(is_finite.astype(mx.float32), axis=0)
        count = mx.maximum(count, mx.array(1.0))
        mean = mx.sum(x_masked, axis=0) / count

        # nanstd: variance with finite values only
        diff = mx.where(is_finite, x - mean, mx.array(0.0))
        var = mx.sum(diff * diff, axis=0) / mx.maximum(count - 1, mx.array(1.0))
        std = mx.sqrt(var)

        # Handle constant features (std=0)
        std = mx.where(std == 0, mx.ones_like(std), std)

        # Single-row case
        if x.shape[0] == 1:
            std = mx.ones_like(std)

        return {"mean": mean, "std": std}

    def transform(
        self,
        x: mx.array,
        fitted_cache: dict[str, mx.array],
    ) -> mx.array:
        """Apply fitted scaling.

        Returns:
            Scaled tensor clipped to [-100, 100].
        """
        mean = fitted_cache["mean"]
        std = fitted_cache["std"]
        eps = mx.array(1.1920929e-7)  # float32 eps
        x = (x - mean) / (std + eps)
        return mx.clip(x, -100.0, 100.0)

    def __call__(
        self,
        x: mx.array,
        num_train_rows: int | None = None,
    ) -> mx.array:
        """Fit on train rows, transform all."""
        if num_train_rows is not None and num_train_rows > 0:
            fit_data = x[:num_train_rows]
        else:
            fit_data = x
        fitted_cache = self.fit(fit_data)
        return self.transform(x, fitted_cache=fitted_cache)


def generate_nan_and_inf_indicator(x: mx.array) -> mx.array:
    """Generate NaN/Inf indicator features (matches TabPFN v2.5).

    Returns tensor with same shape as x:
    - NaN positions → -2.0
    - +Inf positions → 2.0
    - -Inf positions → 4.0
    - Finite positions → 0.0
    """
    is_nan = mx.isnan(x)
    is_posinf = mx.isinf(x) & (x > 0)
    is_neginf = mx.isinf(x) & (x < 0)
    return (
        is_nan.astype(mx.float32) * _NAN_INDICATOR
        + is_posinf.astype(mx.float32) * _INFINITY_INDICATOR
        + is_neginf.astype(mx.float32) * _NEG_INFINITY_INDICATOR
    )


def impute_nan_and_inf_with_mean(
    x: mx.array,
    num_train_rows: int,
    scaler_cache: dict[str, mx.array] | None = None,
) -> mx.array:
    """Impute NaN and Inf values with the feature mean.

    Args:
        x: (Ri, B, C) input tensor.
        num_train_rows: Number of training rows.
        scaler_cache: Optional pre-computed mean.

    Returns:
        Imputed tensor.
    """
    is_finite = mx.isfinite(x)

    if scaler_cache is not None:
        feature_means = scaler_cache["mean"]
    else:
        if num_train_rows == 0:
            feature_means = mx.zeros(x.shape[1:])
        else:
            x_train = mx.where(
                is_finite[:num_train_rows],
                x[:num_train_rows],
                mx.array(0.0),
            )
            count = mx.sum(is_finite[:num_train_rows].astype(mx.float32), axis=0)
            count = mx.maximum(count, mx.array(1.0))
            feature_means = mx.sum(x_train, axis=0) / count

    # Broadcast means and replace non-finite values
    means_expanded = mx.broadcast_to(feature_means[None, :, :] if x.ndim == 3 else feature_means[None], x.shape)
    return mx.where(is_finite, x, means_expanded)


def group_features(
    x_BRiC: mx.array,
    nan_ind_BRiC: mx.array | None,
    feature_group_size: int,
) -> mx.array:
    """Build grouped + indicator-concatenated tensor.

    Applies circular shifts with powers-of-2 offsets and stacks into groups.

    Args:
        x_BRiC: (B, Ri, C) scaled features.
        nan_ind_BRiC: (B, Ri, C) NaN indicators or None.
        feature_group_size: Number of shifts (default 3).

    Returns:
        (B, Ri, C, G) where G = feature_group_size (or 2*G if nan_indicators).
    """
    C = x_BRiC.shape[2]
    grouped = []
    for i in range(feature_group_size):
        shift = -(2 ** i) % C
        if shift == 0:
            grouped.append(x_BRiC)
        else:
            # Roll along column axis (axis=2)
            grouped.append(mx.concatenate([x_BRiC[:, :, shift:], x_BRiC[:, :, :shift]], axis=2))

    # Stack: (B, Ri, C, G)
    x_grouped = mx.stack(grouped, axis=-1)

    if nan_ind_BRiC is not None:
        ind_grouped = []
        for i in range(feature_group_size):
            shift = -(2 ** i) % C
            if shift == 0:
                ind_grouped.append(nan_ind_BRiC)
            else:
                ind_grouped.append(
                    mx.concatenate([nan_ind_BRiC[:, :, shift:], nan_ind_BRiC[:, :, :shift]], axis=2)
                )
        ind_stacked = mx.stack(ind_grouped, axis=-1)
        x_grouped = mx.concatenate([x_grouped, ind_stacked], axis=-1)

    return x_grouped
