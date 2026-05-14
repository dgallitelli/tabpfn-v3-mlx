"""Tests for the TabPFN v3 MLX model."""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import pytest

from tabpfn_mlx.model import TabPFNV3, TabPFNV3Cache
from tabpfn_mlx.config import TabPFNV3Config


def _small_config(**kwargs):
    defaults = dict(
        embed_dim=32,
        dist_embed_num_blocks=1,
        dist_embed_num_heads=4,
        dist_embed_num_inducing_points=8,
        feat_agg_num_blocks=1,
        feat_agg_num_heads=4,
        feat_agg_num_cls_tokens=2,
        nlayers=2,
        icl_num_heads=4,
        ff_factor=2,
        max_num_classes=3,
        feature_group_size=2,
        use_nan_indicators=False,
    )
    defaults.update(kwargs)
    return TabPFNV3Config(**defaults)


class TestTabPFNV3Forward:
    def test_output_shape_multiclass(self):
        config = _small_config()
        model = TabPFNV3(config, task_type="multiclass")
        x = mx.random.normal((8, 1, 4))  # (R, B, C): 5 train + 3 test
        y = mx.array([0, 1, 2, 0, 1], dtype=mx.float32)
        logits = model(x, y)
        mx.eval(logits)
        assert logits.shape == (3, 1, 3)  # (M, B, T)

    def test_output_shape_batch(self):
        config = _small_config()
        model = TabPFNV3(config, task_type="multiclass")
        B = 2
        x = mx.random.normal((8, B, 4))
        y = mx.array([[0, 1, 2, 0, 1], [2, 0, 1, 2, 0]], dtype=mx.float32).T  # (N, B)
        logits = model(x, y)
        mx.eval(logits)
        assert logits.shape == (3, B, 3)

    def test_probabilities_sum_to_one(self):
        config = _small_config()
        model = TabPFNV3(config, task_type="multiclass")
        np.random.seed(42)
        X_train = np.random.randn(10, 6).astype(np.float32)
        y_train = np.array([0, 1, 2] * 3 + [0], dtype=np.float32)
        X_test = np.random.randn(5, 6).astype(np.float32)
        probs = model.predict_proba(X_train, y_train, X_test)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_nan_indicators_enabled(self):
        config = _small_config(use_nan_indicators=True)
        model = TabPFNV3(config, task_type="multiclass")
        x = mx.random.normal((8, 1, 4))
        y = mx.array([0, 1, 2, 0, 1], dtype=mx.float32)
        logits = model(x, y)
        mx.eval(logits)
        assert logits.shape == (3, 1, 3)

    def test_kv_cache_build(self):
        config = _small_config()
        model = TabPFNV3(config, task_type="multiclass")
        x = mx.random.normal((8, 1, 4))
        y = mx.array([0, 1, 2, 0, 1], dtype=mx.float32)
        logits, cache = model(x, y, return_kv_cache=True)
        mx.eval(logits)
        assert not cache.is_empty()
        assert len(cache.icl_cache) == 2  # nlayers=2
        assert cache.train_embeddings.shape == (1, 5, config.icl_emsize)

    def test_kv_cache_reuse(self):
        config = _small_config()
        model = TabPFNV3(config, task_type="multiclass")
        x = mx.random.normal((8, 1, 4))
        y = mx.array([0, 1, 2, 0, 1], dtype=mx.float32)

        logits_full = model(x, y)
        logits_cached, cache = model(x, y, return_kv_cache=True)
        mx.eval(logits_full)
        mx.eval(logits_cached)

        diff = np.abs(np.array(logits_full) - np.array(logits_cached)).max()
        assert diff < 1e-5

    def test_different_feature_counts(self):
        config = _small_config(feature_group_size=3)
        model = TabPFNV3(config, task_type="multiclass")
        for n_features in [3, 7, 12, 20]:
            x = mx.random.normal((8, 1, n_features))
            y = mx.array([0, 1, 2, 0, 1], dtype=mx.float32)
            logits = model(x, y)
            mx.eval(logits)
            assert logits.shape == (3, 1, 3)


class TestTabPFNV3Config:
    def test_icl_emsize(self):
        config = TabPFNV3Config(embed_dim=128, feat_agg_num_cls_tokens=4)
        assert config.icl_emsize == 512

    def test_invalid_kv_heads(self):
        with pytest.raises(ValueError):
            TabPFNV3Config(icl_num_heads=8, icl_num_kv_heads=3)

    def test_valid_gqa(self):
        config = TabPFNV3Config(icl_num_heads=8, icl_num_kv_heads=4)
        assert config.icl_num_kv_heads == 4


class TestTabPFNV3Parameters:
    def test_full_config_param_count(self):
        config = TabPFNV3Config()
        model = TabPFNV3(config, task_type="multiclass")
        params = nn.utils.tree_flatten(model.parameters())
        total = sum(p.size for _, p in params)
        # ~53M params for the full config
        assert total > 50_000_000
        assert total < 60_000_000
