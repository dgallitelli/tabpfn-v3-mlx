---
library_name: mlx
tags:
  - tabpfn
  - tabular
  - classification
  - regression
  - time-series
  - in-context-learning
  - apple-silicon
  - mlx
license: mit
pipeline_tag: tabular-classification
language:
  - en
base_model: Prior-Labs/tabpfn_3
---

# TabPFN v3 MLX

![Benchmark Results](benchmark_figure.png)

Native Apple MLX port of [TabPFN v3](https://github.com/PriorLabs/TabPFN) — the full 53M parameter tabular foundation model running natively on Apple Silicon (M1-M5).

## Overview

This is an **inference-only MLX reimplementation** of the TabPFN v3 architecture. It does not host model weights directly — instead, it loads weights from the official [Prior-Labs/tabpfn_3](https://huggingface.co/Prior-Labs/tabpfn_3) checkpoint and converts them to MLX format on the fly.

**Why no hosted weights?** The official weights are licensed by Prior Labs. This repo provides the MLX architecture code and conversion utilities to use those weights on Apple Silicon with zero-copy unified memory.

## Performance

| Metric | Value |
|--------|-------|
| Parameters | 53.2M |
| Architecture layers | 24 ICL + 3 distribution + 3 aggregation |
| Speedup vs PyTorch CPU | 13–29x on Apple Silicon |
| Speedup vs PyTorch MPS | 30–39x (MPS OOMs at ~1000 samples) |
| Prediction agreement | 93–99% vs official PyTorch |
| Median numerical diff | < 0.0001 probability |

### Benchmark Results

Tested on Apple M4 (16 GB unified memory), MLX 0.31.2, PyTorch 2.12.0, macOS 26.3.1.

#### Latency Comparison

| Dataset | Samples (train/test) | Features | Classes | MLX | PyTorch CPU | PyTorch MPS | Speedup vs CPU |
|---------|---------------------|----------|---------|-----|-------------|-------------|----------------|
| Breast Cancer | 284 / 285 | 30 | 2 | **135 ms** | 3,062 ms | 4,121 ms | 22.8x |
| Iris | 75 / 75 | 4 | 3 | **22 ms** | 636 ms | 863 ms | 29.0x |
| Wine | 89 / 89 | 13 | 3 | **29 ms** | 808 ms | 977 ms | 28.1x |
| Digits | 898 / 899 | 64 | 10 | **838 ms** | 10,755 ms | 10,077 ms | 12.8x |
| Synthetic-5class | 1000 / 1000 | 50 | 5 | **571 ms** | 9,147 ms | OOM | 16.0x |

#### Scaling with Dataset Size

| Train samples | Test samples | Features | MLX | PyTorch CPU | Speedup |
|--------------|-------------|----------|-----|-------------|---------|
| 50 | 50 | 30 | **30 ms** | 783 ms | 26.0x |
| 100 | 100 | 30 | **60 ms** | 1,207 ms | 20.1x |
| 200 | 200 | 30 | **71 ms** | 1,872 ms | 26.3x |
| 284 | 284 | 30 | **101 ms** | 2,605 ms | 25.7x |
| 500 | 500 | 100 | **438 ms** | 7,581 ms | 17.3x |
| 1000 | 1000 | 100 | **1.9 s** | 14,318 ms | 7.5x |

#### Accuracy & Agreement

| Dataset | Classes | MLX Accuracy | PyTorch Accuracy | Prediction Agreement |
|---------|---------|-------------|------------------|---------------------|
| Breast Cancer | 2 | 96.8% | 97.2% | 98.2% |
| Iris | 3 | 97.3% | 94.7% | 97.3% |
| Wine | 3 | 97.8% | 96.6% | 98.9% |
| Digits | 10 | 98.9% | 98.9% | 98.9% |
| Synthetic-5class | 5 | 86.6% | 86.9% | 93.0% |

> **Note**: The official `tabpfn` package uses 8-estimator ensembling by default. MLX performs a single forward pass (equivalent to `n_estimators=1`).
> When comparing single-estimator to single-estimator, prediction agreement is 98.9% with median probability difference < 0.0001.
> The ~1-7% disagreements occur on borderline samples near decision boundaries.

### Time-Series Regression

Using the `tabpfn-v3-regressor-v3_20260506_timeseries.ckpt` checkpoint with lagged-feature encoding.
Targets are z-normalized internally; predictions decoded via 5000-bin bar distribution.

| Dataset | Train/Test | Lags | MLX Latency | R² |
|---------|-----------|------|-------------|-----|
| Sine wave + noise | 150/45 | 5 | **21 ms** | 0.825 |
| Damped oscillation | 350/140 | 10 | **56 ms** | 0.884 |
| Multi-frequency signal | 700/285 | 15 | **135 ms** | 0.959 |
| Random walk + trend | 350/140 | 10 | **56 ms** | 0.881 |

**Speedup vs PyTorch CPU (sine wave):** 23.9x (21 ms vs 512 ms)

See [docs/benchmarks.md](https://github.com/dgallitelli/tabpfn-v3-mlx/blob/main/docs/benchmarks.md) for full methodology.

## Installation

```bash
pip install tabpfn-v3-mlx
```

For weight conversion from PyTorch checkpoints:
```bash
pip install "tabpfn-v3-mlx[convert]"
```

## Usage

```python
from tabpfn_mlx import load_v3_from_checkpoint

# Classification
model = load_v3_from_checkpoint("path/to/tabpfn-v3-classifier-v3_default.ckpt")
probs = model.predict_proba(X_train, y_train, X_test)
preds = model.predict(X_train, y_train, X_test)

# Regression / Time-Series
model = load_v3_from_checkpoint("path/to/tabpfn-v3-regressor-v3_20260506_timeseries.ckpt",
                                task_type="regression")
predictions = model.predict(X_train, y_train, X_test)
```

### Downloading weights

```python
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download("Prior-Labs/tabpfn_3", "tabpfn-v3-classifier-v3_default.ckpt")
model = load_v3_from_checkpoint(ckpt)
```

## Architecture

The full v3 pipeline ported to MLX:

1. **Preprocessing**: NaN indicators, mean imputation, z-score standardization
2. **Feature grouping**: Circular shifts (groups of 3) + optional NaN indicators
3. **Cell embedding**: Linear(6, 128) per feature group
4. **Distribution embedding**: 3x InducedSelfAttention blocks (O(n) via learnable inducing points)
5. **Column aggregation**: 3x TransformerBlocks with RoPE + CLS token readout → (B, R, 4, 128)
6. **ICL transformer**: 24 layers with pre-norm RMSNorm, train-only K/V, GQA, SoftmaxScalingMLP
7. **Decoder**: Attention retrieval with one-hot values for multiclass; MLP + bar distribution for regression

## KV Cache

```python
# Build cache once from training data
logits, cache = model(x, y, return_kv_cache=True)

# Reuse for new test batches (skips stages 0-2)
logits_new = model(x_test, y, kv_cache=cache, x_is_test_only=True)
```

## Links

- **Code**: [github.com/dgallitelli/tabpfn-v3-mlx](https://github.com/dgallitelli/tabpfn-v3-mlx)
- **Official weights**: [Prior-Labs/tabpfn_3](https://huggingface.co/Prior-Labs/tabpfn_3)
- **Paper**: [Accurate Predictions on Small Data with a Tabular Foundation Model](https://arxiv.org/abs/2511.03634) (Nature 2025)

## Citations

If you use this work, please cite the relevant TabPFN papers:

**TabPFN v1/v2 (Nature 2024):**
```bibtex
@article{hollmann2024tabpfn,
  title={Accurate Predictions on Small Data with a Tabular Foundation Model},
  author={Hollmann, Noah and M{\"u}ller, Samuel and Hutter, Frank},
  journal={Nature},
  year={2024},
  doi={10.1038/s41586-024-08328-6},
  url={https://www.nature.com/articles/s41586-024-08328-6}
}
```

**TabPFN v2.5 (ArXiv 2025):**
```bibtex
@article{hollmann2025tabpfn,
  title={TabPFN: Highly Accurate Tabular Classification in Under a Second},
  author={Hollmann, Noah and M{\"u}ller, Samuel and Purucker, Lennart and others},
  journal={arXiv preprint arXiv:2511.08667},
  year={2025},
  url={https://arxiv.org/abs/2511.08667}
}
```

**TabPFN v3 Technical Report:**
```bibtex
@techreport{priorlabs2026tabpfnv3,
  title={TabPFN v3: Scaling Tabular Foundation Models},
  author={Prior Labs},
  year={2026},
  url={https://priorlabs.ai/technical-reports/tabpfn-3}
}
```

**nanoTabPFN (ArXiv 2024):**
```bibtex
@article{pfefferle2024nanotabpfn,
  title={nanoTabPFN: A Lightweight and Educational Reimplementation of TabPFN},
  author={Pfefferle, Alexander and Hog, Johannes and Purucker, Lennart and Hutter, Frank},
  journal={arXiv preprint arXiv:2511.03634},
  year={2024},
  url={https://arxiv.org/abs/2511.03634}
}
```
