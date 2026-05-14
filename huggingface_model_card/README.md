---
library_name: mlx
tags:
  - tabpfn
  - tabular
  - classification
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

Native Apple MLX port of [TabPFN v3](https://github.com/PriorLabs/TabPFN) — the full 53M parameter tabular foundation model running natively on Apple Silicon (M1-M5).

## Overview

This is an **inference-only MLX reimplementation** of the TabPFN v3 architecture. It does not host model weights directly — instead, it loads weights from the official [Prior-Labs/tabpfn_3](https://huggingface.co/Prior-Labs/tabpfn_3) checkpoint and converts them to MLX format on the fly.

**Why no hosted weights?** The official weights are licensed by Prior Labs. This repo provides the MLX architecture code and conversion utilities to use those weights on Apple Silicon with zero-copy unified memory.

## Performance

| Metric | Value |
|--------|-------|
| Parameters | 53.2M |
| Architecture layers | 24 ICL + 3 distribution + 3 aggregation |
| Speedup vs PyTorch CPU | 4.1x on Apple Silicon |
| Prediction agreement | 100% vs official PyTorch |
| Max numerical diff | < 0.08 probability (float32 cross-platform) |

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

# Load from official Prior-Labs checkpoint
model = load_v3_from_checkpoint("path/to/tabpfn-v3-classifier-v3_default.ckpt")

# Predict (sklearn-compatible API)
probs = model.predict_proba(X_train, y_train, X_test)
preds = model.predict(X_train, y_train, X_test)
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
7. **Decoder**: Attention retrieval with one-hot values for multiclass

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
