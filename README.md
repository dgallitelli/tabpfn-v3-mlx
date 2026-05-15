# TabPFN v3 MLX

[![CI](https://github.com/dgallitelli/tabpfn-v3-mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/dgallitelli/tabpfn-v3-mlx/actions/workflows/ci.yml)

Native Apple MLX port of [TabPFN v3](https://github.com/PriorLabs/TabPFN) — the full 53M parameter tabular foundation model running natively on Apple Silicon.

TabPFN v3 performs classification and regression via **in-context learning** — given training data and test features, it produces predictions in a single forward pass with no gradient descent.

This port runs the complete architecture natively on M1/M2/M3/M4/M5 via Apple's [MLX](https://github.com/ml-explore/mlx) framework with zero-copy unified memory.

## Architecture

The full v3 pipeline (2,395 lines in PyTorch) ported to MLX:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TabPFN v3 Forward Pass                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Stage 0: Preprocessing                                              │
│  ─────────────────────                                               │
│  x_raw → NaN indicators → mean imputation → z-score scaling         │
│        → circular-shift feature grouping (groups of 3)               │
│                                                                      │
│  Stage 1: Cell + Target Embedding                                    │
│  ────────────────────────────────                                    │
│  grouped_features → Linear(G, 128) → add target embedding (train)   │
│                                                                      │
│  Stage 2a: Distribution Embedding (× 3 blocks)                      │
│  ──────────────────────────────────────────────                      │
│  InducedSelfAttention per column:                                    │
│    inducing_points → cross_attn(ind, train) → hidden                 │
│    all_rows → cross_attn(rows, hidden) → updated embeddings          │
│  Complexity: O(R × n_inducing) instead of O(R²)                      │
│                                                                      │
│  Stage 2b: Column Aggregation (× 3 blocks + readout)                 │
│  ────────────────────────────────────────────────────                 │
│  Prepend CLS tokens → self-attention over features (with RoPE)       │
│  Last block: CLS cross-attends to full sequence → (B, R, 4, 128)    │
│                                                                      │
│  Flatten: (B, R, 4, 128) → (B, R, 512)                              │
│                                                                      │
│  Stage 3: ICL Transformer (× 24 layers)                              │
│  ──────────────────────────────────────                               │
│  Pre-norm RMSNorm → ICL Attention (K/V from train only)              │
│  + SoftmaxScalingMLP (learned query scaling)                         │
│  + GQA (optional fewer KV heads for test rows)                       │
│  + MLP (GELU, no bias, zero-init output)                             │
│  Supports KV caching for efficient repeated inference                │
│                                                                      │
│  Stage 4: Decoder                                                    │
│  ────────────────                                                    │
│  Multiclass: attention retrieval (test→train with one-hot values)    │
│  Regression: MLP → bar distribution buckets                          │
│                                                                      │
│  Output: logits → softmax → probabilities                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Performance

Benchmarked on Apple M4 (16 GB), MLX 0.31.2, PyTorch 2.12.0. Median of 10 runs.

| Dataset | MLX | PyTorch CPU | PyTorch MPS | Speedup vs CPU |
|---------|-----|-------------|-------------|----------------|
| Breast Cancer (284 train, 30 features) | **135 ms** | 3,062 ms | 4,121 ms | 22.8x |
| Iris (75 train, 4 features) | **22 ms** | 636 ms | 863 ms | 29.0x |
| Wine (89 train, 13 features) | **29 ms** | 808 ms | 977 ms | 28.1x |

Prediction agreement with official PyTorch: 98–99% (median probability diff < 0.0001).
Disagreements occur only on borderline samples at decision boundaries.

See the [HuggingFace model card](https://huggingface.co/dgallitelli/tabpfn-v3-mlx) for full scaling analysis.

## Installation

```bash
pip install tabpfn-v3-mlx
```

For weight conversion from PyTorch checkpoints:

```bash
pip install "tabpfn-v3-mlx[convert]"
```

## Quick Start

```python
import numpy as np
from tabpfn_mlx import TabPFNV3, TabPFNV3Config, load_v3_pytorch_weights

# Initialize with default config (53M params, 24 ICL layers)
config = TabPFNV3Config(max_num_classes=2)
model = TabPFNV3(config, task_type="multiclass")

# Load pretrained weights (when available)
# model = load_v3_pytorch_weights(model, "path/to/checkpoint.safetensors")

# Predict
probs = model.predict_proba(X_train, y_train, X_test)
preds = model.predict(X_train, y_train, X_test)
```

## Configuration

```python
from tabpfn_mlx import TabPFNV3Config

config = TabPFNV3Config(
    embed_dim=128,                  # Base embedding dimension
    dist_embed_num_blocks=3,        # Distribution embedder layers
    dist_embed_num_heads=8,         # Heads in distribution embedder
    dist_embed_num_inducing_points=128,  # SetTransformer inducing points
    feat_agg_num_blocks=3,          # Column aggregator layers
    feat_agg_num_heads=8,           # Heads in column aggregator
    feat_agg_num_cls_tokens=4,      # CLS tokens (icl_emsize = embed_dim × this)
    nlayers=24,                     # ICL transformer depth
    icl_num_heads=8,                # ICL attention heads
    icl_num_kv_heads=None,          # GQA KV heads (None = standard MHA)
    ff_factor=2,                    # MLP expansion factor
    max_num_classes=10,             # Maximum classes supported
    feature_group_size=3,           # Circular-shift group size
    use_nan_indicators=True,        # NaN/Inf indicator features
)
# icl_emsize = 128 × 4 = 512
```

## KV Cache (Efficient Repeated Inference)

```python
# Build cache from training data (one-time cost)
logits, cache = model(x, y, return_kv_cache=True)

# Reuse cache for new test batches (skips stages 0-2 + K/V projection)
logits_new = model(x_test_only, y, kv_cache=cache, x_is_test_only=True)
```

## Key Differences from nanoTabPFN (v2)

| Aspect | nanoTabPFN | TabPFN v3 |
|--------|-----------|-----------|
| Parameters | 356K | 53M |
| Layers | 3 | 24 ICL + 3 dist + 3 agg |
| Normalization | Post-norm LayerNorm | Pre-norm RMSNorm |
| Feature attention | Direct O(R²) | Induced O(R×k) |
| Positional encoding | None | RoPE + SoftmaxScalingMLP |
| GQA | No | Yes |
| KV cache | No | Multi-level |
| Decoder | MLP | Attention retrieval |

## Development

```bash
git clone https://github.com/dgallitelli/tabpfn-v3-mlx.git
cd tabpfn-v3-mlx
pip install -e ".[dev]"
pytest
```

## Citation

```bibtex
@article{hollmann2025tabpfn,
    title={Accurate Predictions on Small Data with a Tabular Foundation Model},
    author={Hollmann, Noah and Müller, Samuel and Purucker, Lennart and
            Krishnakumar, Arjun and Körfer, Max and Hoo, Shi Bin and
            Schirrmeister, Robin Tibor and Hutter, Frank},
    journal={Nature},
    year={2025}
}
```

## License

MIT. The TabPFN v3 model architecture and weights are subject to [their own license](https://github.com/PriorLabs/TabPFN/blob/main/LICENSE).
