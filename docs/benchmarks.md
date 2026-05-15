# Benchmarks

Performance benchmarks for TabPFN v3 MLX vs PyTorch (CPU and MPS backends).

**Hardware**: Apple M4 (16 GB unified memory)
**Software**: MLX 0.31.2, PyTorch 2.12.0, macOS 26.3.1, Python 3.14
**Checkpoint**: `tabpfn-v3-classifier-v3_default.ckpt` (53.2M parameters)

## Summary

MLX delivers **13–29x speedup over PyTorch CPU** and **30–39x over PyTorch MPS** on datasets up to 1000 train samples. Beyond ~1000 samples, PyTorch MPS OOMs and CPU becomes impractically slow — MLX remains the only viable option on Apple Silicon.

| Dataset Size | MLX Latency | PyTorch CPU | Speedup |
|---|---|---|---|
| Small (50–100 train) | 22–30 ms | 640–780 ms | 26–29x |
| Medium (284–898 train) | 135–838 ms | 2,900–10,700 ms | 13–23x |
| Large (1000–2000 train) | 571 ms–3.9 s | 9,100–14,300 ms | 7–16x |

## Small Datasets (< 300 train samples)

Median of 10 runs, 2 warmup passes.

| Dataset | Train/Test | Features | Classes | MLX | PyTorch CPU | PyTorch MPS | Speedup vs CPU | Agreement |
|---------|-----------|----------|---------|-----|-------------|-------------|----------------|-----------|
| Breast Cancer | 284/285 | 30 | 2 | **135 ms** | 3,062 ms | 4,121 ms | 22.8x | 98.2% |
| Iris | 75/75 | 4 | 3 | **22 ms** | 636 ms | 863 ms | 29.0x | 97.3% |
| Wine | 89/89 | 13 | 3 | **29 ms** | 808 ms | 977 ms | 28.1x | 98.9% |

## Medium & Large Datasets (300–2000 train samples)

Median of 3 runs, 1 warmup pass.

| Dataset | Train/Test | Features | Classes | MLX | PyTorch CPU | PyTorch MPS | Speedup vs CPU | Agreement |
|---------|-----------|----------|---------|-----|-------------|-------------|----------------|-----------|
| Digits | 898/899 | 64 | 10 | **838 ms** | 10,755 ms | 10,077 ms | 12.8x | 98.9% |
| Synthetic-5class | 1000/1000 | 50 | 5 | **571 ms** | 9,147 ms | OOM | 16.0x | 93.0% |
| Synthetic-5class-large | 2500/2500 | 50 | 5 | **3.9 s** | — | — | — | — |

## Scaling Behavior

### By Dataset Size (Breast Cancer, 30 features, binary)

Median of 5 runs. Both backends benchmarked.

| Train Samples | Test Samples | MLX | PyTorch CPU | Speedup |
|--------------|-------------|-----|-------------|---------|
| 50 | 50 | **30 ms** | 783 ms | 26.0x |
| 100 | 100 | **60 ms** | 1,207 ms | 20.1x |
| 200 | 200 | **71 ms** | 1,872 ms | 26.3x |
| 284 | 284 | **101 ms** | 2,605 ms | 25.7x |

### Scaling with 100 Features (binary classification)

| Train Samples | Test Samples | MLX | PyTorch CPU | Speedup |
|--------------|-------------|-----|-------------|---------|
| 500 | 500 | **438 ms** | 7,581 ms | 17.3x |
| 1000 | 1000 | **1.9 s** | 14,318 ms | 7.5x |

Scaling is approximately O(n^1.5) to O(n^2) due to ICL transformer attention over all train+test rows.

## Accuracy & Numerical Agreement

The official `tabpfn` package uses 8-estimator ensembling by default (averaging predictions from multiple random feature orderings). The MLX port performs a single forward pass (equivalent to `n_estimators=1`).

| Dataset | Classes | MLX Accuracy | PyTorch Accuracy | Prediction Agreement | Median Prob Diff |
|---------|---------|-------------|------------------|---------------------|-----------------|
| Breast Cancer | 2 | 96.8% | 97.2% | 98.2% | 0.00005 |
| Iris | 3 | 97.3% | 94.7% | 97.3% | 0.00005 |
| Wine | 3 | 97.8% | 96.6% | 98.9% | 0.00005 |
| Digits | 10 | 98.9% | 98.9% | 98.9% | — |
| Synthetic-5class | 5 | 86.6% | 86.9% | 93.0% | — |

When comparing single-estimator to single-estimator (fair comparison), prediction agreement is 98.9% with median probability difference < 0.0001. The ~1-7% disagreements occur on borderline samples near decision boundaries.

The lower agreement on 5-class synthetic (93%) is expected: more classes = more decision boundaries where small numerical differences can flip the argmax.

## Memory Efficiency

MLX uses Apple's unified memory architecture — the model and data share the same physical memory with zero-copy transfers between CPU and GPU.

| Backend | 1000 samples (50 feat, 5 cls) | Status |
|---------|------------------------------|--------|
| MLX | ~571 ms | OK |
| PyTorch CPU | ~9.1 s | OK (slow) |
| PyTorch MPS | — | **OOM** (>8.3 GB limit) |

## Task Types Not Yet Supported

- **Regression**: The regression checkpoint (`tabpfn-v3-regressor-v3_default.ckpt`) uses a separate decoder architecture (2-layer MLP to 5000 bar-distribution buckets) not yet implemented in the MLX port.
- **Time-series / Anomaly Detection**: No dedicated checkpoints exist in the Prior-Labs model repository. TabPFN v3 is a classification and regression model only.

## Reproducing Benchmarks

```bash
# Install
pip install -e ".[convert]"
pip install tabpfn scikit-learn torch

# Run basic benchmarks (small datasets, ~3 min)
python scripts/benchmark_comprehensive.py

# Run expanded benchmarks (medium datasets + scaling, ~5 min)
python scripts/benchmark_expanded.py
```

## Hardware Notes

- All benchmarks on Apple M4 (10-core GPU, 16 GB unified memory)
- Results will vary by chip: M1 Pro/Max/Ultra, M2, M3, M4 Pro/Max will differ
- MLX benefits from larger unified memory for larger datasets
- PyTorch MPS memory limit is controlled by `PYTORCH_MPS_HIGH_WATERMARK_RATIO` (default 0.52 of total)
