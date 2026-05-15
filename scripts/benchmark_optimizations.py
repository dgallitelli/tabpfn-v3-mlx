#!/usr/bin/env python3
"""Benchmark performance optimizations: compile, chunking, float16."""

import time
import tracemalloc

import numpy as np
from sklearn.datasets import make_classification

CKPT_PATH = (
    "/Users/dggallit/.cache/huggingface/hub/models--Prior-Labs--tabpfn_3/"
    "snapshots/299726533b098357029ecac7c4fa27f97c7f9238/"
    "tabpfn-v3-classifier-v3_default.ckpt"
)

SIZES = [(1000, 1000), (2000, 2000), (5000, 5000)]
N_FEATURES = 50
N_CLASSES = 5
N_WARMUP = 1
N_RUNS = 3


def generate_data(n_train, n_test):
    X, y = make_classification(
        n_samples=n_train + n_test,
        n_features=N_FEATURES,
        n_informative=30,
        n_classes=N_CLASSES,
        n_clusters_per_class=2,
        random_state=42,
    )
    return (
        X[:n_train].astype(np.float32),
        y[:n_train].astype(np.float32),
        X[n_train:].astype(np.float32),
    )


def benchmark_variant(model, X_train, y_train, X_test, label, chunk_size=None):
    """Run benchmark for a model variant."""
    import mlx.core as mx

    # Warmup
    for _ in range(N_WARMUP):
        _ = model.predict_proba(X_train, y_train, X_test, inference_chunk_size=chunk_size)
        mx.eval(mx.zeros(1))  # sync

    # Timed runs
    latencies = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        _ = model.predict_proba(X_train, y_train, X_test, inference_chunk_size=chunk_size)
        mx.eval(mx.zeros(1))
        latencies.append(time.perf_counter() - start)

    med = np.median(latencies) * 1000
    return med


def measure_memory(model, X_train, y_train, X_test, chunk_size=None):
    """Measure peak memory usage."""
    import mlx.core as mx

    tracemalloc.start()
    _ = model.predict_proba(X_train, y_train, X_test, inference_chunk_size=chunk_size)
    mx.eval(mx.zeros(1))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)  # MB


def main():
    import mlx.core as mx
    from tabpfn_mlx import load_v3_from_checkpoint

    print("TabPFN v3 MLX — Performance Optimization Benchmarks")
    print("=" * 70)
    print(f"Features: {N_FEATURES}, Classes: {N_CLASSES}")
    print(f"Warmup: {N_WARMUP}, Timed runs: {N_RUNS}")
    print()

    # Load variants
    print("Loading models...")
    model_fp32 = load_v3_from_checkpoint(CKPT_PATH, task_type="multiclass")
    model_compiled = load_v3_from_checkpoint(CKPT_PATH, task_type="multiclass", compile=True)
    model_fp16 = load_v3_from_checkpoint(CKPT_PATH, task_type="multiclass", dtype=mx.float16)
    model_fp16_compiled = load_v3_from_checkpoint(
        CKPT_PATH, task_type="multiclass", dtype=mx.float16, compile=True
    )
    print("Models loaded.\n")

    # Results table
    print(f"{'Config':<30} | {'1K train':<12} | {'2K train':<12} | {'5K train':<12}")
    print("-" * 70)

    results = {}
    configs = [
        ("FP32 (baseline)", model_fp32, None),
        ("FP32 + compile", model_compiled, None),
        ("FP32 + chunk=512", model_fp32, 512),
        ("FP16", model_fp16, None),
        ("FP16 + compile", model_fp16_compiled, None),
        ("FP16 + compile + chunk=512", model_fp16_compiled, 512),
    ]

    for label, model, chunk_size in configs:
        row = []
        for n_train, n_test in SIZES:
            X_train, y_train, X_test = generate_data(n_train, n_test)
            try:
                med_ms = benchmark_variant(model, X_train, y_train, X_test, label, chunk_size)
                row.append(f"{med_ms:.0f} ms")
            except Exception as e:
                row.append(f"FAIL")
                print(f"  [{label}] n={n_train} failed: {e}")

        results[label] = row
        print(f"{label:<30} | {row[0]:<12} | {row[1]:<12} | {row[2]:<12}")

    # Memory comparison at 2K
    print(f"\n{'='*70}")
    print("Peak Memory Usage (2K train, 2K test)")
    print(f"{'='*70}")

    X_train, y_train, X_test = generate_data(2000, 2000)
    mem_configs = [
        ("FP32 (baseline)", model_fp32, None),
        ("FP32 + chunk=512", model_fp32, 512),
        ("FP16", model_fp16, None),
        ("FP16 + chunk=512", model_fp16, 512),
    ]
    for label, model, chunk_size in mem_configs:
        try:
            peak_mb = measure_memory(model, X_train, y_train, X_test, chunk_size)
            print(f"  {label:<30}: {peak_mb:.1f} MB peak")
        except Exception as e:
            print(f"  {label:<30}: FAIL ({e})")

    # Speedup summary
    print(f"\n{'='*70}")
    print("Speedup Summary (vs FP32 baseline)")
    print(f"{'='*70}")
    if "FP32 (baseline)" in results:
        baseline = results["FP32 (baseline)"]
        for label, row in results.items():
            if label == "FP32 (baseline)":
                continue
            speedups = []
            for i in range(len(row)):
                try:
                    b_ms = float(baseline[i].replace(" ms", "").replace(",", ""))
                    r_ms = float(row[i].replace(" ms", "").replace(",", ""))
                    speedups.append(f"{b_ms/r_ms:.2f}x")
                except (ValueError, ZeroDivisionError):
                    speedups.append("N/A")
            print(f"  {label:<30}: {', '.join(speedups)}")


if __name__ == "__main__":
    main()
