#!/usr/bin/env python3
"""Benchmark TabPFN v3 MLX on sklearn breast_cancer.

Usage:
    python scripts/benchmark.py [--weights PATH]
"""

import argparse
import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn_mlx import TabPFNV3, TabPFNV3Config, load_v3_pytorch_weights


def main():
    parser = argparse.ArgumentParser(description="Benchmark TabPFN v3 MLX")
    parser.add_argument("--weights", type=str, default=None, help="Path to pretrained weights")
    parser.add_argument("--n-runs", type=int, default=5, help="Number of timed runs")
    args = parser.parse_args()

    config = TabPFNV3Config(max_num_classes=2)
    model = TabPFNV3(config, task_type="multiclass")

    if args.weights:
        model = load_v3_pytorch_weights(model, args.weights)
        print(f"Loaded weights from {args.weights}")
    else:
        print("Using random weights (no pretrained checkpoint provided)")

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)

    print(f"\nDataset: breast_cancer (n_train={len(X_train)}, n_test={len(X_test)}, n_features={X_train.shape[1]})")
    print(f"Model: {config.nlayers} ICL layers, embed_dim={config.embed_dim}, icl_emsize={config.icl_emsize}")

    # Warmup
    _ = model.predict_proba(X_train, y_train, X_test)

    # Timed runs
    latencies = []
    for _ in range(args.n_runs):
        start = time.perf_counter()
        probs = model.predict_proba(X_train, y_train, X_test)
        latencies.append(time.perf_counter() - start)

    preds = probs.argmax(axis=1)
    acc = (preds == y_test).mean()

    print(f"\nResults:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Latency (median): {np.median(latencies)*1000:.1f} ms")
    print(f"  Latency (mean):   {np.mean(latencies)*1000:.1f} ms")
    print(f"  Latency (min):    {np.min(latencies)*1000:.1f} ms")


if __name__ == "__main__":
    main()
