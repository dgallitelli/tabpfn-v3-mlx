#!/usr/bin/env python3
"""Comprehensive benchmark: TabPFN v3 MLX vs PyTorch CPU vs PyTorch MPS.

Tests across multiple dataset sizes, measures latency, accuracy, and numerical agreement.
"""

import json
import sys
import time

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

CKPT_PATH = (
    "/Users/dggallit/.cache/huggingface/hub/models--Prior-Labs--tabpfn_3/"
    "snapshots/299726533b098357029ecac7c4fa27f97c7f9238/"
    "tabpfn-v3-classifier-v3_default.ckpt"
)

N_WARMUP = 2
N_RUNS = 10


def benchmark_mlx(X_train, y_train, X_test, y_test, n_runs=N_RUNS):
    from tabpfn_mlx import load_v3_from_checkpoint

    model = load_v3_from_checkpoint(CKPT_PATH, task_type="multiclass")

    n_classes = len(np.unique(np.concatenate([y_train, y_test])))

    X_tr = X_train.astype(np.float32)
    y_tr = y_train.astype(np.float32)
    X_te = X_test.astype(np.float32)

    for _ in range(N_WARMUP):
        _ = model.predict_proba(X_tr, y_tr, X_te)

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        probs_raw = model.predict_proba(X_tr, y_tr, X_te)
        latencies.append(time.perf_counter() - start)

    # MLX outputs full max_num_classes slots; slice to actual classes
    probs = probs_raw[:, :n_classes]
    # Renormalize after slicing
    probs = probs / probs.sum(axis=1, keepdims=True)

    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs, labels=list(range(n_classes)))

    return {
        "backend": "MLX (Apple Silicon)",
        "accuracy": acc,
        "log_loss": ll,
        "latency_median_ms": np.median(latencies) * 1000,
        "latency_mean_ms": np.mean(latencies) * 1000,
        "latency_min_ms": np.min(latencies) * 1000,
        "latency_std_ms": np.std(latencies) * 1000,
        "probs": probs,
        "latencies": latencies,
    }


def benchmark_pytorch(X_train, y_train, X_test, y_test, device="cpu", n_runs=N_RUNS):
    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier(device=device)
    clf.fit(X_train, y_train)

    for _ in range(N_WARMUP):
        _ = clf.predict_proba(X_test)

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        probs = clf.predict_proba(X_test)
        latencies.append(time.perf_counter() - start)

    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_test, preds)
    n_classes = probs.shape[1]
    ll = log_loss(y_test, probs, labels=list(range(n_classes)))

    return {
        "backend": f"PyTorch ({device.upper()})",
        "accuracy": acc,
        "log_loss": ll,
        "latency_median_ms": np.median(latencies) * 1000,
        "latency_mean_ms": np.mean(latencies) * 1000,
        "latency_min_ms": np.min(latencies) * 1000,
        "latency_std_ms": np.std(latencies) * 1000,
        "probs": probs,
        "latencies": latencies,
    }


def compute_agreement(probs_a, probs_b):
    preds_a = probs_a.argmax(axis=1)
    preds_b = probs_b.argmax(axis=1)
    agreement = (preds_a == preds_b).mean()
    max_diff = np.abs(probs_a - probs_b).max()
    mean_diff = np.abs(probs_a - probs_b).mean()
    return {
        "prediction_agreement": agreement,
        "max_prob_diff": float(max_diff),
        "mean_prob_diff": float(mean_diff),
    }


def run_dataset_benchmark(name, X, y, test_size=0.5, random_state=42):
    print(f"\n{'='*70}")
    print(f"Dataset: {name} (n={len(X)}, features={X.shape[1]}, classes={len(np.unique(y))})")
    print(f"{'='*70}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    results = {}

    # MLX
    print("\n  [MLX] Running...")
    try:
        results["mlx"] = benchmark_mlx(X_train, y_train, X_test, y_test)
        print(f"    Accuracy: {results['mlx']['accuracy']:.4f}")
        print(f"    Latency:  {results['mlx']['latency_median_ms']:.1f} ms (median)")
    except Exception as e:
        print(f"    FAILED: {e}")
        results["mlx"] = None

    # PyTorch CPU
    print("\n  [PyTorch CPU] Running...")
    try:
        results["pytorch_cpu"] = benchmark_pytorch(X_train, y_train, X_test, y_test, device="cpu")
        print(f"    Accuracy: {results['pytorch_cpu']['accuracy']:.4f}")
        print(f"    Latency:  {results['pytorch_cpu']['latency_median_ms']:.1f} ms (median)")
    except Exception as e:
        print(f"    FAILED: {e}")
        results["pytorch_cpu"] = None

    # PyTorch MPS
    print("\n  [PyTorch MPS] Running...")
    try:
        import torch
        if torch.backends.mps.is_available():
            results["pytorch_mps"] = benchmark_pytorch(X_train, y_train, X_test, y_test, device="mps")
            print(f"    Accuracy: {results['pytorch_mps']['accuracy']:.4f}")
            print(f"    Latency:  {results['pytorch_mps']['latency_median_ms']:.1f} ms (median)")
        else:
            print("    SKIPPED: MPS not available")
            results["pytorch_mps"] = None
    except Exception as e:
        print(f"    FAILED: {e}")
        results["pytorch_mps"] = None

    # Agreement
    if results.get("mlx") and results.get("pytorch_cpu"):
        agreement = compute_agreement(results["mlx"]["probs"], results["pytorch_cpu"]["probs"])
        print(f"\n  [Agreement MLX vs PyTorch CPU]")
        print(f"    Prediction agreement: {agreement['prediction_agreement']*100:.1f}%")
        print(f"    Max prob difference:  {agreement['max_prob_diff']:.6f}")
        print(f"    Mean prob difference: {agreement['mean_prob_diff']:.6f}")
        results["agreement_mlx_cpu"] = agreement

    # Speedup
    if results.get("mlx") and results.get("pytorch_cpu"):
        speedup = results["pytorch_cpu"]["latency_median_ms"] / results["mlx"]["latency_median_ms"]
        print(f"\n  [Speedup] MLX is {speedup:.2f}x vs PyTorch CPU")
        results["speedup_vs_cpu"] = speedup

    if results.get("mlx") and results.get("pytorch_mps"):
        speedup_mps = results["pytorch_mps"]["latency_median_ms"] / results["mlx"]["latency_median_ms"]
        print(f"  [Speedup] MLX is {speedup_mps:.2f}x vs PyTorch MPS")
        results["speedup_vs_mps"] = speedup_mps

    return results


def run_scaling_benchmark():
    """Test how latency scales with dataset size."""
    print(f"\n{'='*70}")
    print("Scaling Benchmark: Latency vs Dataset Size (breast_cancer features)")
    print(f"{'='*70}")

    from tabpfn_mlx import load_v3_from_checkpoint
    from tabpfn import TabPFNClassifier

    X_full, y_full = load_breast_cancer(return_X_y=True)

    mlx_model = load_v3_from_checkpoint(CKPT_PATH, task_type="multiclass")
    pytorch_clf = TabPFNClassifier(device="cpu")

    sizes = [50, 100, 200, 284]
    scaling_results = []

    for n_train in sizes:
        n_test = min(n_train, len(X_full) - n_train)
        if n_train + n_test > len(X_full):
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, train_size=n_train, test_size=n_test, random_state=42, stratify=y_full
        )

        # MLX
        X_tr = X_train.astype(np.float32)
        y_tr = y_train.astype(np.float32)
        X_te = X_test.astype(np.float32)
        _ = mlx_model.predict_proba(X_tr, y_tr, X_te)

        mlx_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = mlx_model.predict_proba(X_tr, y_tr, X_te)
            mlx_times.append(time.perf_counter() - start)

        # PyTorch CPU
        pytorch_clf.fit(X_train, y_train)
        _ = pytorch_clf.predict_proba(X_test)

        cpu_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = pytorch_clf.predict_proba(X_test)
            cpu_times.append(time.perf_counter() - start)

        mlx_med = np.median(mlx_times) * 1000
        cpu_med = np.median(cpu_times) * 1000
        speedup = cpu_med / mlx_med

        row = {
            "n_train": n_train,
            "n_test": n_test,
            "mlx_median_ms": mlx_med,
            "cpu_median_ms": cpu_med,
            "speedup": speedup,
        }
        scaling_results.append(row)
        print(f"  n_train={n_train:>3}, n_test={n_test:>3}: MLX {mlx_med:>7.1f} ms | CPU {cpu_med:>7.1f} ms | {speedup:.2f}x")

    return scaling_results


def main():
    print("TabPFN v3 Comprehensive Benchmark")
    print("=" * 70)
    print(f"System: Apple Silicon (MLX)")
    print(f"Checkpoint: {CKPT_PATH.split('/')[-1]}")
    print(f"Warmup runs: {N_WARMUP}, Timed runs: {N_RUNS}")

    import platform
    import mlx.core as mx
    print(f"Platform: {platform.machine()}, macOS {platform.mac_ver()[0]}")
    print(f"MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")

    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    all_results = {}

    # Dataset benchmarks
    datasets = [
        ("Breast Cancer", *load_breast_cancer(return_X_y=True)),
        ("Iris", *load_iris(return_X_y=True)),
        ("Wine", *load_wine(return_X_y=True)),
    ]

    for name, X, y in datasets:
        all_results[name] = run_dataset_benchmark(name, X, y)

    # Scaling benchmark
    all_results["scaling"] = run_scaling_benchmark()

    # Summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Dataset':<16} | {'MLX (ms)':<10} | {'CPU (ms)':<10} | {'MPS (ms)':<10} | {'Speedup':<8} | {'Agreement'}")
    print("-" * 80)

    for name in ["Breast Cancer", "Iris", "Wine"]:
        r = all_results[name]
        mlx_ms = f"{r['mlx']['latency_median_ms']:.1f}" if r.get("mlx") else "N/A"
        cpu_ms = f"{r['pytorch_cpu']['latency_median_ms']:.1f}" if r.get("pytorch_cpu") else "N/A"
        mps_ms = f"{r['pytorch_mps']['latency_median_ms']:.1f}" if r.get("pytorch_mps") else "N/A"
        speedup = f"{r.get('speedup_vs_cpu', 0):.2f}x" if r.get("speedup_vs_cpu") else "N/A"
        agree = f"{r['agreement_mlx_cpu']['prediction_agreement']*100:.1f}%" if r.get("agreement_mlx_cpu") else "N/A"
        print(f"{name:<16} | {mlx_ms:<10} | {cpu_ms:<10} | {mps_ms:<10} | {speedup:<8} | {agree}")

    # Save results to JSON (without numpy arrays)
    json_results = {}
    for name in ["Breast Cancer", "Iris", "Wine"]:
        r = all_results[name]
        json_results[name] = {}
        for backend in ["mlx", "pytorch_cpu", "pytorch_mps"]:
            if r.get(backend):
                json_results[name][backend] = {
                    k: v for k, v in r[backend].items()
                    if k not in ("probs", "latencies")
                }
                json_results[name][backend]["latencies"] = r[backend]["latencies"]
        if r.get("agreement_mlx_cpu"):
            json_results[name]["agreement_mlx_cpu"] = r["agreement_mlx_cpu"]
        if r.get("speedup_vs_cpu"):
            json_results[name]["speedup_vs_cpu"] = r["speedup_vs_cpu"]
        if r.get("speedup_vs_mps"):
            json_results[name]["speedup_vs_mps"] = r["speedup_vs_mps"]

    json_results["scaling"] = all_results["scaling"]

    output_path = "/Users/dggallit/tabpfn-v3-mlx/scripts/benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
