#!/usr/bin/env python3
"""Expanded benchmarks: larger datasets, multiclass, scaling limits.

Tests MLX vs PyTorch CPU vs PyTorch MPS on datasets beyond the basic sklearn trio.
"""

import json
import os
import time
import warnings

import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    load_digits,
    make_classification,
)
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

CKPT_PATH = (
    "/Users/dggallit/.cache/huggingface/hub/models--Prior-Labs--tabpfn_3/"
    "snapshots/299726533b098357029ecac7c4fa27f97c7f9238/"
    "tabpfn-v3-classifier-v3_default.ckpt"
)

N_WARMUP = 1
N_RUNS = 3
# For large datasets, PyTorch is extremely slow — cap comparison at 1000 samples
PYTORCH_MAX_SAMPLES = 1000


def benchmark_mlx(X_train, y_train, X_test, y_test, n_classes, n_runs=N_RUNS):
    from tabpfn_mlx import load_v3_from_checkpoint

    model = load_v3_from_checkpoint(CKPT_PATH, task_type="multiclass")

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

    probs = probs_raw[:, :n_classes]
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    probs = probs / row_sums

    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs, labels=list(range(n_classes)))

    return {
        "backend": "MLX",
        "accuracy": acc,
        "log_loss": ll,
        "latency_median_ms": np.median(latencies) * 1000,
        "latency_mean_ms": np.mean(latencies) * 1000,
        "latency_min_ms": np.min(latencies) * 1000,
        "latency_std_ms": np.std(latencies) * 1000,
        "probs": probs,
        "latencies": latencies,
    }


def benchmark_pytorch(X_train, y_train, X_test, y_test, n_classes, device="cpu", n_runs=N_RUNS):
    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
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
        "prediction_agreement": float(agreement),
        "max_prob_diff": float(max_diff),
        "mean_prob_diff": float(mean_diff),
    }


def run_benchmark(name, X_train, X_test, y_train, y_test, n_classes, run_mps=True):
    print(f"\n  [MLX] Running...")
    results = {}
    n_train = len(X_train)

    try:
        results["mlx"] = benchmark_mlx(X_train, y_train, X_test, y_test, n_classes)
        print(f"    Accuracy: {results['mlx']['accuracy']:.4f} | Latency: {results['mlx']['latency_median_ms']:.1f} ms")
    except Exception as e:
        print(f"    FAILED: {e}")
        results["mlx"] = None

    # Skip PyTorch for large datasets — too slow (quadratic in sample count)
    if n_train > PYTORCH_MAX_SAMPLES:
        print(f"  [PyTorch] SKIPPED (n_train={n_train} > {PYTORCH_MAX_SAMPLES} cap)")
        results["pytorch_cpu"] = None
        results["pytorch_mps"] = None
        results["pytorch_skipped"] = True
        return results

    print(f"  [PyTorch CPU] Running...")
    try:
        results["pytorch_cpu"] = benchmark_pytorch(X_train, y_train, X_test, y_test, n_classes, device="cpu")
        print(f"    Accuracy: {results['pytorch_cpu']['accuracy']:.4f} | Latency: {results['pytorch_cpu']['latency_median_ms']:.1f} ms")
    except Exception as e:
        print(f"    FAILED: {e}")
        results["pytorch_cpu"] = None

    if run_mps:
        print(f"  [PyTorch MPS] Running...")
        try:
            import torch
            if torch.backends.mps.is_available():
                results["pytorch_mps"] = benchmark_pytorch(X_train, y_train, X_test, y_test, n_classes, device="mps")
                print(f"    Accuracy: {results['pytorch_mps']['accuracy']:.4f} | Latency: {results['pytorch_mps']['latency_median_ms']:.1f} ms")
            else:
                results["pytorch_mps"] = None
        except Exception as e:
            print(f"    FAILED: {e}")
            results["pytorch_mps"] = None

    if results.get("mlx") and results.get("pytorch_cpu"):
        agreement = compute_agreement(results["mlx"]["probs"], results["pytorch_cpu"]["probs"])
        speedup_cpu = results["pytorch_cpu"]["latency_median_ms"] / results["mlx"]["latency_median_ms"]
        print(f"  [Results] Speedup: {speedup_cpu:.1f}x vs CPU | Agreement: {agreement['prediction_agreement']*100:.1f}%")
        results["agreement_mlx_cpu"] = agreement
        results["speedup_vs_cpu"] = speedup_cpu

    if results.get("mlx") and results.get("pytorch_mps"):
        speedup_mps = results["pytorch_mps"]["latency_median_ms"] / results["mlx"]["latency_median_ms"]
        results["speedup_vs_mps"] = speedup_mps

    return results


def prepare_datasets():
    datasets = {}

    # 1. Digits: 1797 samples, 64 features, 10 classes
    print("\nPreparing Digits dataset...")
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    datasets["Digits"] = {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "n_classes": 10, "n_samples": len(X), "n_features": X.shape[1],
        "description": "Handwritten digits (64 pixel features, 10 classes)",
    }

    # 2. Synthetic medium: 2000 samples, 50 features, 5 classes
    print("Preparing Synthetic-5class dataset...")
    X, y = make_classification(n_samples=2000, n_features=50, n_informative=30,
                               n_classes=5, n_clusters_per_class=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    datasets["Synthetic-5class"] = {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "n_classes": 5, "n_samples": 2000, "n_features": 50,
        "description": "Synthetic (50 features, 5 classes, 30 informative)",
    }

    # 3. Synthetic large: 5000 samples, 50 features, 5 classes
    print("Preparing Synthetic-5class-large dataset...")
    X, y = make_classification(n_samples=5000, n_features=50, n_informative=30,
                               n_classes=5, n_clusters_per_class=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    datasets["Synthetic-5class-large"] = {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "n_classes": 5, "n_samples": 5000, "n_features": 50,
        "description": "Synthetic large (50 features, 5 classes, 2500 train)",
    }

    # 4. Synthetic very large binary: 10000 samples, 100 features
    print("Preparing Synthetic-binary-10k dataset...")
    X, y = make_classification(n_samples=10000, n_features=100, n_informative=50,
                               random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    datasets["Synthetic-binary-10k"] = {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "n_classes": 2, "n_samples": 10000, "n_features": 100,
        "description": "Synthetic binary large (100 features, 5000 train)",
    }

    # 5. California Housing as 4-class classification
    print("Preparing California-Housing-4class dataset...")
    X, y_cont = fetch_california_housing(return_X_y=True)
    y = np.digitize(y_cont, np.quantile(y_cont, [0.25, 0.5, 0.75]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    datasets["CalHousing-4class"] = {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "n_classes": 4, "n_samples": len(X), "n_features": X.shape[1],
        "description": "California Housing binned into quartiles (8 features, 4 classes)",
    }

    # 6. Scaling stress test: subsample very large to test MLX at different sizes
    # Use the 10k binary dataset at different train sizes
    print("Preparing scaling stress tests...")
    X_full, y_full = make_classification(n_samples=10000, n_features=100, n_informative=50, random_state=42)
    scaling_configs = [
        (500, 500),
        (1000, 1000),
        (2000, 2000),
        (3000, 3000),
        (5000, 5000),
    ]
    datasets["_scaling_stress"] = {
        "X_full": X_full, "y_full": y_full,
        "configs": scaling_configs,
        "n_features": 100,
    }

    return datasets


def run_scaling_stress(datasets):
    print(f"\n{'='*70}")
    print("Scaling Stress Test: MLX vs PyTorch on 100-feature binary classification")
    print(f"{'='*70}")

    from tabpfn_mlx import load_v3_from_checkpoint
    from tabpfn import TabPFNClassifier

    data = datasets["_scaling_stress"]
    X_full, y_full = data["X_full"], data["y_full"]
    configs = data["configs"]

    mlx_model = load_v3_from_checkpoint(CKPT_PATH, task_type="multiclass")
    pytorch_clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)

    results = []
    for n_train, n_test in configs:
        total = n_train + n_test
        if total > len(X_full):
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_full[:total], y_full[:total], train_size=n_train, test_size=n_test, random_state=42
        )

        # MLX
        X_tr = X_train.astype(np.float32)
        y_tr = y_train.astype(np.float32)
        X_te = X_test.astype(np.float32)

        try:
            _ = mlx_model.predict_proba(X_tr, y_tr, X_te)
            mlx_times = []
            for _ in range(3):
                start = time.perf_counter()
                _ = mlx_model.predict_proba(X_tr, y_tr, X_te)
                mlx_times.append(time.perf_counter() - start)
            mlx_med = np.median(mlx_times) * 1000
            mlx_ok = True
        except Exception as e:
            print(f"  MLX failed at n_train={n_train}: {e}")
            mlx_med = None
            mlx_ok = False

        # PyTorch CPU — only for sizes within cap
        if n_train <= PYTORCH_MAX_SAMPLES:
            try:
                pytorch_clf.fit(X_train, y_train)
                _ = pytorch_clf.predict_proba(X_test)
                cpu_times = []
                for _ in range(3):
                    start = time.perf_counter()
                    _ = pytorch_clf.predict_proba(X_test)
                    cpu_times.append(time.perf_counter() - start)
                cpu_med = np.median(cpu_times) * 1000
                cpu_ok = True
            except Exception as e:
                print(f"  PyTorch CPU failed at n_train={n_train}: {e}")
                cpu_med = None
                cpu_ok = False
        else:
            cpu_med = None
            cpu_ok = False

        speedup = (cpu_med / mlx_med) if (mlx_ok and cpu_ok) else None
        row = {
            "n_train": n_train,
            "n_test": n_test,
            "n_features": 100,
            "mlx_median_ms": mlx_med,
            "cpu_median_ms": cpu_med,
            "speedup": speedup,
        }
        results.append(row)

        if mlx_ok and cpu_ok and speedup:
            print(f"  n_train={n_train:>5}, n_test={n_test:>5}: MLX {mlx_med:>8.1f} ms | CPU {cpu_med:>8.1f} ms | {speedup:.1f}x")
        elif mlx_ok:
            print(f"  n_train={n_train:>5}, n_test={n_test:>5}: MLX {mlx_med:>8.1f} ms | CPU skipped (>{PYTORCH_MAX_SAMPLES})")
        else:
            print(f"  n_train={n_train:>5}, n_test={n_test:>5}: MLX FAILED")

    return results


def main():
    import platform
    import mlx.core as mx
    import torch

    print("TabPFN v3 Expanded Benchmarks")
    print("=" * 70)
    print(f"Platform: Apple M4, macOS {platform.mac_ver()[0]}")
    print(f"MLX: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Warmup: {N_WARMUP}, Timed runs: {N_RUNS}")

    datasets = prepare_datasets()
    all_results = {}

    # Run classification benchmarks
    for name, data in datasets.items():
        if name.startswith("_"):
            continue

        print(f"\n{'='*70}")
        print(f"Dataset: {name}")
        print(f"  {data['description']}")
        print(f"  Train: {len(data['X_train'])}, Test: {len(data['X_test'])}, Features: {data['n_features']}, Classes: {data['n_classes']}")
        print(f"{'='*70}")

        # Skip MPS for very large datasets (too slow, adds no info)
        run_mps = len(data["X_train"]) <= 2500

        results = run_benchmark(
            name,
            data["X_train"], data["X_test"],
            data["y_train"], data["y_test"],
            data["n_classes"],
            run_mps=run_mps,
        )
        all_results[name] = results

    # Scaling stress test
    all_results["scaling_stress"] = run_scaling_stress(datasets)

    # Summary
    print(f"\n\n{'='*70}")
    print("EXPANDED BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Dataset':<25} | {'Train':<6} | {'Feat':<4} | {'Cls':<3} | {'MLX (ms)':<10} | {'CPU (ms)':<10} | {'Speedup':<8} | {'Agree':<7} | {'MLX Acc'}")
    print("-" * 105)

    for name in [k for k in all_results if not k.startswith("scaling")]:
        r = all_results[name]
        d = datasets[name]
        train_n = len(d["X_train"])
        n_feat = d["n_features"]
        n_cls = d["n_classes"]
        mlx_ms = f"{r['mlx']['latency_median_ms']:.0f}" if r.get("mlx") else "FAIL"
        cpu_ms = f"{r['pytorch_cpu']['latency_median_ms']:.0f}" if r.get("pytorch_cpu") else "FAIL"
        speedup = f"{r['speedup_vs_cpu']:.1f}x" if r.get("speedup_vs_cpu") else "N/A"
        agree = f"{r['agreement_mlx_cpu']['prediction_agreement']*100:.1f}%" if r.get("agreement_mlx_cpu") else "N/A"
        mlx_acc = f"{r['mlx']['accuracy']:.3f}" if r.get("mlx") else "N/A"
        print(f"{name:<25} | {train_n:<6} | {n_feat:<4} | {n_cls:<3} | {mlx_ms:<10} | {cpu_ms:<10} | {speedup:<8} | {agree:<7} | {mlx_acc}")

    # Save results
    json_results = {}
    for name, r in all_results.items():
        if name == "scaling_stress":
            json_results[name] = r
            continue
        json_results[name] = {}
        for backend in ["mlx", "pytorch_cpu", "pytorch_mps"]:
            if r.get(backend):
                json_results[name][backend] = {
                    k: v for k, v in r[backend].items()
                    if k not in ("probs", "latencies")
                }
                json_results[name][backend]["latencies"] = r[backend]["latencies"]
        for key in ["agreement_mlx_cpu", "speedup_vs_cpu", "speedup_vs_mps"]:
            if key in r:
                json_results[name][key] = r[key]
        # Add dataset metadata
        if name in datasets:
            json_results[name]["_meta"] = {
                "n_train": len(datasets[name]["X_train"]),
                "n_test": len(datasets[name]["X_test"]),
                "n_features": datasets[name]["n_features"],
                "n_classes": datasets[name]["n_classes"],
                "description": datasets[name]["description"],
            }

    output_path = "/Users/dggallit/tabpfn-v3-mlx/scripts/benchmark_expanded_results.json"
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
