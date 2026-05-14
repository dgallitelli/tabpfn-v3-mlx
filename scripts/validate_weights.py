"""Validate MLX model against PyTorch reference implementation."""

import sys
sys.path.insert(0, "src")

import numpy as np
import torch
import mlx.core as mx
import time

from tabpfn.architectures.tabpfn_v3 import TabPFNV3 as PyTorchTabPFNV3, TabPFNV3Config as PTConfig
from tabpfn_mlx.config import TabPFNV3Config
from tabpfn_mlx.model import TabPFNV3 as MLXTabPFNV3
from tabpfn_mlx.convert import load_v3_pytorch_weights
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    ckpt_path = (
        "/Users/dggallit/.cache/huggingface/hub/models--Prior-Labs--tabpfn_3/"
        "snapshots/299726533b098357029ecac7c4fa27f97c7f9238/"
        "tabpfn-v3-classifier-v3_default.ckpt"
    )

    # Load PyTorch model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_config = ckpt["config"]
    pt_config = PTConfig(**raw_config)
    pt_model = PyTorchTabPFNV3(config=pt_config, task_type="multiclass", n_out=2)
    pt_model.load_state_dict(ckpt["state_dict"], strict=False)
    pt_model.eval()
    print("PyTorch model loaded (53M params)")

    # Load MLX model
    mlx_config = TabPFNV3Config(
        max_num_classes=160,
        icl_num_kv_heads_test=1,
        decoder_use_softmax_scaling=True,
    )
    mlx_model = MLXTabPFNV3(mlx_config)
    mlx_model = load_v3_pytorch_weights(mlx_model, ckpt_path)
    print("MLX model loaded (53M params)")

    # Breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    configs = [
        ("5-feat/30-train", X_tr[:30, :5], X_te[:10, :5], y_tr[:30], y_te[:10]),
        ("10-feat/50-train", X_tr[:50, :10], X_te[:20, :10], y_tr[:50], y_te[:20]),
        ("20-feat/100-train", X_tr[:100, :20], X_te[:30, :20], y_tr[:100], y_te[:30]),
        ("30-feat/200-train", X_tr[:200, :30], X_te[:50, :30], y_tr[:200], y_te[:50]),
    ]

    print(f"\n{'Config':<20} {'MaxDiff':<12} {'Agreement':<12} {'PT Acc':<10} {'MLX Acc'}")
    print("-" * 70)

    all_agree = True
    for name, xtr, xte, ytr, yte in configs:
        n_train = len(xtr)
        x_all = np.concatenate([xtr, xte], axis=0)

        x_pt = torch.tensor(x_all[:, None, :])
        y_pt = torch.tensor(ytr)
        with torch.no_grad():
            pt_logits = pt_model(x_pt, y_pt)
        pt_probs = torch.softmax(pt_logits[:, 0, :2], dim=-1).numpy()
        pt_preds = pt_probs.argmax(axis=1)

        x_mlx = mx.array(x_all[:, None, :])
        y_mlx = mx.array(ytr)
        mlx_logits = mlx_model(x_mlx, y_mlx)
        mx.eval(mlx_logits)
        mlx_probs = np.array(mx.softmax(mlx_logits[:, 0, :2], axis=-1))
        mlx_preds = mlx_probs.argmax(axis=1)

        max_diff = np.abs(pt_probs - mlx_probs).max()
        agree = (pt_preds == mlx_preds).mean() * 100
        pt_acc = accuracy_score(yte, pt_preds) * 100
        mlx_acc = accuracy_score(yte, mlx_preds) * 100

        if agree < 100:
            all_agree = False

        print(f"{name:<20} {max_diff:<12.6f} {agree:<12.1f} {pt_acc:<10.1f} {mlx_acc:.1f}")

    # Benchmark
    print("\n--- Timing (100 train, 20 test, 10 features) ---")
    n_train, n_test, n_feat = 100, 20, 10
    np.random.seed(42)
    x_all = np.random.randn(n_train + n_test, n_feat).astype(np.float32)
    y = np.random.randint(0, 2, size=n_train).astype(np.float32)

    x_pt = torch.tensor(x_all[:, None, :])
    y_pt = torch.tensor(y)
    x_mlx = mx.array(x_all[:, None, :])
    y_mlx = mx.array(y)

    # Warmup
    with torch.no_grad():
        pt_model(x_pt, y_pt)
    out = mlx_model(x_mlx, y_mlx)
    mx.eval(out)

    n_runs = 10
    t0 = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            pt_model(x_pt, y_pt)
    pt_ms = (time.time() - t0) / n_runs * 1000

    t0 = time.time()
    for _ in range(n_runs):
        out = mlx_model(x_mlx, y_mlx)
        mx.eval(out)
    mlx_ms = (time.time() - t0) / n_runs * 1000

    print(f"PyTorch (CPU):       {pt_ms:.0f} ms")
    print(f"MLX (Apple Silicon): {mlx_ms:.0f} ms")
    print(f"Speedup:             {pt_ms/mlx_ms:.1f}x")

    print("\n" + "=" * 70)
    if all_agree:
        print("PASS: 100% prediction agreement across all configurations.")
    else:
        print("WARN: Some predictions disagree (check above).")


if __name__ == "__main__":
    main()
