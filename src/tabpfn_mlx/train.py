"""Fine-tuning support for TabPFN v3 on MLX."""

from __future__ import annotations

from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


class TabPFNDataset:
    """Dataset wrapper for fine-tuning TabPFN.

    Wraps multiple (X, y) numpy arrays and handles sampling + splitting
    into context/query pairs during training.
    """

    def __init__(self, datasets: list[tuple[np.ndarray, np.ndarray]]):
        """Initialize with a list of (X, y) tuples.

        Args:
            datasets: List of (features, labels) pairs. Each X is (N, C),
                each y is (N,) with integer class labels.
        """
        self.datasets = datasets
        self._rng = np.random.default_rng(42)

    def sample_batch(
        self,
        batch_size: int,
        context_size: int,
        query_size: int,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Sample a batch of context/query splits.

        Returns:
            List of (X_context, y_context, X_query, y_query) tuples.
        """
        batch = []
        for _ in range(batch_size):
            idx = self._rng.integers(0, len(self.datasets))
            X, y = self.datasets[idx]
            n = len(X)

            total_needed = context_size + query_size
            if n < total_needed:
                ctx_n = max(2, int(n * context_size / total_needed))
                qry_n = n - ctx_n
            else:
                ctx_n = context_size
                qry_n = query_size

            perm = self._rng.permutation(n)
            ctx_idx = perm[:ctx_n]
            qry_idx = perm[ctx_n : ctx_n + qry_n]

            batch.append((
                X[ctx_idx].astype(np.float32),
                y[ctx_idx].astype(np.float32),
                X[qry_idx].astype(np.float32),
                y[qry_idx].astype(np.int32),
            ))
        return batch

    def __len__(self) -> int:
        return len(self.datasets)


def freeze_layers(model, n_layers: int) -> None:
    """Freeze the first n ICL transformer layers for transfer learning.

    Frozen parameters won't receive gradients during fine-tuning.
    """
    model.freeze()
    # Unfreeze everything except the first n ICL blocks
    model.unfreeze()
    for i in range(min(n_layers, len(model.icl_blocks))):
        model.icl_blocks[i].freeze()


def unfreeze_all(model) -> None:
    """Unfreeze all model parameters."""
    model.unfreeze()


def _compute_loss(
    model,
    X_context: mx.array,
    y_context: mx.array,
    X_query: mx.array,
    y_query: mx.array,
    n_classes: int,
) -> mx.array:
    """Compute cross-entropy loss for a single context/query pair."""
    # Build combined input: (R, 1, C) rows-first, batch=1
    x_combined = mx.concatenate([X_context, X_query], axis=0)
    x_tensor = x_combined[:, None, :]  # (R, 1, C)

    logits = model(x_tensor, y_context)  # (M, 1, T)

    # Slice to actual classes and squeeze batch
    logits_query = logits[:, 0, :n_classes]  # (M, n_classes)

    # Cross-entropy loss
    log_probs = mx.log(mx.softmax(logits_query, axis=-1) + 1e-8)
    targets = y_query.astype(mx.int32)
    loss = -mx.mean(log_probs[mx.arange(targets.shape[0]), targets])
    return loss


def fine_tune(
    model,
    train_datasets: list[tuple[np.ndarray, np.ndarray]],
    *,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    batch_size: int = 4,
    context_size: int = 128,
    query_size: int = 32,
    n_freeze_layers: int = 0,
    callback: Callable[[dict], bool] | None = None,
    seed: int = 42,
) -> dict:
    """Fine-tune TabPFN on domain-specific datasets using ICL objective.

    Training objective: given a context set (train), predict the query set
    (test) labels via the model's in-context learning mechanism.

    Args:
        model: TabPFNV3 model instance.
        train_datasets: List of (X, y) pairs for training.
        epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        batch_size: Number of datasets to sample per step.
        context_size: Number of context (train) rows per sample.
        query_size: Number of query (test) rows per sample.
        n_freeze_layers: Freeze the first N ICL layers (transfer learning).
        callback: Called after each epoch with metrics dict. Return True to
            stop early.
        seed: Random seed for reproducibility.

    Returns:
        Dict with training history: {"losses": [...], "epochs_completed": int}
    """
    dataset = TabPFNDataset(train_datasets)
    dataset._rng = np.random.default_rng(seed)

    # Determine n_classes from data
    all_classes = set()
    for _, y in train_datasets:
        all_classes.update(y.astype(int).tolist())
    n_classes = len(all_classes)

    # Freeze layers if requested
    if n_freeze_layers > 0:
        freeze_layers(model, n_freeze_layers)

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

    # Build loss + grad function
    loss_and_grad_fn = nn.value_and_grad(model, _loss_batch)

    history = {"losses": [], "epochs_completed": 0}
    steps_per_epoch = max(1, len(train_datasets) // batch_size)

    for epoch in range(epochs):
        epoch_losses = []

        for _ in range(steps_per_epoch):
            batch = dataset.sample_batch(batch_size, context_size, query_size)

            loss_val, grads = loss_and_grad_fn(
                model, batch, n_classes
            )
            mx.eval(loss_val)

            optimizer.update(model, grads)
            mx.eval(model.parameters())

            epoch_losses.append(float(loss_val))

        avg_loss = float(np.mean(epoch_losses))
        history["losses"].append(avg_loss)
        history["epochs_completed"] = epoch + 1

        if callback is not None:
            should_stop = callback({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "history": history,
            })
            if should_stop:
                break

    # Unfreeze all after training
    unfreeze_all(model)
    return history


def _loss_batch(
    model,
    batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    n_classes: int,
) -> mx.array:
    """Compute average loss over a batch of context/query pairs."""
    losses = []
    for X_ctx, y_ctx, X_qry, y_qry in batch:
        loss = _compute_loss(
            model,
            mx.array(X_ctx),
            mx.array(y_ctx),
            mx.array(X_qry),
            mx.array(y_qry),
            n_classes,
        )
        losses.append(loss)
    return mx.mean(mx.stack(losses))
