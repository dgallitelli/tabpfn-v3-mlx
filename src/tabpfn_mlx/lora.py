"""LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning of TabPFN v3."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    Computes: y = Wx + (alpha/r) * B @ A @ x
    where A is (r, in) and B is (out, r), initialized so BA = 0 at start.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        in_features = base_layer.weight.shape[1]
        out_features = base_layer.weight.shape[0]

        self.base_weight = base_layer.weight
        self.base_bias = base_layer.bias if hasattr(base_layer, "bias") and base_layer.bias is not None else None
        self.rank = rank
        self.scale = alpha / rank

        # A: (rank, in_features) — normal init
        self.lora_A = mx.random.normal((rank, in_features)) * 0.01
        # B: (out_features, rank) — zero init so LoRA starts as identity
        self.lora_B = mx.zeros((out_features, rank))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        # Base linear
        out = x @ self.base_weight.T
        if self.base_bias is not None:
            out = out + self.base_bias

        # LoRA path
        lora_input = x
        if self.dropout is not None:
            lora_input = self.dropout(lora_input)
        lora_out = (lora_input @ self.lora_A.T) @ self.lora_B.T
        out = out + self.scale * lora_out
        return out


def apply_lora(
    model,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> dict[str, LoRALinear]:
    """Apply LoRA adapters to a TabPFN v3 model.

    Replaces targeted nn.Linear layers with LoRALinear wrappers, freezing
    base weights and only training the low-rank adapters.

    Args:
        model: TabPFNV3 model instance.
        rank: LoRA rank (typically 4-32).
        alpha: Scaling factor (typically 2*rank).
        dropout: Dropout on LoRA path (0 = disabled).
        target_modules: List of module name patterns to adapt.
            Default: ["q_projection", "v_projection"] (attention QV projections).

    Returns:
        Dict mapping module paths to their LoRALinear instances.
    """
    if target_modules is None:
        target_modules = ["q_projection", "v_projection"]

    # Freeze entire model first
    model.freeze()

    lora_layers = {}

    # Apply to ICL blocks (the main transformer stack)
    for block_idx, block in enumerate(model.icl_blocks):
        attn = block.icl_attention
        for name in target_modules:
            layer = getattr(attn, name, None)
            if layer is not None and isinstance(layer, nn.Linear):
                lora_layer = LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout)
                setattr(attn, name, lora_layer)
                path = f"icl_blocks.{block_idx}.icl_attention.{name}"
                lora_layers[path] = lora_layer

    # Unfreeze only LoRA parameters
    for lora_layer in lora_layers.values():
        lora_layer.unfreeze()

    return lora_layers


def merge_lora(model, lora_layers: dict[str, LoRALinear]) -> None:
    """Merge LoRA weights back into base weights for deployment.

    After merging, the model runs at full speed with no LoRA overhead.
    The LoRALinear layers are replaced with standard nn.Linear layers.
    """
    for path, lora_layer in lora_layers.items():
        # Compute merged weight: W' = W + scale * B @ A
        merged_weight = lora_layer.base_weight + lora_layer.scale * (
            lora_layer.lora_B @ lora_layer.lora_A
        )

        # Create a new nn.Linear with merged weights
        out_features, in_features = merged_weight.shape
        has_bias = lora_layer.base_bias is not None
        new_linear = nn.Linear(in_features, out_features, bias=has_bias)
        new_linear.weight = merged_weight
        if has_bias:
            new_linear.bias = lora_layer.base_bias

        # Navigate to parent and replace
        parts = path.split(".")
        obj = model
        for part in parts[:-1]:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        setattr(obj, parts[-1], new_linear)

    # Unfreeze entire model
    model.unfreeze()


def count_lora_params(lora_layers: dict[str, LoRALinear]) -> dict[str, int]:
    """Count trainable vs total parameters.

    Returns:
        Dict with "trainable", "frozen", "total", "trainable_pct" keys.
    """
    trainable = 0
    for lora_layer in lora_layers.values():
        trainable += lora_layer.lora_A.size + lora_layer.lora_B.size

    return {
        "trainable": trainable,
        "lora_layers": len(lora_layers),
    }


def lora_fine_tune(
    model,
    train_datasets: list[tuple],
    *,
    rank: int = 8,
    alpha: float = 16.0,
    epochs: int = 10,
    lr: float = 1e-4,
    context_size: int = 128,
    query_size: int = 32,
    target_modules: list[str] | None = None,
    merge_after: bool = True,
    **kwargs,
) -> dict:
    """Convenience function: apply LoRA + fine-tune + optionally merge.

    Args:
        model: TabPFNV3 model instance.
        train_datasets: List of (X, y) pairs.
        rank: LoRA rank.
        alpha: LoRA alpha scaling.
        epochs: Training epochs.
        lr: Learning rate.
        context_size: Context rows per sample.
        query_size: Query rows per sample.
        target_modules: Modules to adapt (default: Q/V projections).
        merge_after: If True, merge LoRA weights into base after training.
        **kwargs: Additional arguments passed to fine_tune().

    Returns:
        Training history dict.
    """
    from tabpfn_mlx.train import fine_tune

    lora_layers = apply_lora(
        model, rank=rank, alpha=alpha, target_modules=target_modules
    )

    history = fine_tune(
        model,
        train_datasets,
        epochs=epochs,
        lr=lr,
        context_size=context_size,
        query_size=query_size,
        **kwargs,
    )

    if merge_after:
        merge_lora(model, lora_layers)

    history["lora_layers"] = len(lora_layers)
    history["lora_rank"] = rank
    return history
