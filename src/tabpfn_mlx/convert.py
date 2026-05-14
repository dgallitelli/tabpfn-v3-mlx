"""Weight conversion from PyTorch checkpoints to MLX format."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path


def convert_checkpoint(weights_path: str) -> dict[str, mx.array]:
    """Convert a PyTorch nanoTabPFN checkpoint to MLX weight dict.

    Handles the QKV weight splitting (PyTorch fused in_proj → MLX separate projections).

    Args:
        weights_path: Path to .pt or .safetensors checkpoint

    Returns:
        Dictionary mapping MLX parameter names to arrays
    """
    path = Path(weights_path)

    if path.suffix == ".safetensors":
        raw_weights = mx.load(str(path))
        return _remap_keys(raw_weights)

    if path.suffix in (".pt", ".pth", ".bin"):
        import torch
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        np_weights = {k: v.numpy() for k, v in state_dict.items()}
        return _remap_keys({k: mx.array(v) for k, v in np_weights.items()})

    raise ValueError(f"Unsupported weight format: {path.suffix}")


def _remap_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap PyTorch parameter names to MLX naming convention (nanoTabPFN)."""
    mlx_weights = {}

    for key, tensor in weights.items():
        if "in_proj_weight" in key:
            d = tensor.shape[0] // 3
            base = key.replace(".in_proj_weight", "")
            mlx_weights[f"{base}.query_proj.weight"] = tensor[:d]
            mlx_weights[f"{base}.key_proj.weight"] = tensor[d : 2 * d]
            mlx_weights[f"{base}.value_proj.weight"] = tensor[2 * d :]

        elif "in_proj_bias" in key:
            d = tensor.shape[0] // 3
            base = key.replace(".in_proj_bias", "")
            mlx_weights[f"{base}.query_proj.bias"] = tensor[:d]
            mlx_weights[f"{base}.key_proj.bias"] = tensor[d : 2 * d]
            mlx_weights[f"{base}.value_proj.bias"] = tensor[2 * d :]

        else:
            mlx_weights[key] = tensor

    return mlx_weights


def load_pytorch_weights(model: nn.Module, weights_path: str) -> nn.Module:
    """Load a PyTorch checkpoint into an MLX model instance.

    Args:
        model: Initialized MLX model (NanoTabPFNMLX or TabPFNV3)
        weights_path: Path to PyTorch .pt/.safetensors checkpoint

    Returns:
        Model with loaded weights
    """
    mlx_weights = convert_checkpoint(weights_path)
    model.load_weights(list(mlx_weights.items()))
    return model


def convert_v3_checkpoint(weights_path: str) -> dict[str, mx.array]:
    """Convert a TabPFN v3 PyTorch checkpoint to MLX weight dict.

    The v3 architecture uses separate Q/K/V projections (no in_proj fusion),
    so the key remapping is primarily about naming conventions.

    Args:
        weights_path: Path to .pt or .safetensors checkpoint

    Returns:
        Dictionary mapping MLX parameter names to arrays
    """
    path = Path(weights_path)

    if path.suffix == ".safetensors":
        raw_weights = mx.load(str(path))
    elif path.suffix in (".pt", ".pth", ".bin"):
        import torch
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        raw_weights = {k: mx.array(v.numpy()) for k, v in state_dict.items()}
    else:
        raise ValueError(f"Unsupported weight format: {path.suffix}")

    return _remap_v3_keys(raw_weights)


def _remap_v3_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap TabPFN v3 PyTorch state_dict keys to MLX naming.

    Key differences:
    - PyTorch nn.ModuleList uses .N. indexing → MLX uses .layers.N.
    - PyTorch nn.Sequential uses .N. → MLX uses .layers.N.
    - RMSNorm weight → weight (same name, no remap needed)
    - nn.Parameter stored directly → needs flattened path
    """
    mlx_weights = {}

    for key, tensor in weights.items():
        mlx_key = key
        # ModuleList indexing is preserved in MLX
        # nn.Sequential layers use .layers.N. in MLX but .N. in PyTorch
        # Handle the output_projection Sequential
        if "output_projection." in key:
            # PyTorch: output_projection.0.weight → output_projection.layers.0.weight
            parts = key.split("output_projection.")
            rest = parts[1]
            if rest[0].isdigit():
                mlx_key = f"{parts[0]}output_projection.layers.{rest}"

        mlx_weights[mlx_key] = tensor

    return mlx_weights


def load_v3_pytorch_weights(model: nn.Module, weights_path: str) -> nn.Module:
    """Load a TabPFN v3 PyTorch checkpoint into a TabPFNV3 MLX model.

    Args:
        model: Initialized TabPFNV3 model
        weights_path: Path to checkpoint

    Returns:
        Model with loaded weights
    """
    mlx_weights = convert_v3_checkpoint(weights_path)
    model.load_weights(list(mlx_weights.items()))
    return model


def save_mlx_weights(model: nn.Module, output_path: str) -> None:
    """Save MLX model weights in safetensors format.

    Args:
        model: Model with weights to save
        output_path: Output .safetensors file path
    """
    weights = dict(mx.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(output_path, weights)
