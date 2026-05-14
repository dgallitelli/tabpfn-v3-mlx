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


def convert_v3_checkpoint(weights_path: str) -> tuple[dict[str, mx.array], dict]:
    """Convert a TabPFN v3 PyTorch checkpoint to MLX weight dict.

    Handles .ckpt (PyTorch Lightning), .pt, and .safetensors formats.
    Extracts the state_dict and config from .ckpt files.

    Args:
        weights_path: Path to checkpoint file

    Returns:
        Tuple of (mlx_weights dict, config dict from checkpoint)
    """
    path = Path(weights_path)

    config = {}

    if path.suffix == ".safetensors":
        raw_weights = mx.load(str(path))
    elif path.suffix in (".pt", ".pth", ".bin", ".ckpt"):
        import torch
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            config = ckpt.get("config", {})
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        raw_weights = {k: mx.array(v.numpy()) for k, v in state_dict.items()}
    else:
        raise ValueError(f"Unsupported weight format: {path.suffix}")

    return _remap_v3_keys(raw_weights), config


def _remap_v3_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap TabPFN v3 PyTorch state_dict keys to MLX naming.

    Transformations:
    - x_embed.weight/bias → x_embed.linear.weight/bias
    - col_y_encoder.embedding → col_y_encoder.encoder.embedding
    - icl_y_encoder.embedding → icl_y_encoder.encoder.embedding
    - rope.freqs → rope.inv_freq
    - .mlp.0. → .mlp.linear1., .mlp.2. → .mlp.linear2.
    - .base_mlp.0. → .base_linear1., .base_mlp.2. → .base_linear2.
    - .query_mlp.0. → .query_linear1., .query_mlp.2. → .query_linear2.
    - regression_borders is dropped (buffer, not a model parameter)
    """
    mlx_weights = {}

    for key, tensor in weights.items():
        if key == "regression_borders":
            continue

        k = key

        if k.startswith("x_embed.") and "linear" not in k:
            k = k.replace("x_embed.", "x_embed.linear.")

        if k.startswith("col_y_encoder.embedding."):
            k = k.replace("col_y_encoder.embedding.", "col_y_encoder.encoder.embedding.")

        if k.startswith("icl_y_encoder.embedding."):
            k = k.replace("icl_y_encoder.embedding.", "icl_y_encoder.encoder.embedding.")

        if "rope.freqs" in k:
            k = k.replace("rope.freqs", "rope.inv_freq")

        if ".mlp.0." in k:
            k = k.replace(".mlp.0.", ".mlp.linear1.")
        if ".mlp.2." in k:
            k = k.replace(".mlp.2.", ".mlp.linear2.")

        if ".base_mlp.0." in k:
            k = k.replace(".base_mlp.0.", ".base_linear1.")
        if ".base_mlp.2." in k:
            k = k.replace(".base_mlp.2.", ".base_linear2.")

        if ".query_mlp.0." in k:
            k = k.replace(".query_mlp.0.", ".query_linear1.")
        if ".query_mlp.2." in k:
            k = k.replace(".query_mlp.2.", ".query_linear2.")

        mlx_weights[k] = tensor

    return mlx_weights


def load_v3_pytorch_weights(model: nn.Module, weights_path: str) -> nn.Module:
    """Load a TabPFN v3 PyTorch checkpoint into a TabPFNV3 MLX model.

    Args:
        model: Initialized TabPFNV3 model
        weights_path: Path to checkpoint

    Returns:
        Model with loaded weights
    """
    mlx_weights, _ = convert_v3_checkpoint(weights_path)
    model.load_weights(list(mlx_weights.items()))
    return model


def load_v3_from_checkpoint(weights_path: str, task_type: str = "multiclass"):
    """Create and load a TabPFNV3 model directly from a checkpoint.

    Reads config from the checkpoint and creates the model with correct settings.

    Args:
        weights_path: Path to .ckpt checkpoint
        task_type: "multiclass" or "regression"

    Returns:
        Loaded TabPFNV3 model ready for inference
    """
    from tabpfn_mlx.config import TabPFNV3Config
    from tabpfn_mlx.model import TabPFNV3

    mlx_weights, ckpt_config = convert_v3_checkpoint(weights_path)

    config_kwargs = {}
    config_fields = {f.name for f in TabPFNV3Config.__dataclass_fields__.values()}
    for k, v in ckpt_config.items():
        if k in config_fields:
            config_kwargs[k] = v

    config = TabPFNV3Config(**config_kwargs)
    model = TabPFNV3(config, task_type=task_type)
    model.load_weights(list(mlx_weights.items()))
    return model


def save_mlx_weights(model: nn.Module, output_path: str) -> None:
    """Save MLX model weights in safetensors format.

    Args:
        model: Model with weights to save
        output_path: Output .safetensors file path
    """
    weights = dict(nn.utils.tree_flatten(model.parameters()))
    mx.save_safetensors(output_path, weights)
