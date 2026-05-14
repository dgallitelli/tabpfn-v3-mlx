#!/usr/bin/env python3
"""Convert a TabPFN v3 PyTorch checkpoint to MLX safetensors format.

Usage:
    python scripts/convert_weights.py path/to/checkpoint.pt -o weights/model.safetensors
"""

import argparse
from pathlib import Path

import mlx.core as mx

from tabpfn_mlx.convert import convert_v3_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Convert TabPFN v3 PyTorch weights to MLX")
    parser.add_argument("input", type=str, help="Path to PyTorch .pt or .safetensors checkpoint")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output .safetensors path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {input_path}")

    output_path = args.output or str(input_path.with_suffix(".mlx.safetensors"))

    print(f"Converting: {input_path} -> {output_path}")
    weights = convert_v3_checkpoint(str(input_path))
    mx.save_safetensors(output_path, weights)
    print(f"Saved {len(weights)} tensors to {output_path}")


if __name__ == "__main__":
    main()
