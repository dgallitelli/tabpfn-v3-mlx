#!/usr/bin/env python3
"""Generate publication-quality benchmark figure."""

import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ["Iris\n(75)", "Wine\n(89)", "Breast\nCancer\n(284)", "Digits\n(898)", "Synthetic\n5-class\n(1000)", "Sine\nRegression\n(150)"]
mlx_ms = [22, 29, 135, 838, 571, 21]
cpu_ms = [636, 808, 3062, 10755, 9147, 512]
mps_ms = [863, 977, 4121, 10077, None, None]

# Colors
c_mlx = "#0071e3"  # Apple blue
c_cpu = "#e05c44"  # Warm red
c_mps = "#f5a623"  # Amber

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [3, 2]})
fig.patch.set_facecolor("white")

# --- Left panel: Latency comparison (log scale) ---
ax = axes[0]
x = np.arange(len(datasets))
width = 0.25

bars_mlx = ax.bar(x - width, mlx_ms, width, label="MLX (Apple M4)", color=c_mlx, edgecolor="white", linewidth=0.5)
bars_cpu = ax.bar(x, cpu_ms, width, label="PyTorch CPU", color=c_cpu, edgecolor="white", linewidth=0.5)

mps_vals = [v if v is not None else 0 for v in mps_ms]
mps_mask = [v is not None for v in mps_ms]
bars_mps = ax.bar(x[mps_mask] + width, [mps_vals[i] for i in range(len(mps_vals)) if mps_mask[i]],
                  width, label="PyTorch MPS", color=c_mps, edgecolor="white", linewidth=0.5)

# OOM markers
for i, v in enumerate(mps_ms):
    if v is None:
        ax.annotate("OOM", (x[i] + width, max(cpu_ms[i] * 0.6, 300)),
                    ha="center", va="bottom", fontsize=7, color=c_mps, fontweight="bold")

ax.set_yscale("log")
ax.set_ylabel("Latency (ms, log scale)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=9)
ax.set_title("Inference Latency: MLX vs PyTorch", fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.set_ylim(10, 20000)
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Speedup annotations
for i in range(len(datasets)):
    speedup = cpu_ms[i] / mlx_ms[i]
    ax.annotate(f"{speedup:.0f}×", (x[i] - width, mlx_ms[i]),
                ha="center", va="bottom", fontsize=8, fontweight="bold", color=c_mlx)

# --- Right panel: Speedup factor chart ---
ax2 = axes[1]
speedups = [cpu_ms[i] / mlx_ms[i] for i in range(len(datasets))]
short_names = ["Iris", "Wine", "Breast\nCancer", "Digits", "Synthetic\n5-class", "Sine\nRegression"]

bars = ax2.barh(range(len(short_names)), speedups, color=c_mlx, edgecolor="white", linewidth=0.5, height=0.6)
ax2.set_yticks(range(len(short_names)))
ax2.set_yticklabels(short_names, fontsize=9)
ax2.set_xlabel("Speedup Factor (×)", fontsize=11)
ax2.set_title("MLX Speedup over PyTorch CPU", fontsize=13, fontweight="bold", pad=12)
ax2.axvline(x=1, color="gray", linewidth=0.5, linestyle="--")
ax2.set_xlim(0, 35)
ax2.grid(axis="x", alpha=0.3, linewidth=0.5)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

for i, v in enumerate(speedups):
    ax2.text(v + 0.5, i, f"{v:.1f}×", va="center", fontsize=9, fontweight="bold", color=c_mlx)

plt.tight_layout(pad=2)

# Save
plt.savefig("/Users/dggallit/tabpfn-v3-mlx/docs/benchmark_figure.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("/Users/dggallit/tabpfn-v3-mlx/docs/benchmark_figure.svg", bbox_inches="tight", facecolor="white")
print("Saved: docs/benchmark_figure.png (300 DPI) + docs/benchmark_figure.svg")
