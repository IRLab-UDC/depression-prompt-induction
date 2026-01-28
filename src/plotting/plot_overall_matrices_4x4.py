import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_single_cm(ax, cm, title, fontsize=24):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0, out=np.zeros_like(cm, dtype=float))
    im = ax.imshow(cm_norm * 100, cmap="Blues", vmin=0, vmax=100)
    for (r, c), val in np.ndenumerate(cm_norm):
        text_color = "white" if val > 0.5 else "#1e3a8a"
        raw_count = int(cm[r, c])
        ax.text(c, r-0.15, f"{val:.2f}", ha="center", va="center", fontsize=fontsize, color=text_color)
        ax.text(c, r+0.25, f"({raw_count})", ha="center", va="center", fontsize=fontsize-8, color=text_color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"], fontsize=fontsize-6)
    ax.set_yticklabels(["0", "1"], fontsize=fontsize-6)
    ax.set_title(title, fontsize=fontsize + 2, fontweight='bold', pad=15)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def plot_method_comparison(zero_shot_path, few_shot_path, ft_path, rcl_path, output_path):
    paths = [zero_shot_path, few_shot_path, ft_path, rcl_path]
    titles = ["0-shot", "15-shot", "FT", "RCL"]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    for i, (path, title) in enumerate(zip(paths, titles)):
        with open(path) as f:
            data = json.load(f)
        overall = data["overall_binary"]
        cm = np.array([[overall["tn"], overall["fp"]], [overall["fn"], overall["tp"]]])
        im = plot_single_cm(axes[i], cm, title)

    fig.supxlabel("Predicted", fontsize=22)
    fig.supylabel("True", fontsize=22, x=0.05)
    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels(['0.00', '0.20', '0.40', '0.60', '0.80', '1.00'])
    cbar.ax.tick_params(labelsize=16)
    cbar.outline.set_visible(False)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-shot", required=True)
    parser.add_argument("--few-shot", required=True)
    parser.add_argument("--ft", required=True)
    parser.add_argument("--rcl", required=True)
    parser.add_argument("--output", default="results/plots/overall_matrices_4x4.pdf")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_method_comparison(
        Path(args.zero_shot),
        Path(args.few_shot),
        Path(args.ft),
        Path(args.rcl),
        output_path
    )


if __name__ == "__main__":
    main()
