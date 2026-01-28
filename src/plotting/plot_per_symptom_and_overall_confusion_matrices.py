import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SYMPTOMS_INFO_PATH = "data/BDI-Sen-full-dataset/symptoms_info.json"


def plot_single_cm(ax, cm, title, fontsize=8):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, where=row_sums != 0, out=np.zeros_like(cm, dtype=float)) * 100
    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    for (r, c), val in np.ndenumerate(cm_pct):
        ax.text(c, r, f"{val:.0f}%", ha="center", va="center", fontsize=fontsize)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"], fontsize=fontsize - 1)
    ax.set_yticklabels(["0", "1"], fontsize=fontsize - 1)
    ax.set_title(title, fontsize=fontsize)


def plot_confusion_matrices(results_path, output_dir):
    with open(results_path) as f:
        data = json.load(f)
    with open(SYMPTOMS_INFO_PATH) as f:
        symptoms_info = json.load(f)

    per_symptom = data["multilabel"]["per_symptom"]
    symptoms = list(per_symptom.keys())
    overall = data["overall_binary"]
    stem = results_path.stem

    fig, axes = plt.subplots(3, 7, figsize=(16, 7))
    axes = axes.flatten()
    for i, symptom in enumerate(symptoms):
        m = per_symptom[symptom]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        plot_single_cm(axes[i], cm, symptoms_info[symptom]["pretty_name"])
    fig.supxlabel("Predicted", fontsize=10)
    fig.supylabel("Actual", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_confusion_matrices.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(3, 3))
    cm_overall = np.array([[overall["tn"], overall["fp"]], [overall["fn"], overall["tp"]]])
    plot_single_cm(ax, cm_overall, "Overall", fontsize=10)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_confusion_matrix_overall.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path")
    parser.add_argument("--output-dir", default="results/plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrices(Path(args.results_path), output_dir)


if __name__ == "__main__":
    main()
