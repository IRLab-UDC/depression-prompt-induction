import argparse
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))
from utils.dataset_loader import load_dataset

SYMPTOM_COLS = [
    "Sadness", "Pessimism", "Sense_of_failure", "Loss_of_Pleasure", "Guilty_feelings",
    "Sense_of_punishment", "Self-dislike", "Self-incrimination", "Suicidal_ideas",
    "Crying", "Agitation", "Social_withdrawal", "Indecision", "Feelings_of_worthlessness",
    "Loss_of_energy", "Change_of_sleep", "Irritability", "Changes_in_appetite",
    "Concentration_difficulty", "Tiredness_or_fatigue", "Loss_of_interest_in_sex"
]


def plot_symptom_distribution(data_dir, output_path):
    with open("data/symptoms_info.json") as f:
        symptoms_info = json.load(f)

    splits = {
        "train": load_dataset("train", data_dir=data_dir),
        "val": load_dataset("val", data_dir=data_dir),
        "test": load_dataset("test", data_dir=data_dir),
    }

    split_counts = {}
    for name, samples in splits.items():
        counts = Counter()
        for sample in samples:
            if not sample["is_control"]:
                for symptom, label in sample["labels"].items():
                    if label == 1:
                        counts[symptom] += 1
        split_counts[name] = counts

    total_counts = Counter()
    for counts in split_counts.values():
        total_counts.update(counts)

    sorted_symptoms = [s for s, _ in total_counts.most_common()]

    train_sorted = [split_counts["train"].get(s, 0) for s in sorted_symptoms]
    val_sorted = [split_counts["val"].get(s, 0) for s in sorted_symptoms]
    test_sorted = [split_counts["test"].get(s, 0) for s in sorted_symptoms]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(sorted_symptoms))

    ax.bar(x, train_sorted, label="Train", color="#1e3a8a")
    ax.bar(x, val_sorted, bottom=train_sorted, label="Val", color="#3b82f6")
    ax.bar(x, test_sorted, bottom=[t+v for t, v in zip(train_sorted, val_sorted)], label="Test", color="#93c5fd")

    for i, (train, val, test) in enumerate(zip(train_sorted, val_sorted, test_sorted)):
        if train >= 15:
            ax.text(i, train/2, str(int(train)), ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        if val >= 15:
            ax.text(i, train + val/2, str(int(val)), ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        if test >= 15:
            ax.text(i, train + val + test/2, str(int(test)), ha="center", va="center", fontsize=10, color="#1e3a8a", fontweight="bold")

    max_count = max(total_counts.values())
    for i, symptom in enumerate(sorted_symptoms):
        total = train_sorted[i] + val_sorted[i] + test_sorted[i]
        pretty_name = symptoms_info[symptom]["pretty_name"]
        ax.text(i, total + max_count * 0.02, pretty_name,
                ha="left", va="bottom", fontsize=10, rotation=45)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, len(sorted_symptoms) - 0.5)
    ax.legend(fontsize=14, frameon=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/bdi_sen_v2")
    parser.add_argument("--output", default="results/plots/bdisen_symptom_distribution.pdf")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_symptom_distribution(data_dir, output_path)


if __name__ == "__main__":
    main()
