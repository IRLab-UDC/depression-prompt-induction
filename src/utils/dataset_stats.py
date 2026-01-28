import argparse
import json
from pathlib import Path
from collections import Counter
from dataset_loader import load_dataset, RATIOS


def compute_split_stats(samples):
    n = len(samples)

    symptom_counts = Counter()
    positive_samples = 0
    soft_negative_samples = 0
    hard_negative_samples = 0
    positive_annotations = 0
    hard_negative_annotations = 0

    for sample in samples:
        if sample["is_control"]:
            soft_negative_samples += 1
            continue

        pos_labels = [s for s, label in sample["labels"].items() if label == 1]

        if pos_labels:
            positive_samples += 1
            positive_annotations += len(pos_labels)
            symptom_counts.update(pos_labels)
            hard_negative_annotations += len(sample["labels"]) - len(pos_labels)
        else:
            hard_negative_samples += 1
            hard_negative_annotations += len(sample["labels"])

    symptom_samples = positive_samples + hard_negative_samples
    symptom_percentages = {k: round(100 * v / symptom_samples, 2) for k, v in symptom_counts.items()} if symptom_samples > 0 else {}

    return {
        "n_samples": n,
        "positive_samples": positive_samples,
        "hard_negative_samples": hard_negative_samples,
        "soft_negative_samples": soft_negative_samples,
        "positive_annotations": positive_annotations,
        "hard_negative_annotations": hard_negative_annotations,
        "symptom_counts": dict(symptom_counts),
        "symptom_percentages": symptom_percentages,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/bdi_sen_v2")
    parser.add_argument("--output", default="results/dataset_stats.json")
    args = parser.parse_args()

    splits = {
        "train": load_dataset("train", data_dir=args.data_dir),
        "val": load_dataset("val", data_dir=args.data_dir),
        "test": load_dataset("test", data_dir=args.data_dir),
    }

    stats = {}

    for name, samples in splits.items():
        stats[name] = compute_split_stats(samples)
        stats[name]["pos_neg_ratio"] = RATIOS[name]

    all_samples = [s for samples in splits.values() for s in samples]
    stats["overall"] = compute_split_stats(all_samples)
    stats["overall"]["split_sizes"] = {name: len(samples) for name, samples in splits.items()}

    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
