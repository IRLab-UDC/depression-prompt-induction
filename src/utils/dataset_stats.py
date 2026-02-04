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


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_per_disease_stats(data_dir):
    splits = {
        "train": load_jsonl(Path(data_dir) / "train.jsonl"),
        "val": load_jsonl(Path(data_dir) / "val.jsonl"),
        "test": load_jsonl(Path(data_dir) / "test.jsonl"),
    }

    diseases = set()
    for samples in splits.values():
        for sample in samples:
            if "diseases" in sample:
                diseases.update(sample["diseases"])

    diseases = sorted(diseases - {"control"})

    stats = {}

    for disease in diseases:
        disease_stats = {}

        for split_name, samples in splits.items():
            disease_samples = [s for s in samples if "diseases" in s and disease in s["diseases"]]
            control_samples = [s for s in samples if "diseases" in s and "control" in s["diseases"]]

            combined = disease_samples + control_samples
            disease_stats[split_name] = compute_split_stats(combined)
            disease_stats[split_name]["pos_neg_ratio"] = len(control_samples) / len(disease_samples) if disease_samples else 0

        stats[disease] = disease_stats

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dirs", nargs="+", default=["data/bdi_sen_v2", "data/psysym"])
    parser.add_argument("--output", default="results/dataset_stats.json")
    args = parser.parse_args()

    all_stats = {}

    for data_dir in args.data_dirs:
        dataset_name = Path(data_dir).name

        if dataset_name == "psysym":
            all_stats[dataset_name] = compute_per_disease_stats(data_dir)
        else:
            splits = {
                "train": load_dataset("train", data_dir=data_dir),
                "val": load_dataset("val", data_dir=data_dir),
                "test": load_dataset("test", data_dir=data_dir),
            }

            stats = {}

            for name, samples in splits.items():
                stats[name] = compute_split_stats(samples)
                stats[name]["pos_neg_ratio"] = RATIOS[name]

            all_samples = [s for samples in splits.values() for s in samples]
            stats["overall"] = compute_split_stats(all_samples)
            stats["overall"]["split_sizes"] = {name: len(samples) for name, samples in splits.items()}

            all_stats[dataset_name] = {"overall": stats}

    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_stats, f, indent=2)


if __name__ == "__main__":
    main()
