import json
import random
from pathlib import Path

RATIOS = {
    "train": 1.0,
    "val": 1.0,
    "test": 5.0
}

def load_dataset(split, data_dir="data/bdi_sen_v2"):
    data_dir = Path(data_dir)
    processed_path = data_dir / f"{split}.jsonl"

    if processed_path.exists():
        with open(processed_path) as f:
            return [json.loads(line) for line in f]

    path = data_dir / f"{split}-with-control.jsonl"

    with open(path) as f:
        raw_samples = [json.loads(line) for line in f]

    positives = []
    hard_negatives = []
    soft_negatives = []

    for raw in raw_samples:
        is_control = any(a.get("type") == "control" for a in raw["annotations"])

        sample = {
            "sentence": raw["sentence"],
            "is_control": is_control,
            "labels": {}
        }

        if is_control:
            soft_negatives.append(sample)
        else:
            for ann in raw["annotations"]:
                symptom = ann["symptom"]
                sample["labels"][symptom] = ann["label"]

            has_positive = any(label == 1 for label in sample["labels"].values())
            if has_positive:
                positives.append(sample)
            else:
                hard_negatives.append(sample)

    random.seed(42)
    pos_neg_ratio = RATIOS[split]

    n_positives = len(positives)
    n_negatives = int(n_positives * pos_neg_ratio)
    n_hard = n_negatives // 2
    n_soft = n_negatives - n_hard

    selected_hard = random.sample(hard_negatives, min(n_hard, len(hard_negatives)))
    selected_soft = random.sample(soft_negatives, min(n_soft, len(soft_negatives)))

    samples = positives + selected_hard + selected_soft
    random.shuffle(samples)

    with open(processed_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return samples

def get_symptom_labels(samples, symptom):
    labels = []
    for s in samples:
        if s["is_control"]:
            labels.append(0)
        else:
            labels.append(s["labels"].get(symptom, 0))
    return labels
