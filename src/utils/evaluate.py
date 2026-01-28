import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, multilabel_confusion_matrix


def compute_binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "precision": float(p), "recall": float(r), "f1": float(f1),
            "accuracy": float((tp + tn) / (tp + tn + fp + fn))}


def evaluate(results):
    all_preds, all_gt, symptom_names = [], [], []
    multilabel_preds, multilabel_gt = [], []
    per_symptom = {}

    for r in results:
        name = r["symptom"]
        symptom_names.append(name)
        preds = r["predictions"]
        gt = r["ground_truth"]
        all_preds.extend(preds)
        all_gt.extend(gt)
        multilabel_preds.append(preds)
        multilabel_gt.append(gt)
        per_symptom[name] = compute_binary_metrics(gt, preds)

    overall = compute_binary_metrics(all_gt, all_preds)

    multilabel_preds = np.array(multilabel_preds).T
    multilabel_gt = np.array(multilabel_gt).T
    p, r, f1, sup = precision_recall_fscore_support(multilabel_gt, multilabel_preds, average=None, zero_division=0)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(multilabel_gt, multilabel_preds, average="micro", zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(multilabel_gt, multilabel_preds, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(multilabel_gt, multilabel_preds, average="weighted", zero_division=0)

    multilabel = {
        "per_symptom": {name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(sup[i])}
                        for i, name in enumerate(symptom_names)},
        "micro": {"precision": float(p_micro), "recall": float(r_micro), "f1": float(f1_micro)},
        "macro": {"precision": float(p_macro), "recall": float(r_macro), "f1": float(f1_macro)},
        "weighted": {"precision": float(p_weighted), "recall": float(r_weighted), "f1": float(f1_weighted)},
    }

    mcm = multilabel_confusion_matrix(multilabel_gt, multilabel_preds)
    for i, name in enumerate(symptom_names):
        tn, fp, fn, tp = mcm[i].ravel()
        multilabel["per_symptom"][name].update({"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)})

    return {"per_symptom": per_symptom, "overall_binary": overall, "multilabel": multilabel}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", nargs="?", default="runs/google_gemma-3-4b-it_zero_shot.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    results_path = Path(args.results_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    metrics = evaluate(results)
    metrics["meta"] = {
        "results_file": str(results_path),
        "timestamp": datetime.now().isoformat(),
        "n_symptoms": len(results),
        "n_samples": len(results[0]["predictions"]) if results else 0,
    }

    output_path = output_dir / (results_path.stem + "_eval.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
