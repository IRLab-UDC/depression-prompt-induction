import argparse
import json
import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from dataset_loader import load_dataset, get_symptom_labels
import dspy
from classifier import SymptomClassification, load_custom_signatures

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--task-model", default="meta-llama/Llama-3.2-3B-Instruct")
parser.add_argument("--task-host", default="http://aragorn:30000/v1")
parser.add_argument("--split", default="test")
parser.add_argument("--symptom", type=str, default=None)
parser.add_argument("--output", default=None)
parser.add_argument("--optimized-dir", default=None)
args = parser.parse_args()

with open("data/symptoms_info.json") as f:
    symptoms_info = json.load(f)

samples = load_dataset(args.split)
lm = dspy.LM(f"openai/{args.task_model}", api_base=args.task_host, api_key="local", model_type="chat")
dspy.configure(lm=lm)

if args.symptom:
    symptom_key = next((k for k in symptoms_info.keys() if k.lower() == args.symptom.lower()), args.symptom)
    symptoms_to_classify = {symptom_key: symptoms_info[symptom_key]}
else:
    symptoms_to_classify = symptoms_info

custom_signatures = load_custom_signatures("data/si.json")
sentences = [s["sentence"] for s in samples]

optimized_dir = Path(args.optimized_dir) if args.optimized_dir else None

results = []
for symptom_idx, (symptom, info) in enumerate(symptoms_to_classify.items(), 1):
    logging.info(f"[{symptom_idx}/{len(symptoms_to_classify)}] {symptom}")

    signature_class = custom_signatures.get(symptom, SymptomClassification)
    classify = dspy.Predict(signature_class)

    if optimized_dir:
        classifier_path = optimized_dir / f"optimized_{symptom}.json"
        if classifier_path.exists():
            classify.load(path=str(classifier_path))

    preds = []
    for sentence in sentences:
        try:
            result = classify(
                symptom_name=info["pretty_name"],
                symptom_definition=info["definition"],
                text=sentence
            )
            preds.append(1 if result.answer == "YES" else 0)
        except Exception as e:
            logging.error(f"Error: {e}")
            preds.append(0)

    ground_truth = get_symptom_labels(samples, symptom)
    logging.info(f"Predicted {sum(preds)}/{len(preds)}, GT: {sum(ground_truth)}/{len(ground_truth)}")

    details = []
    for i, (sentence, pred, gt) in enumerate(zip(sentences, preds, ground_truth)):
        result_type = ["tn", "fp", "fn", "tp"][pred * 2 + gt]
        details.append({
            "sentence": sentence,
            "prediction": pred,
            "ground_truth": gt,
            "result_type": result_type
        })

    results.append({
        "symptom": symptom,
        "predictions": preds,
        "ground_truth": ground_truth,
        "details": details
    })

if args.output:
    output_path = args.output
else:
    model_name = args.task_model.replace(':', '_').replace('/', '_')
    opt_suffix = "optimized" if args.optimized_dir else "baseline"
    output_path = f"runs/{model_name}_{args.split}_{opt_suffix}_osi.json"

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved to {output_path}")
