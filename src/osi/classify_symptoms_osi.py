import argparse
import json
import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from dataset_loader import load_dataset, get_symptom_labels
import dspy
from classifier import SymptomClassification

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="llama3.2:3b", help="Model to use")
parser.add_argument("--ollama-host", default="http://tulkas:11434")
parser.add_argument("--split", default="test")
parser.add_argument("--symptom", type=str, default=None, help="Single symptom to classify (e.g., 'sadness')")
parser.add_argument("--output", default=None, help="Output path (default: runs/{model}_{split}_osi.json)")
parser.add_argument("--optimized-dir", default=None, help="Directory containing optimized classifiers")
args = parser.parse_args()

SYMPTOMS_INFO_PATH = "data/symptoms_info.json"

if args.output:
    OUTPUT_PATH = args.output
else:
    model_name = args.model.replace(':', '_').replace('/', '_')
    opt_suffix = "optimized" if args.optimized_dir else "baseline"
    OUTPUT_PATH = f"runs/{model_name}_{args.split}_{opt_suffix}_osi.json"

logging.info(f"Loading symptoms info from {SYMPTOMS_INFO_PATH}")
with open(SYMPTOMS_INFO_PATH) as f:
    symptoms_info = json.load(f)

logging.info(f"Loading {args.split} data")
samples = load_dataset(args.split)

logging.info(f"Configuring LM: ollama_chat/{args.model}")
lm = dspy.LM(f"ollama_chat/{args.model}", api_base=args.ollama_host, num_ctx=8192)
dspy.configure(lm=lm)

summary_data = None
optimized_dir = None
if args.optimized_dir:
    optimized_dir = Path(args.optimized_dir)
    optimized_file_path = optimized_dir / "optimized_classifiers.json"
    if optimized_file_path.exists():
        logging.info(f"Loading optimized classifier summary from {optimized_file_path}")
        with open(optimized_file_path) as f:
            summary_data = json.load(f)
    else:
        logging.warning(f"No optimized classifiers found at {optimized_file_path}")

if args.symptom:
    symptom_key = next((k for k in symptoms_info.keys() if k.lower() == args.symptom.lower()), args.symptom)
    symptoms_to_classify = {symptom_key: symptoms_info[symptom_key]}
else:
    symptoms_to_classify = symptoms_info

sentences = [s["sentence"] for s in samples]

results = []
for symptom_idx, (symptom, info) in enumerate(symptoms_to_classify.items(), 1):
    logging.info(f"[{symptom_idx}/{len(symptoms_to_classify)}] Processing: {symptom}")

    classify = dspy.Predict(SymptomClassification)

    if optimized_dir:
        classifier_path = optimized_dir / f"optimized_{symptom}.json"
        if classifier_path.exists():
            logging.info(f"Loading optimized classifier from {classifier_path}")
            classify.load(path=str(classifier_path))
        else:
            logging.warning(f"Optimized classifier not found at {classifier_path}, using baseline")
            logging.info(f"Using baseline classifier for {symptom}")
    else:
        logging.info(f"Using baseline classifier for {symptom}")

    preds = []
    for sent_idx, sentence in enumerate(sentences, 1):
        try:
            result = classify(
                symptom_name=info["pretty_name"],
                symptom_definition=info["definition"],
                text=sentence
            )
            pred = 1 if result.answer == "YES" else 0
            preds.append(pred)
        except Exception as e:
            logging.error(f"Error on sentence {sent_idx}: {e}")
            preds.append(0)

    ground_truth = get_symptom_labels(samples, symptom)
    logging.info(f"Predicted {sum(preds)}/{len(preds)} positive, GT: {sum(ground_truth)}/{len(ground_truth)}")

    predictions_with_sentences = []
    for i in range(len(sentences)):
        pred = preds[i]
        gt = ground_truth[i]

        if pred == 1 and gt == 1:
            result_type = "tp"
        elif pred == 0 and gt == 0:
            result_type = "tn"
        elif pred == 1 and gt == 0:
            result_type = "fp"
        else:
            result_type = "fn"

        predictions_with_sentences.append({
            "sentence": sentences[i],
            "prediction": pred,
            "ground_truth": gt,
            "result_type": result_type
        })

    results.append({
        "symptom": symptom,
        "predictions": preds,
        "ground_truth": ground_truth,
        "details": predictions_with_sentences
    })

logging.info(f"Saving to {OUTPUT_PATH}")
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved to {OUTPUT_PATH}")
