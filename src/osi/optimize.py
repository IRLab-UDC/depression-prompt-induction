import argparse
import json
import sys
import random
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from dataset_loader import load_dataset
import dspy
from dspy.teleprompt import MIPROv2
from classifier import SymptomClassification
from metrics import classification_metric, weighted_classification_metric

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--task-model", default="llama3.2:3b", help="Model for task execution")
parser.add_argument("--prompt-model", default="phi4", help="Model for prompt optimization")
parser.add_argument("--train-size", type=int, default=100)
parser.add_argument("--output", default="optimized_classifiers.json", help="Single file containing all optimized classifiers")
parser.add_argument("--ollama-host", default="http://tulkas:11434")
parser.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
parser.add_argument("--num-threads", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--symptom", type=str, default=None, help="Single symptom to optimize (e.g., 'sadness')")
parser.add_argument("--val-size", type=int, default=25, help="Validation examples per class")
parser.add_argument("--metric", default="weighted", choices=["accuracy", "weighted"], help="Metric to use for optimization")
args = parser.parse_args()

if args.symptom:
    logging.info(f"Running for single symptom: {args.symptom}")

random.seed(args.seed)

SYMPTOMS_INFO_PATH = "data/symptoms_info.json"

logging.info(f"Loading symptoms info from {SYMPTOMS_INFO_PATH}")
with open(SYMPTOMS_INFO_PATH) as f:
    symptoms_info = json.load(f)

logging.info(f"Loading training data")
train_samples = load_dataset("train")

logging.info(f"Loading validation data")
val_samples = load_dataset("val")

logging.info(f"Configuring task model: ollama_chat/{args.task_model}")
task_lm = dspy.LM(f"ollama_chat/{args.task_model}", api_base=args.ollama_host, num_ctx=8192)
dspy.configure(lm=task_lm)

logging.info(f"Configuring prompt model: ollama_chat/{args.prompt_model}")
prompt_lm = dspy.LM(f"ollama_chat/{args.prompt_model}", api_base=args.ollama_host, num_ctx=8192)

symptoms_to_optimize = (
    {args.symptom: symptoms_info[args.symptom]}
    if args.symptom else symptoms_info
)

optimized_results = {}

for symptom, info in symptoms_to_optimize.items():
    print(f"\n{'='*60}")
    print(f"Optimizing: {symptom}")
    print(f"{'='*60}")

    train_pos = [s["sentence"] for s in train_samples if s["labels"].get(symptom, 0) == 1]
    train_hard_neg = [s["sentence"] for s in train_samples if not s["is_control"] and s["labels"].get(symptom, 0) == 0 and any(v == 1 for v in s["labels"].values())]
    train_soft_neg = [s["sentence"] for s in train_samples if s["is_control"]]

    trainset = []
    for s in train_pos[:args.train_size // 2]:
        trainset.append(
            dspy.Example(
                symptom_name=info["pretty_name"],
                symptom_definition=info["definition"],
                text=s,
                answer="YES"
            ).with_inputs("symptom_name", "symptom_definition", "text")
        )

    n_hard = len(train_pos[:args.train_size // 2]) // 2
    n_soft = len(train_pos[:args.train_size // 2]) - n_hard
    for s in train_hard_neg[:n_hard]:
        trainset.append(
            dspy.Example(
                symptom_name=info["pretty_name"],
                symptom_definition=info["definition"],
                text=s,
                answer="NO"
            ).with_inputs("symptom_name", "symptom_definition", "text")
        )
    for s in train_soft_neg[:n_soft]:
        trainset.append(
            dspy.Example(
                symptom_name=info["pretty_name"],
                symptom_definition=info["definition"],
                text=s,
                answer="NO"
            ).with_inputs("symptom_name", "symptom_definition", "text")
        )

    random.shuffle(trainset)

    # Split validation into optimization set and evaluation set
    val_pos = [s["sentence"] for s in val_samples if s["labels"].get(symptom, 0) == 1]
    val_neg = [s["sentence"] for s in val_samples if s["labels"].get(symptom, 0) == 0]

    # Use half for optimization, half for final evaluation
    opt_size = args.val_size // 2
    eval_size = args.val_size - opt_size

    optset = []
    evalset = []

    # Optimization set (used by MIPROv2)
    for s in val_pos[:opt_size]:
        optset.append(
            dspy.Example(
                symptom_name=info["pretty_name"],
                symptom_definition=info["definition"],
                text=s,
                answer="YES"
            ).with_inputs("symptom_name", "symptom_definition", "text")
        )
    for s in val_neg[:opt_size]:
        optset.append(
            dspy.Example(
                symptom_name=info["pretty_name"],
                symptom_definition=info["definition"],
                text=s,
                answer="NO"
            ).with_inputs("symptom_name", "symptom_definition", "text")
        )

    # Evaluation set (held-out for final scoring)
    for s in val_pos[opt_size:opt_size+eval_size]:
        evalset.append(
            dspy.Example(
                symptom_name=info["pretty_name"],
                symptom_definition=info["definition"],
                text=s,
                answer="YES"
            ).with_inputs("symptom_name", "symptom_definition", "text")
        )
    for s in val_neg[opt_size:opt_size+eval_size]:
        evalset.append(
            dspy.Example(
                symptom_name=info["pretty_name"],
                symptom_definition=info["definition"],
                text=s,
                answer="NO"
            ).with_inputs("symptom_name", "symptom_definition", "text")
        )

    random.shuffle(optset)
    random.shuffle(evalset)

    print(f"Train: {len(trainset)} examples")
    print(f"Optimization: {len(optset)} examples (used by MIPROv2)")
    print(f"Evaluation: {len(evalset)} examples (held-out)")

    classify = dspy.Predict(SymptomClassification)

    metric_fn = weighted_classification_metric if args.metric == "weighted" else classification_metric
    logging.info(f"Using metric: {args.metric}")

    optimizer_kwargs = {
        "metric": metric_fn,
        "auto": args.auto,
        "num_threads": args.num_threads,
        "prompt_model": prompt_lm,
        "task_model": task_lm,
        "seed": args.seed,
        "verbose": True,
    }

    optimizer = MIPROv2(**optimizer_kwargs)

    logging.info("Starting optimization...")
    optimized_classify = optimizer.compile(
        classify,
        trainset=trainset,
        valset=optset,  # Use optimization set only
    )

    logging.info("Evaluating optimized classifier on held-out evaluation set")
    opt_scores = []
    for ex in evalset:  # Evaluate on held-out set
        pred = optimized_classify(
            symptom_name=ex.symptom_name,
            symptom_definition=ex.symptom_definition,
            text=ex.text
        )
        score = metric_fn(ex, pred, trace=None)
        opt_scores.append(score)
    opt_score = sum(opt_scores) / len(opt_scores)

    logging.info("Evaluating baseline classifier on held-out evaluation set")
    baseline_classify = dspy.Predict(SymptomClassification)
    base_scores = []
    for ex in evalset:  # Evaluate on held-out set
        pred = baseline_classify(
            symptom_name=ex.symptom_name,
            symptom_definition=ex.symptom_definition,
            text=ex.text
        )
        score = metric_fn(ex, pred, trace=None)
        base_scores.append(score)
    base_score = sum(base_scores) / len(base_scores)

    print(f"Baseline Score ({args.metric}): {base_score:.1%}")
    print(f"Optimized Score ({args.metric}): {opt_score:.1%}")

    # Save individual optimized classifier
    classifier_path = f"optimized_{symptom}.json"
    logging.info(f"Saving optimized classifier to {classifier_path}")
    optimized_classify.save(classifier_path)

    optimized_results[symptom] = {
        "symptom_name": info["pretty_name"],
        "symptom_definition": info["definition"],
        "baseline_score": float(base_score),
        "optimized_score": float(opt_score),
        "metric": args.metric,
        "classifier_path": classifier_path,
    }

logging.info(f"Saving summary to {args.output}")
with open(args.output, "w") as f:
    json.dump(optimized_results, f, indent=2)

print(f"\nSaved {len(optimized_results)} optimized classifiers")
print(f"Summary saved to {args.output}")
