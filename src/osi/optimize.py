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
from classifier import SymptomClassification, load_custom_signatures
from metrics import classification_metric, weighted_classification_metric

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--task-model", default="meta-llama/Llama-3.2-3B-Instruct")
parser.add_argument("--prompt-model", default="google/gemma-3-27b-it")
parser.add_argument("--train-size", type=int, default=100)
parser.add_argument("--output", default=None)
parser.add_argument("--prompt-host", default="http://aragorn:30000/v1")
parser.add_argument("--task-host", default="http://aragorn:30001/v1")
parser.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
parser.add_argument("--num-threads", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--symptom", type=str, default=None)
parser.add_argument("--val-size", type=int, default=25)
parser.add_argument("--metric", default="weighted", choices=["accuracy", "weighted"])
args = parser.parse_args()

if args.output is None:
    task_model_name = args.task_model.replace(':', '_').replace('/', '_')
    output_dir = Path(f"osi_optimizations/{task_model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output = str(output_dir / "optimized_classifiers.json")
else:
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

random.seed(args.seed)

with open("data/symptoms_info.json") as f:
    symptoms_info = json.load(f)

train_samples = load_dataset("train")
val_samples = load_dataset("val")

task_lm = dspy.LM(f"openai/{args.task_model}", api_base=args.task_host, api_key="local", model_type="chat")
dspy.configure(lm=task_lm)

import litellm
litellm.drop_params = False

prompt_lm = dspy.LM(
    f"openai/{args.prompt_model}",
    api_base=args.prompt_host,
    api_key="local",
    model_type="chat",
    temperature=0.9,
    top_p=0.95
)

symptoms_to_optimize = {args.symptom: symptoms_info[args.symptom]} if args.symptom else symptoms_info
custom_signatures = load_custom_signatures("data/si.json")
optimized_results = {}

for symptom, info in symptoms_to_optimize.items():
    print(f"\n{'='*60}")
    print(f"Optimizing: {symptom}")
    print(f"{'='*60}")

    signature_class = custom_signatures.get(symptom, SymptomClassification)

    train_pos = [s["sentence"] for s in train_samples if s["labels"].get(symptom, 0) == 1]
    train_hard_neg = [s["sentence"] for s in train_samples if not s["is_control"] and s["labels"].get(symptom, 0) == 0 and any(v == 1 for v in s["labels"].values())]
    train_soft_neg = [s["sentence"] for s in train_samples if s["is_control"]]

    trainset = []
    n_pos = min(len(train_pos), args.train_size // 2)
    for s in train_pos[:n_pos]:
        trainset.append(dspy.Example(
            symptom_name=info["pretty_name"],
            symptom_definition=info["definition"],
            text=s,
            answer="YES"
        ).with_inputs("symptom_name", "symptom_definition", "text"))

    n_hard = n_pos // 2
    n_soft = n_pos - n_hard
    for s in train_hard_neg[:n_hard]:
        trainset.append(dspy.Example(
            symptom_name=info["pretty_name"],
            symptom_definition=info["definition"],
            text=s,
            answer="NO"
        ).with_inputs("symptom_name", "symptom_definition", "text"))
    for s in train_soft_neg[:n_soft]:
        trainset.append(dspy.Example(
            symptom_name=info["pretty_name"],
            symptom_definition=info["definition"],
            text=s,
            answer="NO"
        ).with_inputs("symptom_name", "symptom_definition", "text"))

    random.shuffle(trainset)

    val_pos = [s["sentence"] for s in val_samples if s["labels"].get(symptom, 0) == 1]
    val_neg = [s["sentence"] for s in val_samples if s["labels"].get(symptom, 0) == 0]

    opt_size = args.val_size // 2
    eval_size = args.val_size - opt_size

    optset = []
    evalset = []

    for s in val_pos[:opt_size]:
        optset.append(dspy.Example(
            symptom_name=info["pretty_name"],
            symptom_definition=info["definition"],
            text=s,
            answer="YES"
        ).with_inputs("symptom_name", "symptom_definition", "text"))
    for s in val_neg[:opt_size]:
        optset.append(dspy.Example(
            symptom_name=info["pretty_name"],
            symptom_definition=info["definition"],
            text=s,
            answer="NO"
        ).with_inputs("symptom_name", "symptom_definition", "text"))

    for s in val_pos[opt_size:opt_size+eval_size]:
        evalset.append(dspy.Example(
            symptom_name=info["pretty_name"],
            symptom_definition=info["definition"],
            text=s,
            answer="YES"
        ).with_inputs("symptom_name", "symptom_definition", "text"))
    for s in val_neg[opt_size:opt_size+eval_size]:
        evalset.append(dspy.Example(
            symptom_name=info["pretty_name"],
            symptom_definition=info["definition"],
            text=s,
            answer="NO"
        ).with_inputs("symptom_name", "symptom_definition", "text"))

    random.shuffle(optset)
    random.shuffle(evalset)

    print(f"Train: {len(trainset)} | Opt: {len(optset)} | Eval: {len(evalset)}")

    classify = dspy.Predict(signature_class)
    metric_fn = weighted_classification_metric if args.metric == "weighted" else classification_metric

    optimizer = MIPROv2(
        metric=metric_fn,
        auto=args.auto,
        num_threads=args.num_threads,
        prompt_model=prompt_lm,
        task_model=task_lm,
        seed=args.seed,
        verbose=True,
        init_temperature=1.2,
    )

    optimized_classify = optimizer.compile(classify, trainset=trainset, valset=optset)

    opt_scores = [metric_fn(ex, optimized_classify(symptom_name=ex.symptom_name, symptom_definition=ex.symptom_definition, text=ex.text), trace=None) for ex in evalset]
    opt_score = sum(opt_scores) / len(opt_scores)

    baseline_classify = dspy.Predict(signature_class)
    base_scores = [metric_fn(ex, baseline_classify(symptom_name=ex.symptom_name, symptom_definition=ex.symptom_definition, text=ex.text), trace=None) for ex in evalset]
    base_score = sum(base_scores) / len(base_scores)

    print(f"Baseline: {base_score:.1%} | Optimized: {opt_score:.1%}")

    classifier_path = str(output_dir / f"optimized_{symptom}.json")
    optimized_classify.save(classifier_path)

    optimized_results[symptom] = {
        "symptom_name": info["pretty_name"],
        "symptom_definition": info["definition"],
        "baseline_score": float(base_score),
        "optimized_score": float(opt_score),
        "metric": args.metric,
    }

with open(args.output, "w") as f:
    json.dump(optimized_results, f, indent=2)

print(f"\nSaved {len(optimized_results)} optimized classifiers to {args.output}")
