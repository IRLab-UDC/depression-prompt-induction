import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from dataset_loader import load_dataset, get_symptom_labels
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="google/gemma-3-4b-it")
parser.add_argument("--split", default="test")
parser.add_argument("--n-shots", type=int, default=15)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_model_len", type=int, default=None)
args = parser.parse_args()

random.seed(args.seed)

MODEL = args.model
N_SHOTS = args.n_shots
SYMPTOMS_INFO_PATH = "data/symptoms_info.json"
OUTPUT_PATH = f"runs/{MODEL.replace('/', '_')}_{args.split}_{N_SHOTS}shot_icl.json"

with open(SYMPTOMS_INFO_PATH) as f:
    symptoms_info = json.load(f)

test_samples = load_dataset(args.split)
train_samples = load_dataset("train")

structured_params = StructuredOutputsParams(choice=["YES", "NO"])
sampling_params = SamplingParams(structured_outputs=structured_params, max_tokens=8)

llm_kwargs = {"model": MODEL}
if args.max_model_len is not None:
    llm_kwargs["max_model_len"] = args.max_model_len
llm = LLM(**llm_kwargs)

SYSTEM = """You are a clinical assistant analyzing text for depression symptoms (BDI-II).

Task: Determine if the text is specifically about the specified symptom dimension.

Answer YES if: The text specifically discusses or provides evidence about THIS PARTICULAR symptom (whether the symptom is present or absent). The sentence must be directly relevant to this specific symptom, not just generally depression-related.
Answer NO if: The text is about a DIFFERENT symptom, or is completely unrelated to this symptom dimension."""


def get_examples(symptom, n_shots):
    pos = [s["sentence"] for s in train_samples if s["labels"].get(symptom, 0) == 1]
    hard_neg = [s["sentence"] for s in train_samples if not s["is_control"] and s["labels"].get(symptom, 0) == 0 and any(v == 1 for v in s["labels"].values())]
    soft_neg = [s["sentence"] for s in train_samples if s["is_control"]]

    n_pos = n_shots // 2 + n_shots % 2
    n_neg = n_shots // 2
    n_hard = n_neg // 2
    n_soft = n_neg - n_hard

    pos_samples = random.sample(pos, min(n_pos, len(pos)))
    hard_samples = random.sample(hard_neg, min(n_hard, len(hard_neg)))
    soft_samples = random.sample(soft_neg, min(n_soft, len(soft_neg)))
    neg_samples = hard_samples + soft_samples

    examples = [(s, "YES") for s in pos_samples] + [(s, "NO") for s in neg_samples]
    random.shuffle(examples)
    return examples


def build_prompt(sentence, symptom, examples):
    info = symptoms_info[symptom]
    messages = [{"role": "system", "content": SYSTEM}]
    for ex_sent, ex_label in examples:
        messages.append({"role": "user", "content": f'Symptom: {info["pretty_name"]} ({info["definition"]})\n\nText: "{ex_sent}"'})
        messages.append({"role": "assistant", "content": ex_label})
    messages.append({"role": "user", "content": f'Symptom: {info["pretty_name"]} ({info["definition"]})\n\nText: "{sentence}"'})
    return messages


sentences = [s["sentence"] for s in test_samples]

results = []
for symptom in symptoms_info:
    examples = get_examples(symptom, N_SHOTS)
    prompts = [build_prompt(s, symptom, examples) for s in sentences]
    outputs = llm.chat(prompts, sampling_params=sampling_params, chat_template_kwargs={"enable_thinking": False})
    preds = [1 if o.outputs[0].text.strip() == "YES" else 0 for o in outputs]
    ground_truth = get_symptom_labels(test_samples, symptom)

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

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
