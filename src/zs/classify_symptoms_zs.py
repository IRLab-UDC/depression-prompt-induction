import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from dataset_loader import load_dataset, get_symptom_labels
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="google/gemma-3-4b-it")
parser.add_argument("--split", default="test")
parser.add_argument("--max_model_len", type=int, default=None)
args = parser.parse_args()

MODEL = args.model
SYMPTOMS_INFO_PATH = "data/symptoms_info.json"
OUTPUT_PATH = f"runs/{MODEL.replace('/', '_')}_{args.split}_zs.json"

with open(SYMPTOMS_INFO_PATH) as f:
    symptoms_info = json.load(f)

samples = load_dataset(args.split)

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

def build_prompt(sentence, symptom):
    info = symptoms_info[symptom]
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f'Symptom: {info["pretty_name"]} ({info["definition"]})\n\nText: "{sentence}"'},
    ]

sentences = [s["sentence"] for s in samples]

results = []
for symptom in symptoms_info:
    prompts = [build_prompt(s["sentence"], symptom) for s in samples]
    outputs = llm.chat(prompts, sampling_params=sampling_params, chat_template_kwargs={"enable_thinking": False})
    preds = [1 if o.outputs[0].text.strip() == "YES" else 0 for o in outputs]
    ground_truth = get_symptom_labels(samples, symptom)

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
