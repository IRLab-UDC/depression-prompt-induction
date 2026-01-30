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
args = parser.parse_args()

MODEL = args.model
SI_PATH = "data/si.json"
OUTPUT_PATH = f"runs/{MODEL.replace('/', '_')}_{args.split}_si.json"

with open(SI_PATH) as f:
    si_data = json.load(f)

samples = load_dataset(args.split)

structured_params = StructuredOutputsParams(choice=["YES", "NO"])
sampling_params = SamplingParams(structured_outputs=structured_params, max_tokens=8)

llm = LLM(model=MODEL)

SYSTEM = """You are a clinical assistant analyzing text for depression symptoms (BDI-II).

Task: Determine if the text indicates the person is CURRENTLY experiencing the specified symptom.

Answer YES if: The text explicitly expresses or clearly implies the person is experiencing this symptom now.
Answer NO if: The text is unrelated, describes past events only, or describes a different symptom.

Use the following classification guidelines:

{si}"""

def build_prompt(sentence, symptom):
    si = si_data[symptom]["si"]
    return [
        {"role": "system", "content": SYSTEM.format(si=si)},
        {"role": "user", "content": f'Text: "{sentence}"'},
    ]

sentences = [s["sentence"] for s in samples]

results = []
for symptom in si_data:
    prompts = [build_prompt(s, symptom) for s in sentences]
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
