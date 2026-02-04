import argparse
import json
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--split", default="test")
parser.add_argument("--data_dir", default="data/psysym")
args = parser.parse_args()

MODEL = args.model
SYMPTOMS_INFO_PATH = "data/symptoms_info.json"
DATA_PATH = Path(args.data_dir) / f"{args.split}.jsonl"
OUTPUT_PATH = f"runs/cross_domain/{MODEL.replace('/', '_')}_{args.split}_sft_psysym.json"

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

with open(SYMPTOMS_INFO_PATH) as f:
    symptoms_info = json.load(f)

with open(DATA_PATH) as f:
    samples = [json.loads(line) for line in f]

structured_params = StructuredOutputsParams(choice=["YES", "NO"])
sampling_params = SamplingParams(structured_outputs=structured_params, max_tokens=8)

llm = LLM(
    model=MODEL,
    limit_mm_per_prompt={"image": 0},
    gpu_memory_utilization=0.95,
    max_model_len=4096,
    enforce_eager=True
)

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


def get_symptom_labels(samples, symptom):
    return [s["labels"][symptom] for s in samples]


sentences = [s["sentence"] for s in samples]

results = []
for symptom in symptoms_info:
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
            "result_type": result_type,
            "diseases": samples[i]["diseases"],
            "kb_symptoms": samples[i]["kb_symptoms"],
            "is_control": samples[i]["is_control"]
        })

    results.append({
        "symptom": symptom,
        "predictions": preds,
        "ground_truth": ground_truth,
        "details": predictions_with_sentences
    })

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {OUTPUT_PATH}")
