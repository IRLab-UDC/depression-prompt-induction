import json
from vllm import LLM, SamplingParams

llm = LLM(model="google/gemma-3-27b-it")

with open("data/si_prompts.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

prompts = [item["prompt"] for item in data]
sampling_params = SamplingParams(max_tokens=2048)
outputs = llm.chat(prompts, sampling_params=sampling_params, chat_template_kwargs={"enable_thinking": False})

results = {}
for item, output in zip(data, outputs):
    si = output.outputs[0].text
    results[item["symptom"]] = {
        "input_prompt": item["prompt"],
        "si": si
    }

with open("data/si.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
