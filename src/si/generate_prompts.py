import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from dataset_loader import load_dataset

random.seed(42)
train_samples = load_dataset("train")
symptoms_info = json.load(open("data/symptoms_info.json"))
clean = lambda s: s.replace("\n", " ").lower().strip()

results = []
for symptom in symptoms_info:
    info = symptoms_info[symptom]
    pos_pool = [s["sentence"] for s in train_samples if s["labels"].get(symptom, 0) == 1]
    hard_pool = [s["sentence"] for s in train_samples if not s["is_control"] and s["labels"].get(symptom, 0) == 0 and any(v == 1 for v in s["labels"].values())]
    soft_pool = [s["sentence"] for s in train_samples if s["is_control"]]

    n_relevants = min(15, len(pos_pool))
    relevants = [clean(s) for s in random.sample(pos_pool, n_relevants)]
    hard_neg = [clean(s) for s in random.sample(hard_pool, min(n_relevants // 2, len(hard_pool)))]
    soft_neg_samples = [clean(s) for s in random.sample(soft_pool, min(n_relevants - len(hard_neg), len(soft_pool)))]
    negatives = hard_neg + soft_neg_samples
    random.shuffle(negatives)

    user_prompt = f"""Analyze these examples to create classification guidelines for determining if a sentence is SPECIFICALLY ABOUT the "{info['pretty_name']}" symptom dimension.

Definition: {info['definition']}

SPECIFICALLY ABOUT THIS SYMPTOM examples (whether symptom present or absent):
{chr(10).join(f"- {s}" for s in relevants)}

NOT ABOUT THIS SYMPTOM examples (about different symptoms or unrelated):
{chr(10).join(f"- {s}" for s in negatives)}

Generate guidelines in this exact format:

CORE QUESTION: [One yes/no question about whether the sentence is specifically about THIS PARTICULAR symptom, not just generally depression-related]

SPECIFICALLY ABOUT THIS SYMPTOM IF the sentence:
- [criterion 1]
- [criterion 2]
- [up to 5 criteria]

NOT ABOUT THIS SYMPTOM IF the sentence:
- [criterion 1]
- [criterion 2]
- [up to 3 criteria]

KEY VOCABULARY: [comma-separated words/phrases specifically indicating this symptom]

TRICKY CASES: [1-2 sentences about edge cases, especially how to distinguish from similar symptoms]"""

    system_prompt = "You are an expert at extracting classification rules from examples. Follow the output format exactly. Be concise - each criterion should be one line."

    results.append({
        "symptom": symptom,
        "pretty_name": info["pretty_name"],
        "definition": info["definition"],
        "relevants": relevants,
        "hard_negatives": hard_neg,
        "soft_negatives": soft_neg_samples,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    })

with open("data/si_prompts.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
