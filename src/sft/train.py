import argparse
import json
import random
import shutil
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from dataset_loader import load_dataset
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="google/gemma-3-4b-it")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--grad-accum", type=int, default=4)
parser.add_argument("--lora-r", type=int, default=32)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

SYMPTOMS_INFO_PATH = "data/symptoms_info.json"
OUTPUT_DIR = f"checkpoints/{args.model.replace('/', '_')}_sft"

with open(SYMPTOMS_INFO_PATH) as f:
    symptoms_info = json.load(f)
symptom_names = list(symptoms_info.keys())

SYSTEM = """You are a clinical assistant analyzing text for depression symptoms (BDI-II).

Task: Determine if the text indicates the person is CURRENTLY experiencing the specified symptom.

Answer YES if: The text explicitly expresses or clearly implies the person is experiencing this symptom now.
Answer NO if: The text is unrelated, describes past events only, or describes a different symptom."""


def build_example(sentence, symptom, label):
    info = symptoms_info[symptom]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f'Symptom: {info["pretty_name"]} ({info["definition"]})\n\nText: "{sentence}"'},
            {"role": "assistant", "content": "YES" if label == 1 else "NO"},
        ]
    }


def prepare_dataset(samples):
    records = []
    for symptom in symptom_names:
        pos_samples = [s["sentence"] for s in samples if s["labels"].get(symptom, 0) == 1]
        hard_neg_pool = [s["sentence"] for s in samples if not s["is_control"] and s["labels"].get(symptom, 0) == 0 and any(v == 1 for v in s["labels"].values())]
        soft_neg_pool = [s["sentence"] for s in samples if s["is_control"]]

        n_pos = len(pos_samples)
        n_hard = min(n_pos // 2, len(hard_neg_pool))
        n_soft = min(n_pos - n_hard, len(soft_neg_pool))

        hard_neg_samples = random.sample(hard_neg_pool, n_hard)
        soft_neg_samples = random.sample(soft_neg_pool, n_soft)

        for s in pos_samples:
            records.append(build_example(s, symptom, 1))
        for s in hard_neg_samples:
            records.append(build_example(s, symptom, 0))
        for s in soft_neg_samples:
            records.append(build_example(s, symptom, 0))

    random.shuffle(records)
    return Dataset.from_list(records)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_r * 2,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

train_samples = load_dataset("train")
val_samples = load_dataset("val")

train_dataset = prepare_dataset(train_samples)
val_dataset = prepare_dataset(val_samples)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    learning_rate=args.lr,
    warmup_ratio=0.03,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    bf16=True,
    seed=args.seed,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
)

trainer.train()

model = model.merge_and_unload()
model.save_pretrained(OUTPUT_DIR)
model.generation_config.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
