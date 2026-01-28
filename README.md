# Learning Evidence of Depression Symptoms via Induced Prompts

Automated classification of BDI-II (Beck Depression Inventory) symptoms from text using Large Language Models. Compares multiple prompting strategies including zero-shot, in-context learning, supervised fine-tuning, symptom induction, and optimized instruction prompting.

## Overview

This project implements and evaluates different LLM-based approaches for detecting 21 depression symptoms from text according to the BDI-II framework. The task is framed as binary classification: determining if a given text indicates the person is currently experiencing each specific symptom.

**Dataset**: BDI-Sen v2 - a labeled dataset of sentences annotated for depression symptoms
- **Train**: 762 samples (381 positive, 381 negative)
- **Validation**: 130 samples (65 positive, 65 negative)
- **Test**: 624 samples (104 positive, 520 negative)
- **21 BDI-II symptoms**: Sadness, Pessimism, Past Failure, Loss of Pleasure, Guilty Feelings, Punishment Feelings, Self-Dislike, Self-Criticalness, Suicidal Thoughts, Crying, Agitation, Loss of Interest, Indecisiveness, Worthlessness, Loss of Energy, Changes in Sleep, Irritability, Changes in Appetite, Concentration Difficulty, Tiredness/Fatigue, Loss of Interest in Sex

## Approaches

### 1. Zero-Shot (ZS)
Classify symptoms using only the symptom definition, no training examples.

```bash
python src/zs/classify_symptoms_zs.py --model google/gemma-3-4b-it --split test
```

### 2. In-Context Learning (ICL)
Provide few-shot examples (default 15-shot) in the prompt with balanced positive/negative examples.

```bash
python src/icl/classify_symptoms_icl.py --model google/gemma-3-4b-it --split test --n-shots 15
```

### 3. Supervised Fine-Tuning (SFT)
Fine-tune models using LoRA on the training set.

```bash
python src/sft/train.py --model google/gemma-3-4b-it --epochs 2 --lr 2e-4 --batch-size 4
python src/sft/classify_symptoms_sft.py --model checkpoints/google_gemma-3-4b-it_sft --split test
```

### 4. Symptom Induction (SI)
Generate classification guidelines from training examples using an LLM, then use those guidelines for inference.

```bash
# Generate guidelines from training examples
python src/si/generate_prompts.py

# Use guidelines for classification (requires manual inference step)
python src/si/infer_si_prompts.py --model google/gemma-3-4b-it
python src/si/classify_symptoms_si.py --model google/gemma-3-4b-it --split test
```

### 5. Optimized Symptom Induction (OSI)
Use DSPy's MIPROv2 optimizer to automatically optimize prompts and instructions.

```bash
# Optimize prompts for all symptoms
python src/osi/optimize.py \
  --task-model llama3.2:3b \
  --prompt-model phi4 \
  --train-size 100 \
  --val-size 25 \
  --auto light \
  --metric weighted

# Classify using optimized prompts
python src/osi/classify_symptoms_osi.py \
  --model llama3.2:3b \
  --split test \
  --optimized-file optimized_classifiers.json
```

## Project Structure

```
.
├── data/
│   ├── bdi_sen_v2/          # Dataset splits (train/val/test)
│   ├── symptoms_info.json   # Symptom definitions
│   └── si_prompts.jsonl     # Generated self-improvement prompts
├── src/
│   ├── zs/                  # Zero-shot classification
│   ├── icl/                 # In-context learning
│   ├── sft/                 # Supervised fine-tuning
│   ├── si/                  # Symptom induction
│   ├── osi/                 # Optimized symptom induction (DSPy)
│   ├── utils/               # Dataset loading and evaluation
│   └── plotting/            # Visualization scripts
├── results/                 # Evaluation metrics and plots
└── runs/                    # Model predictions
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements**:
- transformers
- datasets
- vllm
- bitsandbytes
- matplotlib
- peft
- trl
- accelerate
- dspy
- ollama
- scikit-learn

## Evaluation

Evaluate predictions and compute metrics:

```bash
python src/utils/evaluate.py runs/your_predictions.json
```

**Metrics computed**:
- Per-symptom: Precision, Recall, F1, Confusion Matrix
- Overall binary: Aggregated metrics across all symptom-sentence pairs
- Multilabel: Micro/Macro/Weighted averages treating symptoms as separate labels

Generate comparison tables:

```bash
python src/utils/fill_overall_table.py
python src/utils/fill_per_symptom_table.py
```

## Key Components

### Dataset Loader (`src/utils/dataset_loader.py`)
- Loads train/val/test splits with configurable positive/negative ratios
- Handles control sentences (soft negatives) and symptom-annotated negatives (hard negatives)
- Balances classes for training

### DSPy Classifier (`src/osi/classifier.py`)
Defines the signature for symptom classification with structured inputs/outputs for optimization.

### Metrics (`src/osi/metrics.py`)
- `classification_metric`: Binary accuracy
- `weighted_classification_metric`: Penalizes false negatives more than false positives (useful when missing symptoms is worse than over-predicting)

## Notes

- Models are run using vLLM for efficient inference with structured outputs
- OSI optimization uses Ollama-hosted models for flexibility
- Fine-tuning uses 4-bit quantization (NF4) and LoRA for efficiency
- Test set has a 5:1 negative-to-positive ratio to simulate real-world imbalance

## Citation

If you use this code or dataset, please cite the BDI-Sen dataset and relevant papers.
