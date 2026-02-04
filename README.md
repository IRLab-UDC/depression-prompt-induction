# Learning Evidence of Depression Symptoms via Induced Prompts

Automated sentence-level classification of BDI-II (Beck Depression Inventory) symptoms from text using Large Language Models. This project introduces **Symptom Induction (SI)**, a novel approach that compresses labeled examples into concise natural-language classification guidelines, offering an interpretable alternative to few-shot prompting and model fine-tuning.

## Overview

This research addresses the challenge of detecting 21 depression symptoms from user-generated text according to the BDI-II framework. The task is framed as binary classification at the sentence level: determining whether a given text indicates the person is currently experiencing each specific symptom.

### Dataset: BDI-Sen

A labeled dataset of sentences annotated for depression symptoms with stratified negative sampling:

- **Train**: 762 samples (381 positive, 381 negative) - balanced 1:1
- **Validation**: 130 samples (65 positive, 65 negative) - balanced 1:1
- **Test**: 624 samples (104 positive, 520 negative) - imbalanced 5:1

**Negative Sample Types**:
- **Control sentences** (soft negatives): Unrelated to depression
- **Symptom-annotated negatives** (hard negatives): About other depression symptoms

**21 BDI-II Symptoms**: Sadness, Pessimism, Past Failure, Loss of Pleasure, Guilty Feelings, Punishment Feelings, Self-Dislike, Self-Criticalness, Suicidal Thoughts, Crying, Agitation, Loss of Interest, Indecisiveness, Worthlessness, Loss of Energy, Changes in Sleep, Irritability, Changes in Appetite, Concentration Difficulty, Tiredness/Fatigue, Loss of Interest in Sex

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
**Core Innovation**: Automatically generate concise, interpretable classification guidelines from training examples using an LLM, then inject those guidelines into the system prompt for inference. This approach provides transparency and improved performance on rare symptoms without the cost of few-shot examples or fine-tuning.

```bash
# Step 1: Generate guidelines from training examples
python src/si/generate_prompts.py

# Step 2: Use guidelines for classification
python src/si/infer_si_prompts.py --model google/gemma-3-4b-it
python src/si/classify_symptoms_si.py --model google/gemma-3-4b-it --split test
```

**Generated guidelines include**:
- CORE QUESTION: What this symptom fundamentally measures
- SPECIFICALLY ABOUT: Patterns that indicate presence
- NOT ABOUT: Patterns that don't qualify
- KEY VOCABULARY: Relevant terminology
- TRICKY CASES: Edge cases and clarifications

## Project Structure

```
.
├── data/
│   ├── bdi_sen/                 # Main dataset (train/val/test splits)
│   ├── psysym/                  # Cross-domain validation dataset
│   ├── symptoms_info.json       # 21 symptom definitions
│   └── si_prompts.jsonl         # Generated symptom induction guidelines
├── src/
│   ├── zs/                      # Zero-shot classification
│   ├── icl/                     # In-context learning (few-shot)
│   ├── sft/                     # Supervised fine-tuning (train + inference)
│   ├── si/                      # Symptom induction (prompt generation + inference)
│   ├── cross_domain_experiment/ # Generalization tests on PsySym dataset
│   ├── utils/                   # Dataset loading and evaluation
│   ├── plotting/                # Visualization (confusion matrices, distributions)
│   └── scripts/                 # Shell scripts and SLURM configs
├── checkpoints/                 # Fine-tuned model checkpoints
├── results/                     # Evaluation metrics (JSON files)
├── runs/                        # Raw predictions from each approach
└── paper/                       # LaTeX manuscript
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies**:
- `transformers`, `datasets`: Model and data handling
- `vllm`: Efficient batch inference with structured outputs
- `bitsandbytes`: 4-bit quantization (NF4)
- `peft`: LoRA parameter-efficient fine-tuning
- `trl`: Training utilities
- `scikit-learn`: Evaluation metrics
- `matplotlib`, `seaborn`: Visualization

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

## Models Evaluated

Fine-tuned checkpoints are available for the following models:
- **Google Gemma**: 3-4B-IT, 3-12B-IT
- **Meta Llama**: 3.1-8B-Instruct, 3.2-3B-Instruct
- **Microsoft Phi**: 4, 4-Mini
- **Qwen**: Qwen3-4B, Qwen3-14B

All models are evaluated across the four approaches (ZS, ICL, SFT, SI).

## Cross-Domain Validation

The project includes generalization tests on the **PsySym dataset** to validate model performance on different data sources. Scripts are located in `src/cross_domain_experiment/`.

## Key Technical Details

### Dataset Loader (`src/utils/dataset_loader.py`)
- Configurable positive/negative sampling ratios
- Stratified negative sampling (control vs. symptom-annotated)
- Balanced class sampling with random seed control (seed=42)
- Local dataset caching

### Evaluation (`src/utils/evaluate.py`)
- **Per-symptom metrics**: Precision, Recall, F1, Accuracy, Confusion Matrix
- **Overall binary metrics**: Aggregated across all symptom-sentence pairs
- **Multilabel metrics**: Micro/Macro/Weighted F1 treating symptoms as separate labels

### Inference Pipeline
- Uses vLLM for efficient batch inference
- Structured output constraints (YES/NO responses)
- System prompts with symptom definitions
- Conversation history for ICL approaches

### Fine-Tuning Configuration
- 4-bit NF4 quantization with bitsandbytes
- LoRA adaptation (target modules: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj)
- Training on multi-symptom formatted data

## Visualization

Generate plots for analysis:

```bash
python src/plotting/plot_overall_matrices.py
python src/plotting/plot_bdisen_symptom_distribution.py
python src/plotting/plot_radial_log.py
```

## Research Contribution

**Symptom Induction (SI)** is the core innovation of this work. Rather than relying on manually crafted few-shot examples or resource-intensive fine-tuning, SI automatically extracts and compresses classification knowledge from training data into concise, human-readable guidelines. These guidelines are injected into the system prompt to improve performance on rare symptoms while maintaining interpretability and transparency in classification decisions.

## Citation

If you use this code or dataset, please cite the BDI-Sen dataset and the related research paper.
