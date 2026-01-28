#!/bin/bash
#SBATCH --job-name="si-sft"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --nodelist=aragorn
#SBATCH --mem-per-cpu=64G
#SBATCH -o /mnt/experiments/nlp/eliseo/short_rcl/logs/%x-%j.out
#SBATCH -e /mnt/experiments/nlp/eliseo/short_rcl/logs/%x-%j.err

SIF="/mnt/experiments/slurm/singularity-containers/eliseo/cuda-eliseo.sif"

source "$(dirname "$0")/secrets"

MODELS=(
    "google/gemma-3-4b-it"
    "google/gemma-3-12b-it"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-4B-Instruct-2507"
    "Qwen/Qwen3-14B"
)

for MODEL in "${MODELS[@]}"; do
    singularity run --disable-cache --nv \
        --bind /mnt:/mnt \
        --pwd "$PWD" \
        $SIF \
        /bin/bash -c "src/scripts/configure_setup.sh && source venv/bin/activate && \
        python3 src/sft/train.py \
        --model $MODEL"
done

CHECKPOINTS=(
    "checkpoints/google_gemma-3-4b-it_sft"
    "checkpoints/google_gemma-3-12b-it_sft"
    "checkpoints/meta-llama_Llama-3.2-3B-Instruct_sft"
    "checkpoints/meta-llama_Llama-3.1-8B-Instruct_sft"
    "checkpoints/Qwen_Qwen3-4B-Instruct-2507_sft"
    "checkpoints/Qwen_Qwen3-14B_sft"
)

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    singularity run --disable-cache --nv \
        --bind /mnt:/mnt \
        --pwd "$PWD" \
        $SIF \
        /bin/bash -c "src/scripts/configure_setup.sh && source venv/bin/activate && \
        python3 src/sft/classify_symptoms_sft.py --model $CHECKPOINT"
done
