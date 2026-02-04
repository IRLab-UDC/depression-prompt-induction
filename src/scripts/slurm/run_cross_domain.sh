#!/bin/bash
#SBATCH --job-name="si-cross-domain"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --nodelist=aragorn
#SBATCH --mem-per-cpu=64G
#SBATCH -o /mnt/experiments/nlp/eliseo/short_rcl/logs/%x-%j.out
#SBATCH -e /mnt/experiments/nlp/eliseo/short_rcl/logs/%x-%j.err

SIF="/mnt/experiments/slurm/singularity-containers/eliseo/cuda-eliseo.sif"

source src/scripts/slurm/secrets

MODELS=(
    "google/gemma-3-4b-it"
    # "google/gemma-3-12b-it"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "Qwen/Qwen3-14B"
    # "microsoft/phi-4"
    # "microsoft/Phi-4-mini-instruct"
)

CHECKPOINTS=(
    "checkpoints/google_gemma-3-4b-it_sft"
    # "checkpoints/google_gemma-3-12b-it_sft"
    # "checkpoints/meta-llama_Llama-3.2-3B-Instruct_sft"
    # "checkpoints/meta-llama_Llama-3.1-8B-Instruct_sft"
    # "checkpoints/Qwen_Qwen3-4B-Instruct-2507_sft"
    # "checkpoints/Qwen_Qwen3-14B_sft"
    # "checkpoints/microsoft_phi-4_sft"
    # "checkpoints/microsoft_Phi-4-mini-instruct_sft"
)

for MODEL in "${MODELS[@]}"; do
    singularity run --disable-cache --nv \
        --bind /mnt:/mnt \
        --pwd "$PWD" \
        $SIF \
        /bin/bash -c "src/scripts/configure_setup.sh && source venv/bin/activate && \
        python3 src/cross_domain_experiment/classify_symptoms_zs.py \
        --model $MODEL"
done

for MODEL in "${MODELS[@]}"; do
    singularity run --disable-cache --nv \
        --bind /mnt:/mnt \
        --pwd "$PWD" \
        $SIF \
        /bin/bash -c "src/scripts/configure_setup.sh && source venv/bin/activate && \
        python3 src/cross_domain_experiment/classify_symptoms_icl.py \
        --model $MODEL"
done

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    singularity run --disable-cache --nv \
        --bind /mnt:/mnt \
        --pwd "$PWD" \
        $SIF \
        /bin/bash -c "src/scripts/configure_setup.sh && source venv/bin/activate && \
        python3 src/cross_domain_experiment/classify_symptoms_sft.py --model $CHECKPOINT"
done

for MODEL in "${MODELS[@]}"; do
    singularity run --disable-cache --nv \
        --bind /mnt:/mnt \
        --pwd "$PWD" \
        $SIF \
        /bin/bash -c "src/scripts/configure_setup.sh && source venv/bin/activate && \
        python3 src/cross_domain_experiment/classify_symptoms_si.py \
        --model $MODEL"
done


