#!/bin/bash
#SBATCH --job-name="si-osi"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --nodelist=aragorn
#SBATCH --mem-per-cpu=64G
#SBATCH -o /mnt/experiments/nlp/eliseo/short_rcl/logs/%x-%j.out
#SBATCH -e /mnt/experiments/nlp/eliseo/short_rcl/logs/%x-%j.err

SIF="/mnt/experiments/slurm/singularity-containers/eliseo/cuda-eliseo.sif"

source src/scripts/slurm/secrets

OLLAMA_HOST="http://0.0.0.0:11435"
PROMPT_MODEL="gemma3:27b"

TRAIN_SIZE=300
VAL_SIZE=50
AUTO_MODE="heavy"
METRIC="accuracy"

TASK_MODELS=(
    "gemma3:4b"
    "gemma3:12b"
    "llama3.2:3b"
    "llama3.1:8b"
    "qwen3:4b"
    "qwen3:14b"
    "phi4:14b"
    "phi4-mini:3.8b"
)

curl -s $OLLAMA_HOST/api/pull -d "{\"model\": \"$PROMPT_MODEL\"}"

for TASK_MODEL in "${TASK_MODELS[@]}"; do
    TASK_MODEL_NAME=$(echo "$TASK_MODEL" | sed 's/[:\\/]/_/g')
    OPTIMIZED_DIR="osi_optimizations/${TASK_MODEL_NAME}"

    singularity run --disable-cache --nv \
        --bind /mnt:/mnt \
        --pwd "$PWD" \
        $SIF \
        /bin/bash -c "src/scripts/configure_setup.sh && source venv/bin/activate && \
        python3 src/osi/optimize.py \
        --ollama-host $OLLAMA_HOST \
        --prompt-model $PROMPT_MODEL \
        --task-model $TASK_MODEL \
        --train-size $TRAIN_SIZE \
        --val-size $VAL_SIZE \
        --auto $AUTO_MODE \
        --metric $METRIC && \
        python3 src/osi/classify_symptoms_osi.py \
        --ollama-host $OLLAMA_HOST \
        --model $TASK_MODEL \
        --optimized-dir $OPTIMIZED_DIR && \
        curl -X POST $OLLAMA_HOST/api/generate -d '{\"model\": \"$TASK_MODEL\", \"keep_alive\": 0}'"
done
