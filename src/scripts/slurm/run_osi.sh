#!/bin/bash
#SBATCH --job-name="si-osi"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --nodelist=tulkas
#SBATCH --mem-per-cpu=64G
#SBATCH -o /mnt/experiments/nlp/eliseo/short_rcl/logs/%x-%j.out
#SBATCH -e /mnt/experiments/nlp/eliseo/short_rcl/logs/%x-%j.err

SIF="/mnt/experiments/slurm/singularity-containers/eliseo/cuda-eliseo.sif"
SGLANG_SIF="/mnt/experiments/slurm/singularity-containers/eliseo/sglang.sif"

source src/scripts/slurm/secrets

PROMPT_HOST="http://aragorn:30000/v1"
TASK_HOST="http://tulkas:30001/v1"
PROMPT_MODEL="meta-llama/Llama-3.3-70B-Instruct"

TRAIN_SIZE=500
VAL_SIZE=50
AUTO_MODE="heavy"
METRIC="accuracy"
TEST_SPLIT="test"

TASK_MODELS=(
    # "google/gemma-3-4b-it"
    # "google/gemma-3-12b-it"
    "meta-llama/Llama-3.2-3B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "Qwen/Qwen3-4B-Instruct-2507"
    # "Qwen/Qwen3-14B"
    # "microsoft/phi-4"
    # "microsoft/Phi-4-mini-instruct"
)

for TASK_MODEL in "${TASK_MODELS[@]}"; do
    echo "Starting task model: $TASK_MODEL"
    TASK_MODEL_NAME=$(echo "$TASK_MODEL" | sed 's/[:\\/]/_/g')
    OPTIMIZED_DIR="osi_optimizations/${TASK_MODEL_NAME}"

    singularity run --disable-cache --nv \
        --bind /mnt:/mnt \
        --pwd "$PWD" \
        $SGLANG_SIF \
        /bin/bash -c "python3 -m sglang.launch_server \
        --model-path $TASK_MODEL \
        --host 0.0.0.0 --port 30001 \
        --mem-fraction-static 0.7 \
        --cuda-graph-max-bs 16 \
        --max-prefill-tokens 8192" &

    SERVER_PID=$!
    echo "Server started with PID: $SERVER_PID"

    sleep 2m

    singularity run --disable-cache --nv \
        --bind /mnt:/mnt \
        --pwd "$PWD" \
        $SIF \
        /bin/bash -c "src/scripts/configure_setup.sh && source venv/bin/activate && \
        python3 src/osi/optimize.py \
        --prompt-host $PROMPT_HOST \
        --prompt-model $PROMPT_MODEL \
        --task-host $TASK_HOST \
        --task-model $TASK_MODEL \
        --train-size $TRAIN_SIZE \
        --val-size $VAL_SIZE \
        --auto $AUTO_MODE \
        --metric $METRIC && \
        python3 src/osi/classify_symptoms_osi.py \
        --task-host $TASK_HOST \
        --task-model $TASK_MODEL \
        --optimized-dir $OPTIMIZED_DIR \
        --split $TEST_SPLIT"

    echo "Stopping server (PID: $SERVER_PID)"
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    echo "Server stopped"
done
