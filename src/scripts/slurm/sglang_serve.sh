#!/bin/bash
#SBATCH --job-name="si-sglang"
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --nodelist=aragorn
#SBATCH --mem-per-cpu=64G
#SBATCH -o /mnt/experiments/nlp/eliseo/short_rcl/logs/%x-%j.out
#SBATCH -e /mnt/experiments/nlp/eliseo/short_rcl/logs/%x-%j.err

SIF="/mnt/experiments/slurm/singularity-containers/eliseo/sglang.sif"

source src/scripts/slurm/secrets

singularity run --disable-cache --nv \
    --bind /mnt:/mnt \
    --pwd "$PWD" \
    $SIF \
    /bin/bash -c "export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 && python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.3-70B-Instruct \
    --host 0.0.0.0 --port 30000 \
    --tp-size 2 \
    --cuda-graph-max-bs 8 \
    --context-length 16384 \
    --mem-fraction-static 0.88 \
    --log-requests"
