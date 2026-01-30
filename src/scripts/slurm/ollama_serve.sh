#!/bin/bash
#SBATCH --job-name="si-ollama"
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --nodelist=tulkas
#SBATCH --mem-per-cpu=64G
#SBATCH -o /mnt/experiments/nlp/eliseo/short_rcl/logs/%x-%j.out
#SBATCH -e /mnt/experiments/nlp/eliseo/short_rcl/logs/%x-%j.err

SIF="/mnt/experiments/slurm/singularity-containers/eliseo/ollama_0_15_2.sif"

source src/scripts/slurm/secrets


singularity run --disable-cache --nv \
    --bind /mnt:/mnt \
    --pwd "$PWD" \
    --env OLLAMA_DEBUG=2 \
    --env OLLAMA_KEEP_ALIVE=-1 \
    --env OLLAMA_MODELS=$HF_HOME/ollama \
    --env OLLAMA_CONTEXT_LENGTH=8192 \
    --env OLLAMA_GPU_LAYERS=999 \
    --env OLLAMA_NUM_PARALLEL=4 \
    --env OLLAMA_NUM_THREADS=4 \
    $SIF