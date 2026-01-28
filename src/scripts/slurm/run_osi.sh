#!/bin/bash
#SBATCH --job-name="si-osi"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --nodelist=tulkas
#SBATCH --mem-per-cpu=64G
#SBATCH -o /mnt/experiments/nlp/eliseo/short_rcl/logs/%x-%j.out
#SBATCH -e /mnt/experiments/nlp/eliseo/short_rcl/logs/%x-%j.err

SIF="/mnt/experiments/slurm/singularity-containers/eliseo/cuda-eliseo.sif"

source "$(dirname "$0")/secrets"

singularity run --disable-cache --nv \
    --bind /mnt:/mnt \
    --pwd "$PWD" \
    $SIF \
    /bin/bash -c "src/scripts/configure_setup.sh && source venv/bin/activate && \
    python3 src/osi/optimize.py \
    --train-size 300 \
    --val-size 50 \
    --auto heavy \
    --metric accuracy"

singularity run --disable-cache --nv \
    --bind /mnt:/mnt \
    --pwd "$PWD" \
    $SIF \
    /bin/bash -c "src/scripts/configure_setup.sh && source venv/bin/activate && \
    python3 src/osi/classify_symptoms_osi.py \
    --model llama3.2:3b"

singularity run --disable-cache --nv \
    --bind /mnt:/mnt \
    --pwd "$PWD" \
    $SIF \
    /bin/bash -c "src/scripts/configure_setup.sh && source venv/bin/activate && \
    python3 src/osi/classify_symptoms_osi.py \
    --model llama3.2:3b \
    --optimized-file optimized_classifiers.json"

