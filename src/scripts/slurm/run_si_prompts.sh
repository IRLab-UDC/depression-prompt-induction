#!/bin/bash
#SBATCH --job-name="si-si"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --nodelist=aragorn
#SBATCH --mem-per-cpu=64G
#SBATCH -o /mnt/experiments/nlp/eliseo/short_rcl/logs/%x-%j.out
#SBATCH -e /mnt/experiments/nlp/eliseo/short_rcl/logs/%x-%j.err

SIF="/mnt/experiments/slurm/singularity-containers/eliseo/cuda-eliseo.sif"

source src/scripts/slurm/secrets


singularity run --disable-cache --nv \
    --bind /mnt:/mnt \
    --pwd "$PWD" \
    $SIF \
    /bin/bash -c "src/scripts/configure_setup.sh && source venv/bin/activate && \
    python3 src/si/infer_si_prompts.py"
