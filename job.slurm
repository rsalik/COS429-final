#!/bin/bash
#SBATCH --job-name=cos429-fp-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=rs3216@princeton.edu
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

echo "Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)" && echo "Node: $(hostname)" && echo "Start: $(date)"

# Ensure log directory exists for Slurm outputs.
mkdir -p logs

# Load user shell config to make pyenv available, then select the project env.
source ~/.bashrc
pyenv shell cos429-final-env

# Move to the submission directory so relative paths work as expected.
cd "$SLURM_SUBMIT_DIR"

echo "Python: $(which python)" && python --version
nvidia-smi || true

# Run training.
python -m src.main

echo "End: $(date)"

