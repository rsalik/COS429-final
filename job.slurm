#!/bin/bash
#SBATCH --job-name=cos429-fp-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --mail-user=rs3216@princeton.edu
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=eladgroup

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

module load anaconda/2020a
source activate cos429-final-env

# Pre-downloaded model weights
export TORCH_HOME=${HOME}/.cache/torch
export HF_HOME=${HOME}/.cache/huggingface

# Run your training script
python -m src.main

