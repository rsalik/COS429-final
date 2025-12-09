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