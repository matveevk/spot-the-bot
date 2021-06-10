#!/bin/bash
#SBATCH --job-name=pca
#SBATCH --error=pca-%j.err
#SBATCH --output=pca-%j.log
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1

python 06pca.py
