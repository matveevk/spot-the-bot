#!/bin/bash
#SBATCH --job-name=full_preprocess
#SBATCH --error=full_preprocess-%j.err
#SBATCH --output=full_preprocess-%j.log
#SBATCH --time=12:00:00

#SBATCH --cpus-per-task=1

python 01text_preprocess.py
