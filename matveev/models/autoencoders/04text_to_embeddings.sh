#!/bin/bash
#SBATCH --job-name=text_to_embeddings
#SBATCH --error=text_to_embeddings-%j.err
#SBATCH --output=text_to_embeddings-%j.log
#SBATCH --time=48:00:00

#SBATCH --cpus-per-task=1

python 04text_to_embeddings.py
