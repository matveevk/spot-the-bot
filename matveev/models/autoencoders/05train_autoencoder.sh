#!/bin/bash
#SBATCH --job-name=train_autoencoder
#SBATCH --error=train_autoencoder-%j.err
#SBATCH --output=train_autoencoder-%j.log
#SBATCH --time=24:00:00
#SBATCH --gpus=1

python 05train_autoencoder.py
