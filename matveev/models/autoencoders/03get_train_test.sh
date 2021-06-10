#!/bin/bash
#SBATCH --job-name=get_train_test
#SBATCH --error=get_train_test-%j.err
#SBATCH --output=get_train_test-%j.log
#SBATCH --time=12:00:00

#SBATCH --cpus-per-task=1

python 03get_train_test.py
