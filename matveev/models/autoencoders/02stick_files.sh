#!/bin/bash
#SBATCH --job-name=stick_files
#SBATCH --error=stick_files-%j.err
#SBATCH --output=stick_files-%j.log
#SBATCH --time=12:00:00

#SBATCH --cpus-per-task=1

python 01stick_files.py
