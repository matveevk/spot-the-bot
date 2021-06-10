#!/bin/bash
#SBATCH --job-name=data_prepr
#SBATCH --error=data_prepr-%j.err
#SBATCH --output=data_prepr-%j.log
#SBATCH --time=200:00:00

python dataset_preprocessing.py --inputdir english_texts/EnLit --outputdir english_texts/EnLitPreprocessed
