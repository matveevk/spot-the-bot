#!/bin/bash
#SBATCH --job-name=get_embeddings
#SBATCH --error=get_embeddings-%j.err
#SBATCH --output=get_embeddings-%j.log
#SBATCH --time=24:00:00

#SBATCH --cpus-per-task=1

python text_preprocess.py --inputfile english_texts/big_eng_lstm.txt -o english_texts/big_eng_lstm_2000.txt
python get_embeddings.py --debug --filepath english_texts/big_eng_lstm.txt
