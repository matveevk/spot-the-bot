#!/bin/bash
#SBATCH --job-name=kpm-cluster
#SBATCH --error=kpm-cluster-%j.err
#SBATCH --output=kpm-cluster-%j.log
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1

python 01clusterize.py -T glove_pretrain_train_embeddings.npy -r glove_embeddings_real.npy -g glove_embeddings_gen.npy
