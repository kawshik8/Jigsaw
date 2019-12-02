#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:1
#SBATCH --time=24:00:00
#SBATCH --mem=50000
#SBATCH --job-name=kk4161
#SBATCH --mail-user=kk4161@nyu.edu
#SBATCH --output=slurm_%j.out

python3 src/main.py --transfer-paradigm tunable --model Allp --exp-name Allp-pretrain --results-dir "./results" --data-dir "./data" --pretrain-ckpt-interval 10000 --dup_pos 8 --pretrain-learning-rate 1e-3


