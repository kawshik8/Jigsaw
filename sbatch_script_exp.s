#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=30:00:00
#SBATCH --mem=50000
#SBATCH --job-name=kk4161
#SBATCH --mail-user=kk4161@nyu.edu
#SBATCH --output=slurm_%j.out

source ../env/bin/activate
module load python3/intel/3.5.3

python3 src/main.py --model Exp --transfer-paradigm "tunable" --batch-size 32 --exp-name exp-finetune-whole --results-dir "./results" --data-dir "./data" --finetune-tasks "cifar10_lp100" --pretrain-total-iters 120000 --finetune-total-iters 120000 --finetune_learning_rate 1e-2 --pretrain-learning-rate 1e-2 --pretrain-val-interval 2000 --finetune-val-interval 2000 --dup-pos 8

#python3 src/main.py --batch-size 32 --model Exp --exp-name Exp-pretrain --results-dir "./results" --data-dir "./data" --pretrain-ckpt-interval 10000 --dup_pos 8 --pretrain-learning-rate 1e-3 --transfer-paradigm tunable 



