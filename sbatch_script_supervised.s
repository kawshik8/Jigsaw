#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:1
#SBATCH --time=08:00:00
#SBATCH --mem=50000
#SBATCH --job-name=kk4161
#SBATCH --mail-user=kk4161@nyu.edu
#SBATCH --output=slurm_%j.out

python3 src/main.py --report-interval 1 --data-dir ./data/ --results-dir "./results/" --exp-name "check-baseline" --pretrain-task "none" --finetune-tasks "cifar10_lp100" --transfer-paradigm "tunable" --load_ckpt "none" --finetune_learning_rate 3e-4 --finetune-total-iters 100000 --model baseline 
