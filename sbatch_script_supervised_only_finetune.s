#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:1
#SBATCH --time=08:00:00
#SBATCH --mem=80000
#SBATCH --job-name=rss638
#SBATCH --mail-user=rss638@nyu.edu
#SBATCH --output=slurm_%j.out

python3 src/main.py --report-interval 1 --data-dir ./data/ --results-dir "./results/" --exp-name "baseline_cifar10" --pretrain-task "none" --finetune-tasks "cifar10_lp100" --transfer-paradigm "tunable" --load_ckpt "none" --finetune_learning_rate 3e-4 --finetune-total-iters 100000 --model baseline

python3 src/main.py --report-interval 1 --data-dir ./data/ --results-dir "./results/" --exp-name "selfie_cifar10" --pretrain-task "none" --finetune-tasks "cifar10_lp100" --transfer-paradigm "tunable" --load_ckpt "none" --finetune_learning_rate 3e-4 --finetune-total-iters 100000 --model selfie

python3 src/main.py --report-interval 1 --data-dir ./data/ --results-dir "./results/" --exp-name "Exp_cifar10" --pretrain-task "none" --finetune-tasks "cifar10_lp100" --transfer-paradigm "tunable" --load_ckpt "none" --finetune_learning_rate 3e-4 --finetune-total-iters 100000 --model Exp

python3 src/main.py --report-interval 1 --data-dir ./data/ --results-dir "./results/" --exp-name "Allp_cifar10" --pretrain-task "none" --finetune-tasks "cifar10_lp100" --transfer-paradigm "tunable" --load_ckpt "none" --finetune_learning_rate 3e-4 --finetune-total-iters 100000 --model Allp
