#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=40:00:00
#SBATCH --mem=50000
#SBATCH --job-name=kk4161
#SBATCH --mail-user=kk4161@nyu.edur
#SBATCH --output=slurm_%j.out

source ../env/bin/activate
module load python3/intel/3.5.3

#python3 src/main.py --model selfie --transfer-paradigm "tunable" --batch-size 128 --exp-name selfie-finetune-whole --results-dir "./results" --data-dir "./data" --finetune-tasks "cifar10_lp100" --pretrain-total-iters 120000 --finetune-total-iters 120000 --finetune_learning_rate 1e-2 --pretrain-learning-rate 1e-2 --pretrain-val-interval 2000 --finetune-val-interval 2000

#python3 src/main.py --model selfie --load-ckpt "pretrain_cifar10_un_best.ckpt" --transfer-paradigm "tunable" --batch-size 128 --exp-name selfie-finetune-whole --results-dir "./results" --data-dir "./data" --finetune-tasks "cifar10_lp10" --finetune-total-iters 120000 --finetune_learning_rate 1e-2 --finetune-val-interval 2000 --pretrain-task 'none'

python3 src/main.py --model selfie --batch-size 128 --exp-name selfie-finetune-whole-linear --results-dir "./results" --data-dir "./data" --pretrain-total-iters 120000 --pretrain-val-interval 2000 --pretrain-learning-rate 1e-2 --finetune-tasks cifar10_lp100_res1,cifar10_lp100_res2,cifar10_lp100_res3,cifar10_lp100_res4 --finetune-total-iters 120000 --finetune_learning_rate 1e-2 --finetune-val-interval 2000 
