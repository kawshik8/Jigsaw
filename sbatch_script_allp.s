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

#python3 src/main.py --model Allp --transfer-paradigm "tunable" --batch-size 64 --exp-name allp-finetune-whole --results-dir "./results" --data-dir "./data" --finetune-tasks "cifar10_lp100" --pretrain-total-iters 120000 --finetune-total-iters 120000 --finetune_learning_rate 1e-2 --pretrain-learning-rate 1e-2 --pretrain-val-interval 2000 --finetune-val-interval 2000 --dup-pos 8
### CIFAR 10
python3 src/main.py --model Allp --batch-size 64 --exp-name allp-finetune-whole-linear-nce-1 --results-dir "./results" --data-dir "./data" --finetune-tasks cifar10_lp100_res1,cifar10_lp100_res2,cifar10_lp100_res3,cifar10_lp100_res4,cifar10_lp100 --pretrain-total-iters 120000 --finetune-total-iters 120000 --finetune_learning_rate 1e-2 --pretrain-learning-rate 1e-2 --pretrain-val-interval 2000 --finetune-val-interval 2000 --dup-pos 4
### STL 10
#python3 src/main.py --model Allp --batch-size 64 --exp-name allp-finetune-whole-linear-nce --results-dir "./results" --data-dir "./data" --finetune-tasks stl10_fd0_res1,stl10_fd0_res2,stl10_fd0_res3,stl10_fd0_res4,stl10_fd0 --num-patches 9 --pretrain-task "stl10_un" --pretrain-total-iters 120000 --finetune-total-iters 120000 --pretrain-learning-rate 1e-2 --finetune_learning_rate 1e-2 --pretrain-val-interval 2000 --finetune-val-interval 2000 --dup-pos 4

#python3 src/main.py --transfer-paradigm tunable --model Allp --exp-name Allp-pretrain --results-dir "./results" --data-dir "./data" --pretrain-ckpt-interval 10000 --dup_pos 8 --pretrain-learning-rate 1e-3

python3 src/main.py --model Allp --batch-size 64 --exp-name debug --results-dir "./results" --data-dir "./data" --finetune-tasks cifar10_lp100_res1,cifar10_lp100_res2,cifar10_lp100_res3,cifar10_lp100_res4,cifar10_lp100 --pretrain-total-iters 2 --finetune-total-iters 2 --finetune_learning_rate 1e-2 --pretrain-learning-rate 1e-2 --pretrain-val-interval 2 --finetune-val-interval 2 --dup-pos 4


