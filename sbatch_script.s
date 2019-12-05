#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=30:00:00
#SBATCH --mem=50000
#SBATCH --job-name=kk4161
#SBATCH --mail-user=kk4161@nyu.edur
#SBATCH --output=slurm_%j.out

source ../env/bin/activate
module load python3/intel/3.5.3

python3 src/main.py --model selfie --transfer-paradigm "tunable" --batch-size 128 --exp-name selfie-revised-pretrain-finetune-final --results-dir "./results" --data-dir "./data" --finetune-tasks "cifar10_lp100" --pretrain-total-iters 120000 --finetune-total-iters 120000 --finetune_learning_rate 1e-2 --pretrain-learning-rate 1e-2 --pretrain-val-interval 2000 --finetune-val-interval 2000

#python3 src/main.py --model selfie1 --transfer-paradigm "tunable" --batch-size 128 --exp-name Selfie-revised-pretrain-final-100 --results-dir "./results" --data-dir "./data" --finetune-tasks "cifar10_lp100" --pretrain-total-iters 100000 --finetune-total-iters 100000 --finetune_learning_rate 1e-2 --pretrain-learning-rate 1e-2

#python3 src/main.py --model selfie1 --transfer_paradigm "tunable" --batch-size 128 --exp-name Selfie-revised-1 --results-dir "./results" --data-dir "./data" --finetune-tasks "cifar10_lp10" --pretrain-total-iters 100000 --pretrain-ckpt-interval 10000 --pretrain-learning-rate 1e-3 --finetune-total-iters 20000  

#python3 src/main.py --model selfie1 --pretrain-total-iters 100000 --finetune-total-iters 100000 --finetune_learning_rate 1e-2 --transfer-paradigm "tunable" --batch-size 128 --exp-name Selfie-revised-pretrain-final-100 --results-dir "./results" --data-dir "./data" --finetune-tasks "cifar10_lp100" --load_ckpt "pretrain_cifar10_un_best.ckpt" 



