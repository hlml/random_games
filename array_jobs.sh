#!/bin/bash
#SBATCH --array=1-4
#SBATCH --account=rrg-bengioy-ad         # Yoshua pays for your job
#SBATCH --cpus-per-task=2                # Ask for 4 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=16G                        # Ask for 32 GB of RAM
#SBATCH --time=3:00:00                   # The job will run for 2 hours
#SBATCH -o /home/hattie/scratch/slurm-%j.out  # Write the log in $SCRATCH
SEED=$SLURM_ARRAY_TASK_ID
GAMMA=0.003

source ~/ilenv/bin/activate
module load httpproxy
wandb login
cd ~/random_game

python train_simple.py --train_data cifar10 --model_choice convnet --save_every 10 --save_init_model --print_every 300 --num_epochs 100 --init_type prev --lr_true 0.1 --mom_true 0.9 --l2_true 0.0005 --model_name EXP_cifar_base --group_vars num_epochs seed lr_true mom_true l2_true train_label_noise mixture_width mixture_depth aug_severity no_jsd all_ops use_augmix --save_path clean_exp_cifar_fast --train_normal --no_jsd --seed ${SEED}