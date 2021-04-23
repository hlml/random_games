#!/bin/bash
#SBATCH --account=rrg-bengioy-ad         # Yoshua pays for your job
#SBATCH --cpus-per-task=4                # Ask for 4 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=4:00:00                   # The job will run for 2 hours
#SBATCH -o /home/hattie/scratch/slurm-%j.out  # Write the log in $SCRATCH

source ~/ilenv/bin/activate
module load httpproxy
wandb login
cd ~/random_game

python train_simple.py --train_data cifar10 --model_choice convnet --alpha 1 --gamma 0.003 --save_every 10 --save_init_model --print_every 300 --num_epochs 100 --init_type true --decay_gamma_rate 0.95 --lr_true 0.1 --lr_rand 0.1 --mom_true 0.9 --l2_true 0.0005 --l2_rand 0.0 --decay_gamma_every 300 --simultaneous --model_name EXP_cifar_random_game --group_vars num_iter_true num_iter_rand num_iter_rand_sb alpha gamma num_epochs init_type seed lr_true lr_rand mom_true mom_rand l2_true l2_rand consistent_rand simultaneous decay_gamma_rate train_label_noise use_augmix mixture_width mixture_depth aug_severity no_jsd all_ops --save_path clean_exp_cifar_fast --num_iter_rand_sb 1 --no_jsd