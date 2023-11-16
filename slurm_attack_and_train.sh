#!/bin/bash -l
#SBATCH --job-name=trwrck
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=16GB
#SBATCH --exclude=node12
#SBATCH -o ./slurm_outputs/trwrck-%A_%a.out

source ${SLURM_SUBMIT_DIR}/env/bin/activate
python attack_and_train.py

#(SBATCH --gres=gpu:Volta100:1)