#!/bin/bash -l
#SBATCH --job-name=trwrck
#SBATCH --partition gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --mem=16GB
#SBATCH -o ./slurm_outputs/trwrck-%A_%a.out

source ${SLURM_SUBMIT_DIR}/env/bin/activate
python attack_and_train.py