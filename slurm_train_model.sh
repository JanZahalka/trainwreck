#!/bin/bash -l
#SBATCH --job-name=trwrck_train
#SBATCH --partition gpu
#SBATCH --gres=gpu:Volta100:1
#SBATCH --mem=16GB
#SBATCH -o ./slurm_outputs/trwrck_train-%A_%a.out

source ${SLURM_SUBMIT_DIR}/env/bin/activate
python train_model.py