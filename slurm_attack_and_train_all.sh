#!/bin/bash
sbatch --array=0-50%10 slurm_attack_and_train.sh