#!/bin/bash
sbatch --array=0-100%10 slurm_attack_and_train.sh