#!/bin/bash
sbatch --array=0-50%5 slurm_attack_and_train.sh