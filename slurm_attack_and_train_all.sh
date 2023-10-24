#!/bin/bash
sbatch --array=0-50%8 slurm_attack_and_train.sh