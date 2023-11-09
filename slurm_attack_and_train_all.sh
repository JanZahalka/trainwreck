#!/bin/bash
sbatch --array=0-500%12 slurm_attack_and_train.sh