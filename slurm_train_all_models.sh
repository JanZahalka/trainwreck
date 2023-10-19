#!/bin/bash
sbatch --array=0-11%4 slurm_train_model.sh