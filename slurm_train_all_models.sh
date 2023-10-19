#!/bin/bash
sbatch --array=0-11%4 poc_slurm.sh