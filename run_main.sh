#!/bin/bash
#SBATCH --array=1-10
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
python main.py $1 $SLURM_ARRAY_TASK_ID $2 $3 True $4