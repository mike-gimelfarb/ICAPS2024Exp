#!/bin/bash
#SBATCH --array=1-10
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
# use case: sbatch run_random.sh <domain> <online> <time>
python main.py $1 $SLURM_ARRAY_TASK_ID "random" $2 True $3