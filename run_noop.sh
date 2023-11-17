#!/bin/bash
#SBATCH --array=1-10
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
# use case: sbatch run_noop.sh <domain> <online> <time>
python main.py $1 $SLURM_ARRAY_TASK_ID "noop" $2 True $3