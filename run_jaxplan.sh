#!/bin/bash
#SBATCH --array=1-10
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
# use case: sbatch run_jaxplan.sh <domain> <online> <time>
python main.py $1 $SLURM_ARRAY_TASK_ID "jaxplan" $2 True $3