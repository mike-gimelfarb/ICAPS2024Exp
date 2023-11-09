#!/bin/bash
#SBATCH --account=rrg-ssanner
#SBATCH --array=1-1
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# use case: sbatch run_jaxplan.sh <domain> <online> <time>
python main.py $1 $SLURM_ARRAY_TASK_ID "jaxplan" $2 True $3