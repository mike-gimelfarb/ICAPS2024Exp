#!/bin/bash
#SBATCH --array=1-10
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
# use case: sbatch run_gurobiplan.sh <domain> <online> <time>
module load gurobi/10.0.3
source ~/baselines/bin/activate
grb_ts
python main.py $1 $SLURM_ARRAY_TASK_ID "gurobiplan" $2 True $3