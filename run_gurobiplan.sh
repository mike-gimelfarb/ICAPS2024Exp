#!/bin/bash
#SBATCH --array=1-10
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
# use case: sbatch run_gurobiplan.sh <domain> <online> <time>
module load gurobi/10.0.3
source ~/baselines/bin/activate
echo "Threads ${SLURM_CPUS_ON_NODE:-1}" > gurobi.env
echo "OutputFlag 0" >> gurobi.env
echo "NonConvex 2" >> gurobi.env
python main.py $1 $SLURM_ARRAY_TASK_ID "gurobiplan" $2 True $3