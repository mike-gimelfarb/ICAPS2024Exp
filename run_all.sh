#!/bin/bash
# use case: source ./run_all.sh <domain> <time>
#sbatch run_gurobiplan.sh $1 True $2
sbatch run_jaxplan.sh $1 True $2
sbatch run_noop.sh $1 True $2
#sbatch run_prost.sh $1 True $2
sbatch run_random.sh $1 True $2