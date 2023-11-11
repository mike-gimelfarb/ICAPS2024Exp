#!/bin/bash
# launches parallel jobs for all baselines
# use case: source ./run.sh <domain> <online> <time>
sbatch run_gurobiplan.sh $1 $2 $3
sbatch run_jaxplan.sh $1 $2 $3
if [ $2 = "true" ]; then
	sbatch run_noop.sh $1 $2 $3
	sbatch run_random.sh $1 $2 $3	
fi
