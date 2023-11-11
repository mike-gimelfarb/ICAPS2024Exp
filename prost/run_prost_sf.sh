#!/bin/bash
#SBATCH --array=1-1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
singularity exec prost.sif rddlprost $1 $2 $3 cp /OUTPUTS .