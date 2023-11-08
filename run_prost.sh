#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
rddlprost $1 $2 $3
