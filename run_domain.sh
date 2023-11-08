#!/bin/bash
#SBATCH --array=1-10
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
python main.py "Wildfire_MDP_ippc2014" $SLURM_ARRAY_TASK_ID "jaxplan" False True 1