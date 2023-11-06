#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
python main.py "Wildfire_MDP_ippc2014" "1" "jaxplan" False True 1