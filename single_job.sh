#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --mem=8G
python main.py "Wildfire_MDP_ippc2014" "1" "jaxplan" False True 1