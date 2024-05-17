#!/bin/bash

#SBATCH --job-name=ts # Job name
#SBATCH --gres=gpu:1             # how many gpus would you like to use (here I use 1)
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=32G                  # Job memory request
#SBATCH --time=4:00:00              # Time limit hrs:min:sec
#SBATCH --partition=ava_m.p          # partition name
#SBATCH --output=logs/job_%j.log   # output log
#SBATCH --exclude=ava-m5,ava-m6 # node to not use
#SBATCH --array=0

~/anaconda3/envs/conformal/bin/python -m src.experiments.covid_exp



