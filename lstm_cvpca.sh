#!/bin/bash

# Run within code directory:
# sbatch lstm_cvpca.sh

# Set partition
#SBATCH --partition=all

# How long is job (in minutes)?
#SBATCH --time=600

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=16 --mem=256000

# Name of jobs?
#SBATCH --job-name=lstm_cvpca

# Where to output log files?
#SBATCH --output='logs/lstm_cvpca_%A_%a.log'

# Number jobs to run in parallel (1-10)
#SBATCH --array=23,24,27,28,29,30,31

# Purge any modules
echo "Purging modules"
module purge

# Load conda environment
echo "Loading conda environment"
source ~/.bashrc
conda activate ctf

# Print job submission info
echo "Slurm job ID:" $SLURM_JOB_ID
echo "Slurm array task ID:" $SLURM_ARRAY_TASK_ID
date

# Run fMRIPrep script with participant argument
echo "Running cvPCA job $SLURM_ARRAY_TASK_ID"

python lstm_cvpca.py $SLURM_ARRAY_TASK_ID

echo "Finished running cvPCA job $SLURM_ARRAY_TASK_ID"
date
