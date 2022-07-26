#!/bin/bash

# Run within code directory:
# sbatch compare_spis.sh

# Set partition
#SBATCH --partition=all

# How long is job (in minutes)?
#SBATCH --time=840

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=1 --mem=8000

# Name of jobs?
#SBATCH --job-name=compare_spis

# Where to output log files?
#SBATCH --output='logs/compare_spis_%A_%a.log'

# Number jobs to run in parallel (1-4545)
#SBATCH --array=1-1000

# Purge any modules
echo "Purging modules"
module purge

# Load conda environment
echo "Loading conda environment"
source ~/.bashrc
conda activate pyspi

# Print job submission info
echo "Slurm job ID:" $SLURM_JOB_ID
echo "Slurm array task ID:" $SLURM_ARRAY_TASK_ID
date

# Set parameters based on array index
# Add 0, 1000, 2000, 3000, 4000-4545
param=$((0 + $SLURM_ARRAY_TASK_ID - 1))

# Run fMRIPrep script with participant argument
echo "Running PySPI job $param"

python compute_spis.py $param

echo "Finished running PySPI job $param"
date
