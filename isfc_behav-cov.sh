#!/bin/bash

# Run within code directory:
# sbatch isfc_behav-cov.sh

# Set partition
#SBATCH --partition=all

# How long is job (in minutes)?
#SBATCH --time=960

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=2 --mem=16000

# Name of jobs?
#SBATCH --job-name=isfc_behav-cov

# Where to output log files?
#SBATCH --output='logs/isfc_behav-cov_%A_%a.log'

# Number jobs to run in parallel (1-10)
#SBATCH --array=227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,266,268,269,271,310,311,312,313,314,315,316,335,336,337,338,339,340,341,342,343,346,410,552,647,648,649,671,672,673,674,676,677,761,764,766,770

# originally 1-1024

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
echo "Running ISFC covariance job $SLURM_ARRAY_TASK_ID"

python isfc_behav-cov.py $SLURM_ARRAY_TASK_ID

echo "Finished running ISFC covariance job $SLURM_ARRAY_TASK_ID"
date
