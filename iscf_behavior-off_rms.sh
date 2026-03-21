#!/bin/bash

# Run within code directory:
# sbatch iscf_behavior-off_rms.sh

# Set partition
#SBATCH --partition=all

# How long is job (in minutes)?
#SBATCH --time=600

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=1 --mem=16000

# Name of jobs?
#SBATCH --job-name=iscf_behavior-off_rms

# Where to output log files?
#SBATCH --output='logs/iscf_behavior-off_rms_%A_%a.log'

# Number jobs to run in parallel (1-10)
#SBATCH --array=1-100

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
echo "Running ISCF RMS job $SLURM_ARRAY_TASK_ID"

python iscf_behavior_on-off_rms.py assist $SLURM_ARRAY_TASK_ID

echo "Finished running ISCF RMS job $SLURM_ARRAY_TASK_ID"
date
