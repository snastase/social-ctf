#!/bin/bash

# Run within code directory:
# sbatch iscf_behavior-on_rms.sh

# Set partition
#SBATCH --partition=all

# How long is job (in minutes)?
#SBATCH --time=720

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=1 --mem=16000

# Name of jobs?
#SBATCH --job-name=iscf_behavior-on_rms

# Where to output log files?
#SBATCH --output='logs/iscf_behavior-on_rms_%A_%a.log'

# Purge any modules
echo "Purging modules"
module purge

# Load conda environment
echo "Loading conda environment"
source ~/.bashrc
conda activate ctf

# Print job submission info
echo "Slurm job ID:" $SLURM_JOB_ID
date

# Run fMRIPrep script with participant argument
python iscf_behavior_on-off_rms.py assist on

echo "Finished running ISCF RMS job"
date

