#!/bin/bash

# Run within social-ctf/ directory:
# sbatch tsne_slurm.sh

# Set partition
#SBATCH --partition=all

# How long is job (in minutes)?
#SBATCH --time=4000

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=32 --mem-per-cpu=8000

# Name of jobs?
#SBATCH --job-name=tsne

# Where to output log files?
#SBATCH --output='logs/test_tsne_new_data_%A_%a.log'

# Remove modules because Singularity shouldn't need them
echo "Purging modules"
module purge
source ~/.bashrc
conda activate tsne

# Print job submission info
date

# Run t-SNE script with index argument
echo "Running t-SNE"

./tsne_slurm.py

echo "Finished running t-SNE"
date
