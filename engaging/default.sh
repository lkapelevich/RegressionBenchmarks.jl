#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --time=1-00:00
#SBATCH -o /home/lkap/research/logs/mlhw_%j.out
module load julia
srun julia engaging.jl