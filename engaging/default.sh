#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=sched_mit_sloan_batch
#SBATCH --time=4-00:00
#SBATCH -o /home/lkap/smalllogs/regression_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leakapelevich@gmail.com
module load julia
srun julia engaging.jl