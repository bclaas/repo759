#!/usr/bin/env zsh
#SBATCH --job-name=FirstSlurm
#SBATCH --output=FirstSlurm.out
#SBATCH --error=FirstSlurm.err
#SBATCH --time=0-00:01:00
#SBATCH --nodes=2

hostname
