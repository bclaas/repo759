#!/usr/bin/env zsh
#SBATCH --job-name=task2
#SBATCH -p instruction
#SBATCH --job-name=task2
#SBATCH --output=job_output-%j.txt
#SBATCH --time=00:01:00

./task2

