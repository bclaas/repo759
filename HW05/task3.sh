#!/usr/bin/env zsh
#SBATCH --job-name=task3
#SBATCH -p instruction
#SBATCH --job-name=task3
#SBATCH --output=job_output-%j.txt
#SBATCH --time=00:01:00

./task3 $1
i
