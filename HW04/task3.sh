#!/usr/bin/env zsh
#SBATCH --job-name=task3
#SBATCH -p instruction
#SBATCH --job-name=task3
#SBATCH --output=job_output-%j.txt
#SBATCH --time=00:01:00

# Run the task with the desired parameters
./task3 $1 $2 $3
