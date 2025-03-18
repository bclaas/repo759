#!/bin/bash
#SBATCH --job-name=task3_simulation
#SBATCH --output=task3_output.txt
#SBATCH --error=task3_error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=4G

# Run the task with the desired parameters
./task3 $1 $2 $3