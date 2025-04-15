#!/usr/bin/env zsh
#SBATCH --job-name=task1
#SBATCH -p instruction
#SBATCH --job-name=task1
#SBATCH --output=job_output-%j.txt
#SBATCH --time=00:01:00

# Run the task with the desired parameters
for i in {5..14}
do
	n=$((2**i))
	./task1 $n 1024
done	
