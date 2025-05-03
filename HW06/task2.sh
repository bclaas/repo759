#!/usr/bin/env zsh
#SBATCH --job-name=task2
#SBATCH -p instruction
#SBATCH --job-name=task2
#SBATCH --output=job_output-%j.txt
#SBATCH --time=00:01:00

# Run the task with the desired parameters
for i in {10..29}
do
	n=$((2**i))
	./task2 $n 128 1024
done	
