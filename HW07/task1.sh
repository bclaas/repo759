#!/usr/bin/env zsh
#SBATCH --job-name=task1
#SBATCH -p instruction
#SBATCH --job-name=task1
#SBATCH --output=job_output-%j.txt
#SBATCH --time=00:10:00
#SBATCH --mem=8G

# Run the task with the desired parameters
for i in {5..14}
do
	n=$((2**i))
	./task1 $n $1
done	
