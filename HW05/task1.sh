#!/usr/bin/env zsh
#SBATCH --job-name=task1
#SBATCH -p instruction
#SBATCH --job-name=task1
#SBATCH --output=job_output-%j.txt
#SBATCH --time=00:01:00

nvcc task1.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1
./task1

