#!/bin/bash

#SBATCH --job-name=slurmexample2
#SBATCH --output=slurmexample2.txt
#
#SBATCH --nodes=6
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

srun hostname
