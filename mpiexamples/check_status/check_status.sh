#!/bin/bash
#
#SBATCH --job-name=check_status
#SBATCH --output=res_check_status.out
#
#SBATCH --ntasks=2
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun check_status
