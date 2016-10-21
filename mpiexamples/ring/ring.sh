#!/bin/bash
#
#SBATCH --job-name=ring
#SBATCH --output=res_ring.out
#
#SBATCH --ntasks=16
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun ring
