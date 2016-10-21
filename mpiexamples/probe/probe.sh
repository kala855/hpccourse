#!/bin/bash
#
#SBATCH --job-name=probe
#SBATCH --output=res_probe.out
#
#SBATCH --ntasks=2
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun probe
