#!/bin/bash
#
#SBATCH --job-name=omp_workshare2
#SBATCH --output=res_ompworkshare2.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

./omp_workshare2
