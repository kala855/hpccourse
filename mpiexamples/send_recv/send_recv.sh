#!/bin/bash
#
#SBATCH --job-name=send_recv
#SBATCH --output=send_recv.out
#
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun send_recv
