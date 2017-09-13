#!/bin/bash
#
#SBATCH --job-name=mpi_hello_world
#SBATCH --output=res_mpi_hello_world.out
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun mpi_hello_world
