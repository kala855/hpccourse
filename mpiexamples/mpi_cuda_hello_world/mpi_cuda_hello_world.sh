#!/bin/bash
#
#SBATCH --job-name=mpi_cuda_hello_world
#SBATCH --output=res_mpi_cuda_hello_world.out
#SBATCH --nodes=2
#SBATCH --tasks=2
#SBATCH --gres=gpu:1

echo $CUDA_VISIBLE_DEVICES
mpirun mpi_cuda_hello_world
