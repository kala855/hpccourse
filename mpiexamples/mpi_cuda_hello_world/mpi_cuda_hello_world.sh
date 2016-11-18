#!/bin/bash
#
#SBATCH --job-name=mpi_cuda_hello_world
#SBATCH --output=res_mpi_cuda_hello_world.out
#SBATCH --nodes=2
#SBATCH --tasks=2
#SBATCH --gres=gpu:maxwel


export CUDA_VISIBLE_DEVICES=0

mpirun mpi_cuda_hello_world
