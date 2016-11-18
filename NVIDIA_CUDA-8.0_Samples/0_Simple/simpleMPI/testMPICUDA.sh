#!/bin/bash
#
#SBATCH --nodes=2
#SBATCH --tasks=2
#SBATCH --gres=gpu:maxwel

export CUDA_VISIBLE_DEVICES=0

mpirun simpleMPI
