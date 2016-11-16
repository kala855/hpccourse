#!/bin/bash
#
#sbatch --gres=gpu:1
#sbatch --ntasks=6

export CUDA_VISIBLE_DEVICES=0

mpirun simpleMPI
