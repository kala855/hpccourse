#!/bin/bash
#sbatch --gres=gpu:1

export CUDA_VISIBLE_DEVICES=0
./deviceQuery
