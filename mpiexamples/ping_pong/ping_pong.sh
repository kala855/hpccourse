#!/bin/bash
#
#SBATCH --job-name=ping_pong
#SBATCH --output=res_ping_pong.out
#
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun ping_pong
