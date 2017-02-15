#!/bin/bash

#SBATCH --job-name=slurmexample1
#SBATCH --output=slurmexample1.txt
#
#SBATCH --nodes=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

hostname
sleep 60
