#!/bin/bash
#$ -cwd
#$ -o "matrixmulcompout"
#$ -e "matrixmulcomperr"
#$ -V
cd ./build
/home/cmake/cmake-3.1.3/bin/cmake ..
make
