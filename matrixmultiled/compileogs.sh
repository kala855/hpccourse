#!/bin/bash
#$ -cwd
#$ -o "matrixmultiledcompout"
#$ -e "matrixmultiledcomperr"
#$ -V
cd ./build
/home/cmake/cmake-3.1.3/bin/cmake ..
make
