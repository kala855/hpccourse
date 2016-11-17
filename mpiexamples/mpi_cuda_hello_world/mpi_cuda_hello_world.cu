#include <stdio.h>
#include "someDefinitions.h"
#include <cuda.h>
#include <stdlib.h>


void cudaCall(int rank){
    FILE *fp;
    int device;
    cudaGetDevice(&device);
    char buffer[6];
    sprintf(buffer,"file%d",rank);
    fp = fopen(buffer,"w");
    fprintf(fp,"Number of device is %d\n",device);
}
