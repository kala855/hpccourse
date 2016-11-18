#include <stdio.h>
#include <stdlib.h>
#include "someDefinitions.h"
#include <cuda.h>
#include <stdlib.h>

// Error handling macro
#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        printf("CUDA error calling, code is %d\n", err);}



__global__ void test (float *number, float *res){
    res[0] = number[0] * number[0];
}

void cudaCall(int rank, float *number){
    FILE *fp;
    cudaDeviceProp prop;
    int device;

    CUDA_CHECK(cudaGetDevice(&device));

    cudaGetDeviceProperties(&prop, device);

    char buffer[6];
    sprintf(buffer,"file%d",rank);
    fp = fopen(buffer,"w");
    fprintf(fp,"Number of device is %d and the name is %s\n",device, prop.name);
    fclose(fp);


    float *h_res = NULL;
    h_res = (float*)malloc(1*sizeof(float));
    float *d_res = NULL;
    cudaMalloc((void**)&d_res, 1*sizeof(float));

    float *d_number = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_number, 1*sizeof(float)));

    cudaMemcpy(d_number,number, 1*sizeof(float), cudaMemcpyHostToDevice);

    test<<<1,1>>>(d_number, d_res);

    cudaMemcpy(h_res, d_res, 1*sizeof(float), cudaMemcpyDeviceToHost);

    number[0] = h_res[0];

    cudaFree(d_res);
    free(h_res);
}
