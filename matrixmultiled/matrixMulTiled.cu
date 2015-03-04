#include<cuda.h>
#include<stdio.h>
#include<time.h>
#include<fstream>

#define TILE_WIDTH 32


__global__ void matrixMulKernelTiled(int *d_M, int *d_N, int *d_P, int width){
    __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for(int m = 0; m < width / TILE_WIDTH; ++m){
	Mds[ty][tx] = d_M[row*width + m*TILE_WIDTH + tx];
	Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty) * width + col];
	__syncthreads();

	for(int k = 0; k < TILE_WIDTH; ++k){
		Pvalue += Mds[ty][k] * Nds[k][tx];	    
	}
	__syncthreads();
    }
    d_P[row*width+col] = Pvalue;
}

__global__ void matrixMulKernel(int *d_M, int *d_N, int *d_P, int width){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int Pvalue;
    if((row < width)&&(col < width)){
        Pvalue = 0;
        for (int k = 0; k < width ; ++k){
            Pvalue += d_M[row*width+k] * d_N[k*width+col];
        }
        d_P[row*width+col] = Pvalue;
    }
}

int matrixMulHost(int *h_M, int *h_N, int *h_P, int width){
    int Pvalue;

    for(int row = 0; row < width ; ++row){
        for(int col = 0; col < width ; ++col){
            Pvalue = 0;
            for(int k = 0; k < width ; ++k){
                Pvalue += h_M[row*width+k] * h_N[k*width+col];
            }
            h_P[row*width+col] = Pvalue;
        }
    }
    return 0;
}

int initValues(int *data, int width){
    for(int i = 0; i < width*width; i++)
        data[i] = 2;
    return 0;
}

int printData(int *data, int width){
    for(int i = 0; i < width; ++i){
        for(int j = 0; j < width; ++j){
            printf("%d ", data[(i*width)+j]);
        }
        printf("\n");
    }
    return 0;
}

int testValues(int *A, int *B, int width){

    for(int i = 0; i < width; ++i){
        for(int j = 0; j < width; ++j){
            if(A[(i*width)+j]!=B[(i*width)+j]){
                printf("Mal Cálculo...\n");
                return 0;
            }
        }
    }
    printf("Buen Cálculo ...\n");
    return 0;
}

int main(){
    int *h_M, *h_N, *h_P,*h_P_d;
    int *d_M, *d_N,*d_P;
    int width = 2048;
    cudaError_t error = cudaSuccess;
    int size = width * width * sizeof(int);
    clock_t start, end, startGPU, endGPU;
    double cpu_time_used, gpu_time_used;

    h_M = (int*)malloc(size);
    h_N = (int*)malloc(size);
    h_P = (int*)malloc(size);
    h_P_d = (int*)malloc(size);

    if(h_P_d == NULL)
        return 0;

    initValues(h_M, width);
    initValues(h_N, width);

    /////////Algoritmo Secuencial////////////////////////////////////////////
    start = clock();
    matrixMulHost(h_M, h_N, h_P, width);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Tiempo algoritmo secuencial: %.10f\n", cpu_time_used);
    /////////Algoritmo Secuencial/////////////////////////////////////////////

    error = cudaMalloc((void**)&d_M,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_M");
        exit(0);
    }

    error = cudaMalloc((void**)&d_N,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_N");
        exit(0);
    }

    error = cudaMalloc((void**)&d_P,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_P");
        exit(0);
    }

    //////////////////////Algoritmo Paralelo///////////////////////////
    startGPU = clock();
    error = cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando datos a d_M");
        exit(0);
    }

    error = cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando datos a d_N");
        exit(0);
    }

    int blockSize = 32;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(width/float(blockSize)),ceil(width/float(blockSize)),1);
    //matrixMulKernel<<<dimGrid,dimBlock>>>(d_M,d_N,d_P,width);
    matrixMulKernelTiled<<<dimGrid,dimBlock>>>(d_M,d_N,d_P,width);
    cudaDeviceSynchronize();
    cudaMemcpy(h_P_d,d_P,size,cudaMemcpyDeviceToHost);
    endGPU = clock();
    gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
    printf("Tiempo algoritmo paralelo: %.10f\n", gpu_time_used);
    printf("La aceleración obtenida es de %.10fX\n",cpu_time_used/gpu_time_used);
    ///////////////////////Algoritmo Paralelo////////////////////////////

    testValues(h_P_d,h_P,width);

    free(h_M);
    free(h_N);
    free(h_P);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    return 0;
}
