/**
 * gemm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#include "../common/polybenchUtilFuncts.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i ,j , ld) ((( j )*( ld ))+( i ))

#define GPU_DEVICE 0

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Problem size */
#define NI 4096
#define NJ 4096
#define NK 4096

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 32412.0f
#define BETA 2123.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void gemm(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	int i,j,k;
	
	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NJ; j++)
    	{
			C[i*NJ + j] *= BETA;
	
			for (k = 0; k < NK; ++k)
			{
	  			C[i*NJ + j] += ALPHA * A[i*NK + k] * B[k*NJ + j];
			}
      	}
	}
}


void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	int i, j;

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NK; j++)
		{
      		A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
		}
	}

  	for (i = 0; i < NK; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
      		B[i*NJ + j] = ((DATA_TYPE) i*j) / NJ;
		}
	}

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
      		C[i*NJ + j] = ((DATA_TYPE) i*j) / NJ;
		}
	}
}


void compareResults(DATA_TYPE* C, DATA_TYPE* C_outputFromGpu)
{
	int i, j, fail;
	fail = 0;
	
	// Compare C1 and C2
	for (i=0; i < NI; i++) 
	{
		for (j=0; j < NJ; j++) 
		{
			if (percentDiff(C[i*NJ + j], C_outputFromGpu[i*NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void gemm_kernel(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NJ))
	{	
		c[i * NJ + j] *= BETA;
		int k;
		for(k=0; k < NK; k++)
		{
			c[i * NJ + j] += ALPHA * a[i * NK + k] * b[k * NJ +j];
		}
	}
}


void gemmCuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* C_outputFromGpu)
{
	double t_start, t_end;
	double t_start_k, t_end_k;

	cublasStatus_t stat;
	cublasHandle_t handle;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;

	DATA_TYPE alpha = ALPHA;
	DATA_TYPE beta = BETA;

	stat = cublasCreate(&handle);
	
	t_start = rtclock();

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);

	
	t_start_k = rtclock();

	stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,  NI, NJ, NK, &alpha, 
		A_gpu, NK, B_gpu, NJ, &beta, 
		C_gpu, NI);
	if(stat != CUBLAS_STATUS_SUCCESS){
		printf("Error in culbas sgemv 1, error code = %d\n", stat);
		return;
	}

	cudaDeviceSynchronize();
	t_end_k = rtclock();


	cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);    

	t_end = rtclock();
	fprintf(stdout, "cBLAS kernel : %0.6lf\n", t_end_k - t_start_k);
	fprintf(stdout, "cBLAS copy + kernel : %0.6lf\n", t_end - t_start);
	cublasDestroy(handle);
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
}
	

int main(int argc, char *argv[])
{

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* C;  
	DATA_TYPE* C_outputFromGpu; 

	A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE)); 
	B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));   
	C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 
	C_outputFromGpu = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 

	init(A, B, C);
	
	GPU_argv_init();
	
	gemmCuda(A, B, C, C_outputFromGpu);


        /* 
	double t_start, t_end;
	t_start = rtclock();	
	gemm(A, B, C);
	t_end = rtclock();
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(C, C_outputFromGpu);
        */
	

	free(A);
	free(B);  
	free(C);  
	free(C_outputFromGpu); 

    	return 0;
}

