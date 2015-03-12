/**
 * gemver.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "../../common/polybenchUtilFuncts.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i ,j , ld) ((( j )*( ld ))+( i ))

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size */
#define N 4096

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 43532.0f
#define BETA 12313.0f

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


void gemver(DATA_TYPE *A, DATA_TYPE *u1, DATA_TYPE *u2, DATA_TYPE *v1, DATA_TYPE *v2,
        DATA_TYPE *w, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *z)
{
	int i,j;

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      A[i*N + j] = A[i*N + j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x[i] = x[i] + BETA * A[j*N+i] * y[j];

  for (i = 0; i < N; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      w[i] = w[i] +  ALPHA * A[i*N+j] * x[j];

}



void init(DATA_TYPE* A, 
        DATA_TYPE* u1, DATA_TYPE* u2, DATA_TYPE* v1, DATA_TYPE* v2,
        DATA_TYPE* w, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* z)
{
  	int i, j;

 	for (i = 0; i < N; i++)
    	{
            u1[i] = (DATA_TYPE)i;
            u2[i] = ((DATA_TYPE)i+1)/N/2.0;
            v1[i] = ((DATA_TYPE)i+1)/N/4.0;
            v2[i] = ((DATA_TYPE)i+1)/N/6.0;
            y[i] = ((DATA_TYPE)i+1)/N/8.0;
            z[i] = ((DATA_TYPE)i+1)/N/9.0;
    		x[i] = 0.0;
    		w[i] = 0.0;
            for (j = 0; j < N; j++) 
            {
                A[i*N + j] = ((DATA_TYPE) i*j) / N;
            }
    	}
}


void compareResults(DATA_TYPE* y, DATA_TYPE* y_outputFromGpu)
{
	int i, fail;
	fail = 0;
	
	for (i=0; i<(N); i++) 
	{
		if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
			fail++;
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


void gemverCuda(DATA_TYPE* A, DATA_TYPE* u1, DATA_TYPE* u2, DATA_TYPE* v1, DATA_TYPE* v2, 
		DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* w, DATA_TYPE* z, DATA_TYPE* w_outputFromGpu)
{
	double t_start, t_end;		
	double t_start_k, t_end_k;		

	cublasStatus_t stat;
	cublasHandle_t handle;
	DATA_TYPE alpha = ALPHA;
	DATA_TYPE beta = BETA;
	DATA_TYPE one = 1.0;

	stat = cublasCreate(&handle);

	t_start = rtclock();
	DATA_TYPE *A_gpu, *B_gpu;
	DATA_TYPE *u1_gpu, *u2_gpu;
	DATA_TYPE *v1_gpu, *v2_gpu;
	DATA_TYPE *w_gpu, *x_gpu, *y_gpu, *z_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * N * N);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * N * N);
	cudaMalloc((void **)&u1_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&u2_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&v1_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&v2_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&w_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&z_gpu, sizeof(DATA_TYPE) * N);
	
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(u1_gpu, u1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(u2_gpu, u2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(v1_gpu, v1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(v2_gpu, v2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(w_gpu, w, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(z_gpu, z, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);

	
	t_start_k = rtclock();

	stat = cublasScopy(handle, N*N, A_gpu, 1, B_gpu, 1);
	if(stat != CUBLAS_STATUS_SUCCESS){
		printf("Error in culbas scopy\n");
		return;
	}

	stat = cublasSger(handle, N, N, &one, 
			u1_gpu,1, v1_gpu, 1,
			B_gpu, N);  
	if(stat != CUBLAS_STATUS_SUCCESS){
		printf("Error in culbas sger\n");
		return;
	}

	stat = cublasSger(handle, N, N, &one, 
			u2_gpu,1, v2_gpu, 1,
			B_gpu, N);  
	if(stat != CUBLAS_STATUS_SUCCESS){
		printf("Error in culbas sger\n");
		return;
	}

	stat = cublasScopy(handle, N, z_gpu, 1, x_gpu, 1);
	if(stat != CUBLAS_STATUS_SUCCESS){
		printf("Error in culbas scopy\n");
		return;
	}

	stat = cublasSgemv(handle, CUBLAS_OP_N, N, N, &beta, B_gpu, 
		N, y_gpu, 1, &one, x_gpu, 1);
	if(stat != CUBLAS_STATUS_SUCCESS){
		printf("Error in culbas sgemv\n");
		return;
	}

	stat = cublasSgemv(handle, CUBLAS_OP_N, N, N, &alpha, B_gpu, 
		N, x_gpu, 1,  &one, w_gpu, 1);
	if(stat != CUBLAS_STATUS_SUCCESS){
		printf("Error in culbas sgemv\n");
		return;
	}

	cudaDeviceSynchronize();
	t_end_k = rtclock();

	cudaMemcpy(w_outputFromGpu, w_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);

	t_end = rtclock();
	fprintf(stdout, "cBLAS kernel: %0.6lf\n", t_end_k - t_start_k);
	fprintf(stdout, "cBLAS Runtime: %0.6lf\n", t_end - t_start);

	cublasDestroy(handle);
}


int main(int argc, char *argv[])
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* u1;  
	DATA_TYPE* u2;  
	DATA_TYPE* v1;  
	DATA_TYPE* v2;  
	DATA_TYPE* w;  
	DATA_TYPE* x;  
	DATA_TYPE* y;
	DATA_TYPE* z;
	DATA_TYPE* w_outputFromGpu;
	
	A = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
	u1 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE)); 
	u2 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE)); 
	v1 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE)); 
	v2 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE)); 
	w = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE)); 
	x = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE)); 
	y = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	z = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	w_outputFromGpu = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));

	init(A, u1, u2, v1, v2, w, x, y, z);
	
	GPU_argv_init();
	gemverCuda(A, u1, u2, v1, v2, w, x, y, z, w_outputFromGpu);
	
	init(A, u1, u2, v1, v2, w, x, y, z);
	t_start = rtclock();
	gemver(A, u1, u2, v1, v2, w, x, y, z);
	t_end = rtclock();
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);
	
	compareResults(w, w_outputFromGpu);

	free(A);
	free(u1);  
	free(u2);  
	free(v1);  
	free(v2);  
	free(w);  
	free(x);  
	free(y);
	free(z);
	free(w_outputFromGpu);

	return 0;
}

