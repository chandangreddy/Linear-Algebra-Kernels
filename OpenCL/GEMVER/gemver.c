/**
 * gemver.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size */
#define N 4096

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 256
#define DIM_LOCAL_WORK_GROUP_Y 1

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif


char str_temp[1024];

DATA_TYPE ALPHA = 43532;
DATA_TYPE BETA = 12313;

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel1;
cl_kernel clKernel2;
cl_kernel clKernel3;
cl_command_queue clCommandQue;
cl_program clProgram;

cl_mem a_mem_obj;
cl_mem w_mem_obj;
cl_mem x_mem_obj;
cl_mem y_mem_obj;
cl_mem z_mem_obj;
cl_mem u1_mem_obj;
cl_mem v1_mem_obj;
cl_mem u2_mem_obj;
cl_mem v2_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;



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


void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("gemver.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}


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


void cl_initialization()
{	
	// Get platform and device information
	errcode = clGetPlatformIDs(1, &platform_id, &num_platforms);
	if(errcode == CL_SUCCESS) printf("number of platforms is %d\n",num_platforms);
	else printf("Error getting platform IDs\n");

	errcode = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform name is %s\n",str_temp);
	else printf("Error getting platform name\n");

	errcode = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform version is %s\n",str_temp);
	else printf("Error getting platform version\n");

	errcode = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
	if(errcode == CL_SUCCESS) printf("number of devices is %d\n", num_devices);
	else printf("Error getting device IDs\n");

	errcode = clGetDeviceInfo(device_id,CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("device name is %s\n",str_temp);
	else printf("Error getting device name\n");
	
	// Create an OpenCL context
	clGPUContext = clCreateContext( NULL, 1, &device_id, NULL, NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating context\n");
 
	//Create a command-queue
	clCommandQue = clCreateCommandQueue(clGPUContext, device_id, 0, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
}


void cl_mem_init(DATA_TYPE *A, 
        DATA_TYPE *u1, DATA_TYPE *u2, DATA_TYPE *v1, DATA_TYPE *v2,
        DATA_TYPE *w, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *z)
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N * N, NULL, &errcode);
	u1_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	u2_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	v1_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	v2_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	w_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	x_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	y_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	z_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * N, NULL, &errcode);
	
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N * N, A, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, u1_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, u1, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, u2_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, u2, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, v1_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, v1, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, v2_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, v2, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, w_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, w, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, x_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, x, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, y_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, y, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, z_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * N, z, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
}


void cl_load_prog()
{
	// Create a program from the kernel source
	clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);

	if(errcode != CL_SUCCESS) printf("Error in creating program\n");

	// Build the program
	errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in building program\n");
		
	// Create the OpenCL kernel
	clKernel1 = clCreateKernel(clProgram, "gemver_kernel", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel1\n");
	clFinish(clCommandQue);
}


void cl_launch_kernel()
{
	double t_start, t_end;

	int n = N;

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)N) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

	t_start = rtclock();
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *)&u1_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), (void *)&u2_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 3, sizeof(cl_mem), (void *)&v1_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 4, sizeof(cl_mem), (void *)&v2_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 5, sizeof(cl_mem), (void *)&w_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 6, sizeof(cl_mem), (void *)&x_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 7, sizeof(cl_mem), (void *)&y_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 8, sizeof(cl_mem), (void *)&z_mem_obj);
	errcode |= clSetKernelArg(clKernel1, 9, sizeof(DATA_TYPE), (void *)&ALPHA);
	errcode |= clSetKernelArg(clKernel1, 10, sizeof(DATA_TYPE), (void *)&BETA);
	errcode |= clSetKernelArg(clKernel1, 11, sizeof(int), (void *)&n);

	if(errcode != CL_SUCCESS) printf("Error in seting arguments1\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel1, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel1\n");
	clFinish(clCommandQue);

	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}


void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel1);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(u1_mem_obj);
	errcode = clReleaseMemObject(u2_mem_obj);
	errcode = clReleaseMemObject(v1_mem_obj);
	errcode = clReleaseMemObject(v2_mem_obj);
	errcode = clReleaseMemObject(w_mem_obj);
	errcode = clReleaseMemObject(x_mem_obj);
	errcode = clReleaseMemObject(y_mem_obj);
	errcode = clReleaseMemObject(z_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


int main(void) 
{
    printf("Hello");
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
    printf("init done loading prog\n");
	read_cl_file();
	cl_initialization();
	cl_mem_init(A, u1, u2, v1, v2, w, x, y, z);
	cl_load_prog();

    printf("done loading prog\n");

	cl_launch_kernel();

	errcode = clEnqueueReadBuffer(clCommandQue, w_mem_obj, CL_TRUE, 0, N*sizeof(DATA_TYPE), w_outputFromGpu, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");

	t_start = rtclock();
	gemver(A, u1, u2, v1, v2, w, x, y, z);
	t_end = rtclock(); 
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);   
	compareResults(w, w_outputFromGpu);
	cl_clean_up();
	
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

