/**
 * 3mm.c: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "../common/polybenchUtilFuncts.h"

#include <clAmdBlas.h>

static const clAmdBlasOrder order = clAmdBlasRowMajor;
static const clAmdBlasTranspose trans = clAmdBlasNoTrans;


//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size. */
# define NI 4096
# define NJ 4096
# define NK 4096
# define NL 4096
# define NM 4096

#define ALPHA 1
#define BETA 0

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

typedef float DATA_TYPE;


char str_temp[1024];

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
cl_mem b_mem_obj;
cl_mem c_mem_obj;
cl_mem d_mem_obj;
cl_mem e_mem_obj;
cl_mem f_mem_obj;
cl_mem g_mem_obj;

FILE *fp;
char *source_str;
size_t source_size;




void compareResults(DATA_TYPE *G, DATA_TYPE *G_outputFromGpu)
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < NI; i++)
	{
		for (j=0; j < NL; j++)
		{
			if (percentDiff(G[i*NL + j], G_outputFromGpu[i*NL + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{
				fail++;				
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}

void read_cl_file()
{
	// Load the kernel source code into the array source_str
	fp = fopen("3mm.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
}

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D)
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
			B[i*NJ + j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}
  
	for (i = 0; i < NJ; i++)
	{
		for (j = 0; j < NM; j++)
		{
			C[i*NM + j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}
  
	for (i = 0; i < NM; i++)
	{
		for (j = 0; j < NL; j++)
		{
			D[i*NL + j] = ((DATA_TYPE) i*(j+2)) / NK;
		}
	}
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


void cl_mem_init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E, DATA_TYPE* F, DATA_TYPE* G)
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY, sizeof(DATA_TYPE) * NI * NK, NULL, &errcode);
	b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY, sizeof(DATA_TYPE) * NK * NJ, NULL, &errcode);
	c_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NJ * NM, NULL, &errcode);
	d_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NM * NL, NULL, &errcode);
	e_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NI * NJ, NULL, &errcode);
	f_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NJ * NL, NULL, &errcode);
	g_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NI * NL, NULL, &errcode);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NK, A, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NK * NJ, B, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NJ * NM, C, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, d_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NM * NL, D, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, e_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NJ, E, 0, NULL, NULL);	
	errcode = clEnqueueWriteBuffer(clCommandQue, f_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NJ * NL, F, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, g_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NL, G, 0, NULL, NULL);
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
		
	// Create the OpenCL kernels
	clKernel1 = clCreateKernel(clProgram, "mm3_kernel1", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	clKernel2 = clCreateKernel(clProgram, "mm3_kernel2", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	clKernel3 = clCreateKernel(clProgram, "mm3_kernel3", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	clFinish(clCommandQue);
}

void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel1);
	errcode = clReleaseKernel(clKernel2);
	errcode = clReleaseKernel(clKernel3);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(b_mem_obj);
	errcode = clReleaseMemObject(c_mem_obj);
	errcode = clReleaseMemObject(d_mem_obj);
	errcode = clReleaseMemObject(e_mem_obj);
	errcode = clReleaseMemObject(f_mem_obj);
	errcode = clReleaseMemObject(g_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void mm3_cpu(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G)
{
	int i,j,k;
	
	/* E := A*B */
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			E[i*NJ + j] = 0;
			for (k = 0; k < NK; ++k)
			{
				E[i*NJ + j] += A[i*NK + k] * B[k*NJ + j];
			}
		}
	}
		
	/* F := C*D */
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NL; j++)
		{
			F[i*NL + j] = 0;
			for (k = 0; k < NM; ++k)
			{
				F[i*NL + j] += C[i*NM + k] * D[k*NL + j];
			}
		}
	}

  	/* G := E*F */
	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NL; j++)
		{
			G[i*NL + j] = 0;
			for (k = 0; k < NJ; ++k)
			{
				G[i*NL + j] += E[i*NJ + k] * F[k*NL + j];
			}
		}
	}
}


double cl_blas(){
	int err = -1;
	double t_start, t_end;
	cl_event event = NULL;

	DATA_TYPE alpha = 1.0;
	DATA_TYPE beta = 0.0;

	t_start = rtclock();

        /* Call clAmdBlas function. */
        err = clAmdBlasSgemm(order, trans, trans, NI, NJ, NK, ALPHA, a_mem_obj,
                        NK, b_mem_obj, NJ, BETA, e_mem_obj, NJ, 
			1, &clCommandQue, 0, NULL, &event);
        if (err != CL_SUCCESS) {
            printf("clAmdBlasSgemm() failed with %d\n", err);
        }
	else {
		// Wait for calculations to be finished. 
		err = clWaitForEvents(1, &event);
	}

        /* Call clAmdBlas function. */
        err = clAmdBlasSgemm(order, trans, trans, NI, NL, NM, ALPHA, c_mem_obj,
                        NM, d_mem_obj, NL, BETA, f_mem_obj, NL, 
			1, &clCommandQue, 0, NULL, &event);
        if (err != CL_SUCCESS) {
            printf("clAmdBlasSgemm() failed with %d\n", err);
        }
	else {
		/* Wait for calculations to be finished. */
		err = clWaitForEvents(1, &event);
	}

        /* Call clAmdBlas function. */
        err = clAmdBlasSgemm(order, trans, trans, NI, NL, NJ, ALPHA, e_mem_obj,
                        NJ, f_mem_obj, NL, BETA, g_mem_obj, NL, 
			1, &clCommandQue, 0, NULL, &event);
        if (err != CL_SUCCESS) {
            printf("clAmdBlasSgemm() failed with %d\n", err);
        }
	else {
		/* Wait for calculations to be finished. */
		err = clWaitForEvents(1, &event);
	}



	t_end = rtclock(); 
	fprintf(stdout, "Kerenl time: %0.6lfs\n", t_end - t_start);   

    return t_end - t_start;

}

int main(void) 
{
	double t_start, t_end;


	/* Setup clAmdBlas. */
	int err = clAmdBlasSetup();
	if (err != CL_SUCCESS) {
		printf("clAmdBlasSetup() failed with %d\n", err);                      
		cl_clean_up();
		return 1;                                                              
	} 
	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* C;
	DATA_TYPE* D;
	DATA_TYPE* E;
	DATA_TYPE* F;
	DATA_TYPE* G;
	DATA_TYPE* G_outputFromGpu;

	A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));
	C = (DATA_TYPE*)malloc(NJ*NM*sizeof(DATA_TYPE));
	D = (DATA_TYPE*)malloc(NM*NL*sizeof(DATA_TYPE));
	E = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
	F = (DATA_TYPE*)malloc(NJ*NL*sizeof(DATA_TYPE));
	G = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));
	G_outputFromGpu = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));

	int i;
	init_array(A, B, C, D);
	read_cl_file();
	cl_initialization();

    t_start = rtclock();
	cl_mem_init(A, B, C, D, E, F, G);
    t_end = rtclock();
    double t_copy = t_end - t_start;

	double t_kernel = cl_blas();

    t_start = rtclock();
	errcode = clEnqueueReadBuffer(clCommandQue, g_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NL, G_outputFromGpu, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");
    t_end = rtclock();
    t_copy += t_end - t_start;
    fprintf(stdout, "Copy + kernel  Runtime: %0.6lfms\n", (t_copy + t_kernel)*1000);   

	//t_start = rtclock();
	//mm3_cpu(A, B, C, D, E, F, G);
	//t_end = rtclock(); 
	//fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);   
	//compareResults(G, G_outputFromGpu);
	cl_clean_up();

	free(A);
	free(B);
	free(C);
	free(D);
	free(E);
	free(F);
	free(G);
	free(G_outputFromGpu);

	return 0;
}

