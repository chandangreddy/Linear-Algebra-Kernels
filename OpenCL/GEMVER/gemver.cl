/**
 * gemver.cl: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

typedef float DATA_TYPE;


	
__kernel void gemver_kernel(__global DATA_TYPE *a, 
                                      __global DATA_TYPE *u1, __global DATA_TYPE *u2,
                                      __global DATA_TYPE *v1, __global DATA_TYPE *v2,
                                      __global DATA_TYPE *w, __global DATA_TYPE *x,
                                      __global DATA_TYPE *y, __global DATA_TYPE *z,
                                      DATA_TYPE alpha, DATA_TYPE beta, int n) 
{    
	int i = get_global_id(0);

	if (i < n)
	{
		int j;
		for(j = 0; j < n; j++)
              a[i*n + j] = a[i*n + j] + u1[i] * v1[j] + u2[i] * v2[j];

		for(j = 0; j < n; j++)
              x[i] = x[i] + beta * a[j*n + i] * y[j];

        x[i] = x[i] + z[i];

		for(j = 0; j < n; j++)
              w[i] = w[i] +  alpha * a[i*n + j] * x[j];

	}
}

