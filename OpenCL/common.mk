OpenCL_SDK=/opt/AMDAPP
INCLUDE=-I${OpenCL_SDK}/include
LIBPATH=-L${OpenCL_SDK}/lib/
LIB=-lOpenCL -lm

BLAS_SDK=/opt/clAmdBlas-1.10.321
BLAS_INCLUDE=-I${BLAS_SDK}/include
BLAS_LIBPATH=-L${BLAS_SDK}/lib64
BLAS_LIB=-lclAmdBlas

all:
	gcc -O3 ${INCLUDE} ${LIBPATH}  ${CFILES} -o ${EXECUTABLE} ${LIB}

blas:
	gcc -O3 ${INCLUDE} ${LIBPATH} ${BLAS_INCLUDE} ${BLAS_LIBPATH} ${BLAS_FILES} -o ${B_EXECUTABLE} ${LIB} ${BLAS_LIB}

run:
	./${B_EXECUTABLE}

clean:
	rm -f *~ *.exe
