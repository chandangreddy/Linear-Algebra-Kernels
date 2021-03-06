#!bin/sh

#PPCG_DIR=/home/baghdadi/src/ppcgs/ppcg_master_exact
#PPCG_DIR=/home/chandan/ppcg/ppcg_master_exact
PPCG_DIR=/home/baghdadi/src/ppcgs/pencil-driver-0.2/ppcg
#PENCIL_RUNTIME=/home/baghdadi/src/ppcgs/pencil-driver-0.2/pencil-util/runtime/
PENCIL_UTIL=/home/baghdadi/src/ppcgs/pencil-driver-0.2/pencil-util

PPCG=${PPCG_DIR}/ppcg

BENCHMARK_DIR=${PWD}

#OpenCL_SDK=/opt/AMDAPP
OpenCL_SDK=/usr/local/cuda-5.5

AUTOTUNE_DIR=/home/chandan/autotune/branch/autotuner

TARGET=opencl

POLYBENCH=/home/chandan/linear_algerbra_kernels/polybench-c-3.2

INCLUDE=-I${OpenCL_SDK}/include -I${POLYBENCH}/utilities
LIBPATH=-L${OpenCL_SDK}/lib/
LIB=-lOpenCL -lm

${SRC}_host.c: ${SRC}.c
	${PPCG} ${PFLAGS} --target=opencl -I${POLYBENCH}/utilities  ${SRC}.c


${SRC}_host_best.c: ${SRC}.c
	conf="`tail -2 ${SRC}_autotune.log| head -1 | sed -e 's/^.*ppcg = \(.*\), status = passed.*/\1/'`"; \
	echo  $${conf} ; \
	${PPCG} `eval "echo $${conf}"` -I${POLYBENCH}/utilities  ${SRC}.c -o ${SRC}_host_best.c
	

${SRC}_host.cu: ${SRC}.c
	${PPCG} ${PFLAGS} --target=cuda -I${POLYBENCH}/utilities ${SRC}.c

${SRC}_cuda.exe: ${SRC}_host.cu
	nvcc -O3 ${SRC}_host.cu ${SRC}_kernel.cu   ${POLYBENCH}/utilities/polybench.cu  -I${POLYBENCH}/utilities -DPOLYBENCH_TIME ${CPFLAGS} -o ${SRC}_cuda.exe 

${SRC}_host.exe: ${SRC}_host.c 
	gcc -std=gnu99 -O3 ${INCLUDE} ${LIBPATH} ${SRC}_host.c -DPOLYBENCH_TIME  -I${PENCIL_UTIL}/include  -I${PENCIL_UTIL}/runtime/include ${POLYBENCH}/utilities/polybench.c -L${PENCIL_UTIL}/runtime/src/.libs  -o ${SRC}_host.exe  ${LIB} -lprl -lOpenCL
	

${SRC}_host_best.exe: ${SRC}_host_best.c 
	gcc -std=gnu99 -O3 ${INCLUDE} ${LIBPATH} ${SRC}_host_best.c -DPOLYBENCH_TIME  -I${PENCIL_UTIL}/include  -I${PENCIL_UTIL}/runtime/include ${POLYBENCH}/utilities/polybench.c -L${PENCIL_UTIL}/runtime/src/.libs  -o ${SRC}_host_best.exe  ${LIB} -lprl -lOpenCL
	
opencl: ${SRC}_host.exe 
	PRL_PROFILING=1 ./${SRC}_host.exe 

opencl_best: ${SRC}_host_best.exe 
	PRL_PROFILING=1 ./${SRC}_host_best.exe 

cuda: ${SRC}_cuda.exe
	./${SRC}_cuda.exe

autotune_cuda: 
	python ${AUTOTUNE_DIR}/main.py --target=${TARGET} --log-results-to-file ${SRC}_autotune.log \
	--ppcg-cmd "${PPCG} --target=${TARGET} --dump-sizes ${AUTOTUNER_FLAGS} -I${POLYBENCH}/utilities ${SRC}.c" \
	--build-cmd "nvcc -O3 ${POLYBENCH}/utilities/polybench.cu  -I${POLYBENCH}/utilities -DPOLYBENCH_TIME " \
	--run-cmd "./" \
	--block-size-range 1-2 \
	--grid-size-range 10-10 \
	--tile-size-range 1-2 \
	--tile-dimensions ${NUM_TILE_DIMS} \
	--block-dimensions 2 \
	--grid-dimensions 2 \
	--no-shared-memory \
	--no-private-memory \
	--verbose \
	exhaustive \
	--only-powers-of-two \
	--filter-testcases
	#--params-from-file
	#--parallelize-compilation \
	#--num-compile-threads 4
	#--all-fusion-structures \
	#--params-from-file
	#--execution-time-from-binary \
	#random


autotune_opencl: 
	python ${AUTOTUNE_DIR}/main.py --target=opencl --log-results-to-file ${SRC}_autotune.log \
	--ppcg-cmd "${PPCG} --target=opencl --dump-sizes ${AUTOTUNER_FLAGS} -I${POLYBENCH}/utilities ${SRC}.c" \
	--build-cmd "gcc -std=gnu99 -O3 ${INCLUDE} ${LIBPATH} -DPOLYBENCH_TIME  -I${PENCIL_UTIL}/include  -I${PENCIL_UTIL}/runtime/include ${POLYBENCH}/utilities/polybench.c -L${PENCIL_UTIL}/runtime/src/.libs ${LIB} -lprl -lOpenCL" \
	--run-cmd "./" \
	--runs  1 \
	--block-size-range 1-9 \
	--grid-size-range 10-10 \
	--tile-size-range 4-4 \
	--tile-dimensions  ${NUM_TILE_DIMS} \
	--block-dimensions 2 \
	--grid-dimensions 2 \
	--verbose \
	exhaustive \
	--only-powers-of-two \
	--filter-testcases \
	#--parallelize-compilation \
	#--num-compile-threads 4 \
	#--execution-time-from-binary \
	#--params-from-file
	#--all-fusion-structures \
	#--params-from-file
	#random

clean:
	rm -f ${SRC}_kernel.cl ${SRC}_host.c ${SRC}_kernel.h ${SRC}_host.exe *.cu ${SRC}_cuda.exe ${SRC}_kernel.hu  ${SRC}_*_best*
