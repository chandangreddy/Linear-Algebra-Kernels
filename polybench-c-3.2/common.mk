#PPCG_DIR=/home/baghdadi/src/ppcgs/ppcg_master_exact
PPCG_DIR=/home/chandan/ppcg/ppcg_master_exact
PPCG=${PPCG_DIR}/ppcg
POLYBENCH=/home/chandan/polybench/polybench-c-3.2


OpenCL_SDK=/opt/AMDAPP
INCLUDE=-I${OpenCL_SDK}/include -I${POLYBENCH}/utilities
LIBPATH=-L${OpenCL_SDK}/lib/
LIB=-lOpenCL -lm

${SRC}_host.c: ${SRC}.c
	${PPCG} ${PFLAGS} --target=opencl --opencl-print-time-measurements -I${POLYBENCH}/utilities ${SRC}.c

${SRC}_host.cu: ${SRC}.c
	${PPCG} ${PFLAGS} --target=cuda -I${POLYBENCH}/utilities ${SRC}.c

${SRC}_cuda.exe: ${SRC}_host.cu
	nvcc -O3 ${SRC}_host.cu ${SRC}_kernel.cu   ${POLYBENCH}/utilities/polybench.cu  -I${POLYBENCH}/utilities -DPOLYBENCH_TIME ${CPFLAGS} -o ${SRC}_cuda.exe 

${SRC}_host.exe: ${SRC}_host.c 
	gcc -std=gnu99 -O3 ${INCLUDE} ${LIBPATH} ${SRC}_host.c -DPOLYBENCH_TIME  ${POLYBENCH}/utilities/polybench.c ${PPCG_DIR}/ocl_utilities.c -o ${SRC}_host.exe ${LIB}
	
opencl: ${SRC}_host.exe 
	./${SRC}_host.exe 

cuda: ${SRC}_cuda.exe
	./${SRC}_cuda.exe

AUTOTUNE_DIR=/home/chandan/polybench/autotuner-master
autotune: 
	python ${AUTOTUNE_DIR}/main.py --target=cuda --log-results-to-file ${SRC}_autotune.log \
	--ppcg-cmd "${PPCG} --target=cuda --dump-sizes ${AUTOTUNER_FLAGS} -I${POLYBENCH}/utilities ${SRC}.c" \
	--build-cmd "nvcc -O3 ${POLYBENCH}/utilities/polybench.cu  -I${POLYBENCH}/utilities -DPOLYBENCH_TIME " \
	--run-cmd "./" \
	--execution-time-from-binary \
	--block-size-range 1-2 \
	--grid-size-range 1-2 \
	--tile-size-range 1-2 \
	--tile-dimensions 1 \
	--block-dimensions 1 \
	--grid-dimensions 1 \
	--no-shared-memory \
	--no-private-memory \
	exhaustive \
	--only-powers-of-two \
	--parallelize-compilation \
	--num-compile-threads 4
	#--all-fusion-structures \
	#--params-from-file
	#random

clean:
	rm -f ${SRC}_kernel.cl ${SRC}_host.c ${SRC}_kernel.h ${SRC}_host.exe *.cu ${SRC}_cuda.exe ${SRC}_kernel.hu ${SRC}_*.log
