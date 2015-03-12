all:
	nvcc -O3 ${CUFILES} -o ${EXECUTABLE} 

blas:
	nvcc -O3 -lcublas ${BLASFILE} -o ${BLASEXE}
clean:
	rm -f *~ *.exe
