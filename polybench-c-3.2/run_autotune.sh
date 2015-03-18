
#!/bin/bash

orig=$(pwd)
for src in  'atax' 'bicg' 'gemm' 'gemver' 'mvt' 'gesummv' '2mm' '3mm'
do
    currDir=$orig/linear-algebra/kernels/$src
    echo $currDir
    if [ -d $currDir ]
    then
		cd $currDir
		pwd
		make clean
		make autotune_opencl 
		cd - 
    fi
done
