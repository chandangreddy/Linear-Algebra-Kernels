#!/bin/bash
orig=$(pwd)
log_file=${orig}/results
num_runs=30
run_autotuner=0

get_median()
{
    FILE=$1
    sort -n $FILE | nawk 'NF{a[NR]=$1;c++}END {printf (c%2==0)?(a[int(c/2)+1]+a[int(c/2)])/2:a[int(c/2)+1]}'
}

for src in  'atax' 'bicg' 'gemm' 'gemver' 'mvt' 'gesummv' '2mm' '3mm'
do
    currDir=$orig/linear-algebra/kernels/$src
    echo $currDir
    if [ -d $currDir ]
    then
		cd $currDir
		pwd
                if [ ${run_autotuner} = 1 ] ; then 
                    make clean
                    make autotune_opencl 
                fi
                for((i=0;i<${num_runs};i++)) 
                do
                    #echo $i
                    make opencl_best | grep CPU_Total | sed -e 's/CPU_Total:[ ]*\([0-9.]*\) ms/\1/' >> results
                done
                {
                printf "${src}: Total execution time = ">>${log_file}
                get_median results >>${log_file} 
                printf " ms\n" >>${log_file}
		cd -  
                }
    fi
done
