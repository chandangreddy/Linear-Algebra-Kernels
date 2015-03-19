#!/bin/bash
orig=$(pwd)
log_file=${orig}/results_blas
num_runs=3
run_autotuner=0

get_median()
{
    FILE=$1
    sort -n $FILE | nawk 'NF{a[NR]=$1;c++}END {printf (c%2==0)?(a[int(c/2)+1]+a[int(c/2)])/2:a[int(c/2)+1]}'
}

for src in  'ATAX' 'BICG' 'GEMM' 'GEMVER' 'MVT' 'GESUMMV' '2MM' '3MM'
do
    currDir=$orig/$src
    echo $currDir
    if [ -d $currDir ]
    then
		cd $currDir
		pwd
		make clean
		#make blas
                #./*.exe
                for((i=0;i<${num_runs};i++)) 
                do
                    #echo $i
                    make run | grep Copy | sed -e 's/Copy + kernel Runtime:[ ]*\([0-9.]*\)ms/\1/' >> results
                done
                {
                printf "${src}: Total execution time = ">>${log_file}
                get_median results >>${log_file} 
                printf " ms\n" >>${log_file}
		cd -  
                }
		cd ..
    fi
done
