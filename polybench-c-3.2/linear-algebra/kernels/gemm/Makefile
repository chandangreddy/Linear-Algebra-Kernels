SRC := gemm

EXECUTABLE := {SRC}.exe
CFILES := {SRC}_host.c

PFLAGS +=  --sizes="{kernel[i]->tile[64,64,16];kernel[i]->block[8,16];kernel[i]->grid[1024,1024]}"
PFLAGS += --ctx="[ni, nj, nk] -> {:ni=nj=nk=4096}" 
#PFLAGS += --ctx="[ni, nj, nk] -> {:ni mod 128 = 0, nj mod 128 = 0,  nk mod 128 = 0}" 

AUTOTUNER_FLAGS += --ctx=\"[ni, nj, nk] -> {:ni=nj=nk=4096}\" 

include ../../../common.mk
