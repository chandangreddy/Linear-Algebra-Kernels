SRC := atax

EXECUTABLE := {SRC}.exe
CFILES := {SRC}_host.c

PFLAGS := --ctx="[nx, ny] -> {:nx=ny=4096}"

PFLAGS +=  --sizes="{kernel[i]->tile[8,4];kernel[i]->block[8,4];kernel[i]->grid[1024,1024]}"

AUTOTUNER_FLAGS := --ctx=\"[nx, ny] -> {:nx=ny=4096}\"

include ../../../common.mk
