SRC := atax

EXECUTABLE := {SRC}.exe
CFILES := {SRC}_host.c

PFLAGS := --ctx="[nx, ny] -> {:nx=ny=4096}"

#PFLAGS +=  --sizes="{kernel[i]->tile[8,4];kernel[i]->block[8,4];kernel[i]->grid[1024,1024]}"
PFLAGS +=  --sizes="{kernel[i]->tile[64,32];kernel[i]->block[32,16];kernel[i]->grid[1024,1024]}"

AUTOTUNER_FLAGS := --ctx=\"[nx, ny] -> {:nx=ny=4096}\"

NUM_TILE_DIMS := 2

include ../../../common.mk
