SRC := bicg

EXECUTABLE := {SRC}.exe
CFILES := {SRC}_host.c

PFLAGS := --ctx="[nx, ny] -> {:nx=ny=4096}"
PFLAGS += --sizes="{kernel[i]->tile[32,8,8];kernel[i]->block[32,4];kernel[i]->grid[1024,1024]}" 

AUTOTUNER_FLAGS := --ctx=\"[nx, ny] -> {:nx=ny=4096}\"

NUM_TILE_DIMS := 2

include ../../../common.mk
