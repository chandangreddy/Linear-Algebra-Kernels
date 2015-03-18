SRC := gesummv

EXECUTABLE := {SRC}.exe
CFILES := {SRC}_host.c

PFLAGS += --ctx="[n] -> {:n=4096}" #--no-shared-memory --no-private-memory

PFLAFS +=  --sizes="{kernel[i]->tile[8,4,8];kernel[i]->block[8,4];kernel[i]->grid[1024,1024]}" 

AUTOTUNER_FLAGS += --ctx=\"[n] -> {:n=4096}\" 

NUM_TILE_DIMS := 2

include ../../../common.mk
