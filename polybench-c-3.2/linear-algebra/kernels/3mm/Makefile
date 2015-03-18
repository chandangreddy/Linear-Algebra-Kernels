SRC := 3mm

EXECUTABLE := {SRC}.exe
CFILES := {SRC}_host.c


PFLAGS +=  --sizes="{kernel[i]->tile[64,64,16];kernel[i]->block[8,16];kernel[i]->grid[1024,1024]}"

PFLAGS += --ctx="[ni, nj, nk, nl, nm] -> {:ni=nj=nk=nl=nm=4096}" 

AUTOTUNER_FLAGS += --ctx=\"[ni, nj, nk, nl, nm] -> {:ni=nj=nk=nl=nm=4096}\" 

NUM_TILE_DIMS := 3

include ../../../common.mk
