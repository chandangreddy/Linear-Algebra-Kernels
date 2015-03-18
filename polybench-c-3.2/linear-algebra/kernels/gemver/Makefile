SRC := gemver

EXECUTABLE := {SRC}.exe
CFILES := {SRC}_host.c

PFLAGS := --isl-schedule-fuse=min
PFLAGS += --ctx="[n] -> {:n=4096}"

AUTOTUNER_FLAGS += --ctx=\"[n] -> {:n=4096}\"

NUM_TILE_DIMS := 2

include ../../../common.mk
