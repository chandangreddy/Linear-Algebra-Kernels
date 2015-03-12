# This file contains the PPCG parameter values that are explored.
# The exploration script considers each combination of the parameter values.

([
  # Tile sizes
  [(16,16), (32,32), (64,64)],

  # Grid sizes
  [(16,16), (32,32), (256,256), (1024,1024)],

  # Block sizes
  [(1,1), (1,2), (1,4), (1,8),
   (16,16), (32,32), (64,64)],

  #private memory
  [False],

  #Shared memory
  [False, True],

  #Fusion
  ['max', 'min']
])
