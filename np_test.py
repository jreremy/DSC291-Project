import sys
if len(sys.argv) == 2 and sys.argv[1] == 'single':
  import os
  os.environ["OPENBLAS_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1" 
  os.environ["NUMEXPR_NUM_THREADS"] = "1" 
  os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
import time
A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)
def fn(A, B):
  start = time.time()
  for i in range(100):
    np.matmul(A, B)
  print(time.time() - start)
fn(A, B)

