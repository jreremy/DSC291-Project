import sys
import time
if len(sys.argv) == 2 and sys.argv[1] == 'single':
  import os
  print('Profiling NumPy with a Single CPU Core')
  os.environ["OPENBLAS_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1" 
  os.environ["NUMEXPR_NUM_THREADS"] = "1" 
  os.environ["OMP_NUM_THREADS"] = "1" 
else: 
  print('Profiling NumPy with Multi CPU Cores')

import numpy as np
def profile_fn(fn, inputs, iterations):
  total_time = 0
  for it in range(iterations):
    start = time.time()
    fn(*inputs)
    total_time += time.time() - start
  return total_time

def mat_mul_np(A, B):
  C = np.matmul(A, B)
  return C

A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)
print(f'Matrix Multiplication Time: {profile_fn(mat_mul_np, [A, B], 100)} seconds')

