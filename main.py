import numpy as np
import scipy as sp
import scipy.linalg
import time
import matplotlib.pyplot as plt

def mat_mul_np(A, B):
  np.matmul(A, B)

def mat_mul_sp(A, B):
  sp.matmul(A, B)

def mat_vec_product_np(A, b):
  np.matmul(A, b)

def mat_vec_product_sp(A, b):
  sp.matmul(A, b)

def dot_product_np(a, b):
  np.dot(a, b)

def dot_product_sp(a, b):
  sp.dot(a, b)

def element_wise_ops_np(A, B):
  A += B
  A *= B

def LU_decomp_np(A):
  n = A.shape[0]
  U = A.copy()
  L = np.identity(n)
  for k in range(n-1):
    for j in range(k+1, n):
      L[j, k] = U[j, k]/U[k,k]
      U[j, k:n] -= L[j, k] * U[k, k:n]

def LU_decomp_sp(A):
  P, L, U = scipy.linalg.lu(A)


def QR_decomp_np(A):
  q, r = np.linalg.qr(A)

def QR_decomp_sp(A):
  q, r = sp.linalg.qr(A)

def eig_np(A):
  vals, vecs = np.linalg.eig(A)


def eig_sp(A):
  vals, vecs = np.linalg.eig(A)

def profile_fn(fn, inputs, iterations):
  total_time = 0
  for it in range(iterations):
    start = time.time()
    fn(*inputs)
    total_time += time.time() - start
  return total_time

def plot_times(times, title):
  tools = ['NumPy', 'SciPy']
  for i, t in enumerate(times):
    plt.bar(i, t, label=tools[i])
  plt.ylabel('Time (seconds)')
  plt.xlabel('Programming Tool')
  plt.title(title)
  plt.legend()
  plt.xticks([],[])
  plt.show()

np.random.seed(42)

times = []
A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)
times.append(profile_fn(mat_mul_np, [A, B], 100))
times.append(profile_fn(mat_mul_sp, [A, B], 100))
plot_times(times, 'Matrix Multiplication')



times = []
A = np.random.rand(1000, 1000)
b = np.random.rand(1000)
times.append(profile_fn(mat_vec_product_np, [A, b], 10000))
times.append(profile_fn(mat_vec_product_sp, [A, b], 10000))
plot_times(times, 'Matrix Vector Multiplication')

times = []
a = np.random.rand(1000)
b = np.random.rand(1000)
times.append(profile_fn(dot_product_np, [a, b], 100000))
times.append(profile_fn(dot_product_sp, [a, b], 100000))
plot_times(times, 'Dot Product')

times = []
A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)
times.append(profile_fn(element_wise_ops_np, [A, B], 1000))
plot_times(times, 'Element-Wise Operations')

times = []
A = np.random.rand(1000, 1000)
times.append(profile_fn(LU_decomp_np, [A], 10))
times.append(profile_fn(LU_decomp_sp, [A], 10))
plot_times(times, 'LU Decomposition')


times = []
A = np.random.rand(1000, 1000)
times.append(profile_fn(QR_decomp_np, [A], 10))
times.append(profile_fn(QR_decomp_sp, [A], 10))
plot_times(times, 'QR Decomposition')

times = []
A = np.random.rand(1000, 1000)
A = A @ A.transpose()
times.append(profile_fn(eig_np, [A], 10))
times.append(profile_fn(eig_sp, [A], 10))
plot_times(times, 'Eigen Decomposition')