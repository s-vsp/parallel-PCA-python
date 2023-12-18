import numpy as np
from numba import njit, prange
import time

@njit(parallel=True)
def parallel_matrix_multiplication(a, b):
    a_shape0 = a.shape[0]
    a_shape1 = a.shape[1]
    b_shape1 = b.shape[1]
    c = np.zeros((a_shape0, b_shape1))
    for i in prange(a_shape0):
        for j in range(b_shape1):
            result = 0.0
            for k in range(a_shape1):
                result += a[i, k] * b[k, j]
            c[i, j] = result
    return c


def sequential_matrix_multiplication(a, b):
    shape0 = a.shape[0]
    shape1 = b.shape[1]
    c = np.zeros((shape0, shape1))
    for i in range(shape0):
        for j in range(shape1):
            for k in range(a.shape[1]):
                c[i, j] += a[i, k] * b[k, j]
    return c


# a = np.random.rand(500, 200)
# b = np.random.rand(200, 500)

# numpy_start_time = time.time()
# numpy_result = np.matmul(a, b)
# numpy_end_time = time.time()
# numpy_time = numpy_end_time - numpy_start_time
# print("numpy time:", numpy_time)

# parallel_start_time = time.time()
# parallel_result = parallel_matrix_multiplication(a, b)
# parallel_end_time = time.time()
# parallel_time = parallel_end_time - parallel_start_time
# print("parralell time:", parallel_time)

# sequential_start_time = time.time()
# sequential_result = sequential_matrix_multiplication(a, b)
# sequential_end_time = time.time()
# sequential_time = sequential_end_time - sequential_start_time
# print("seqential time:", sequential_time)

# same = True
# for i in range(numpy_result.shape[0]):
#     for j in range(numpy_result.shape[1]):
#         if abs(numpy_result[i,j] - parallel_result[i,j]) > 0.00000001:
#             same = False
#             print(numpy_result[i,j], parallel_result[i,j])

# print("results ok:", same)