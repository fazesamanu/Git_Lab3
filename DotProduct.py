import numpy as np

from numba import jit, prange

@jit(nopython=True, parallel=True)
def parallel_dot(a, b):
    result = np.zeros((len(a), len(b[0])), dtype=np.float64)
    for i in prange(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i, j] += a[i][k] * b[k][j]
    return result

a = np.array([[2, 1], [0, 3]])
b = np.array([[1, 1], [3, 2]])
result = parallel_dot(a, b)

print(result)


