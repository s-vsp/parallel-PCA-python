import numpy as np
from numba import njit, prange

from parallel_matrix_multiplication import parallel_matrix_multiplication


def gram_schmidt_process(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R


@njit(parallel=True)
def parallel_gram_schmidt_process(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in prange(n):
        v = A[:, j]

        for i in range(j):
            R[i, j] = parallel_matrix_multiplication(Q[:, i][np.newaxis, :], A[:, j][:, np.newaxis]).item()
            v -= R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    
    return Q, R


if __name__ == "__main__":
    ##### GRAM-SCHMIDT PARALLELIZATION #####
    A = np.array([
            [12., -51., 4.],
            [6., 167., -68.],
            [-4., 24., -41.]
        ])
    print("Matrix A:")
    print(A)
    Q_1, R_1 = gram_schmidt_process(A)
    print("\nQ matrix:")
    print(Q_1)
    print("\nR matrix:")
    print(R_1)
    print("\n---------------------------\n")
    B = np.array([
            [12., -51., 4.],
            [6., 167., -68.],
            [-4., 24., -41.]
        ])
    print("Matrix A:")
    print(B)
    Q_2, R_2 = parallel_gram_schmidt_process(B)
    print("\nQ matrix:")
    print(Q_2)
    print("\nR matrix:")
    print(R_2)
    print(f"Difference between Qs = {np.round(np.sum(Q_1 - Q_2),3)}; Difference between Rs = {np.round(np.sum(R_1 - R_2),3)}")
    #########################################

    ##### QR Algorithm #####
    # TODO ...
