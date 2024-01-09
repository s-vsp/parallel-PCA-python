import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from numba import njit, jit, prange


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


@jit(nopython=True, parallel=True)
def parallel_sort(eigen_vals, eigen_vects):
    indices = np.argsort(np.abs(eigen_vals))[::-1]
    sorted_eigenvalues = eigen_vals[indices]
    sorted_eigenvectors = eigen_vects[:, indices]

    return sorted_eigenvalues, sorted_eigenvectors


@njit(parallel=True)
def parallel_transpose(A: np.ndarray):
    rows, cols = A.shape
    result = np.empty((cols, rows), dtype=A.dtype)

    for i in prange(rows):
        for j in prange(cols):
            result[j, i] = A[i, j]

    return result


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

    for j in range(n):
        v = A[:, j]

        for i in range(j):
            R[i, j] = parallel_matrix_multiplication(Q[:, i][np.newaxis, :], A[:, j][:, np.newaxis]).item()
            # R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R


def qr_algorithm(D: np.ndarray, tol: float = 1e-10, max_iters: int = 1000):
    D = D
    E = np.eye(D.shape[0])

    for i in range(max_iters):
        Q, R = parallel_gram_schmidt_process(D)
        D = parallel_matrix_multiplication(R, Q)
        E = parallel_matrix_multiplication(E, Q)

        if np.max(np.abs(np.triu(D, 1))) < tol:
            break

    eigen_vals = D.diagonal()
    eigen_vects = E

    return eigen_vals, eigen_vects


def svd(M: np.ndarray):  # m x n
    M_T = parallel_transpose(M)  # n x m
    X = parallel_matrix_multiplication(M_T, M)  # n x n
    eigen_vals, eigen_vects = qr_algorithm(X)  # n x 1, n x n
    eigen_vals, eigen_vects = parallel_sort(eigen_vals, eigen_vects)
    singular_vals = np.sqrt(eigen_vals)
    sigma = np.eye(M.shape[0])  # m x m
    temp = np.ones(shape=(M.shape[0],))
    temp[:singular_vals.shape[0]] = singular_vals
    sigma *= temp  # m x m
    sigma_inv = np.linalg.inv(sigma)  # m x m
    V = eigen_vects  # n x n
    V_T = parallel_transpose(V)  # n x n
    U = parallel_matrix_multiplication(M, parallel_matrix_multiplication(V, sigma_inv))
    return U, sigma, V_T


def load_data():
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets
    return X, y


if __name__ == "__main__":
    ##### GRAM-SCHMIDT PARALLELIZATION #####
    # A = np.array([
    #         [12., -51., 4.],
    #         [6., 167., -68.],
    #         [-4., 24., -41.]
    # ])
    # # A = np.random.rand(100,100)
    # # B = np.copy(A)
    # print("Matrix A:")
    # print(A)
    # Q_1, R_1 = gram_schmidt_process(A)
    # print("\nQ matrix:")
    # print(Q_1)
    # print("\nR matrix:")
    # print(R_1)
    # print("\n---------------------------\n")
    # B = np.array([
    #         [12., -51., 4.],
    #         [6., 167., -68.],
    #         [-4., 24., -41.]
    # ])
    # print("Matrix A:")
    # print(B)
    # Q_2, R_2 = parallel_gram_schmidt_process(B)
    # print("\nQ matrix:")
    # print(Q_2)
    # print("\nR matrix:")
    # print(R_2)
    # print(f"Difference between Qs = {np.round(np.sum(Q_1 - Q_2),3)}; Difference between Rs = {np.round(np.sum(R_1 - R_2),3)}")
    #########################################

    ##### QR Algorithm #####
    # x1,x2 = qr_algorithm(A)
    # x1,x2 = parallel_sort(x1,x2)
    # print(x1)
    # print(x2)
    # print("-----------------------------------")
    # y1, y2 = np.linalg.eig(A)
    # print(y1)
    # print(y2)
    #########################################

    ##### SVD #####
    # X = np.array([[4, 0],[3, -5]])
    # U, S, Vh = np.linalg.svd(X)
    # print(U)
    # print("---------")
    # print(S)
    # print("---------")
    # print(Vh)
    # U2, S2, Vh2 = svd(X)
    # print("_______________________")
    # print(U2)
    # print("---------")
    # print(S2)
    # print("---------")
    # print(Vh2)
    #########################################

    ##### PCA #####
    X, y = load_data()
    X = X.to_numpy()
    y = y["Diagnosis"].map({"M": 0, "B": 1})
    U, S, _ = svd(X)
    W = U[:, :2]
    X_low = parallel_matrix_multiplication(X, W)

    print(S)
    plt.plot(S.diagonal()[:10]/sum(S.diagonal()), 'bo')
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.scatter(X_low[:, 0], X_low[:, 1], c=y.to_numpy())
    plt.show()
    #########################################
