import numpy as np
import pandas as pd

from typing import Tuple, Union
from ucimlrepo import fetch_ucirepo
from mpi4py import MPI


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads Wisconsin Breast Cancer data directly from UCI repos.
    """ 
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets
    return X, y


def parallel_covariance_matrix(X: Union[pd.DataFrame, np.ndarray], comm) -> np.ndarray:
    """
    Computes covariance matrix in parallel manner by computing partial covariance matrices on available processes.

    Args:
        - X: Data matrix/array rows x colums
        - comm: MPI communicator
    
    Returns:
        - covariance_matrix: Computed covariance matrix from the matrix X
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    world = comm.size
    rank = comm.Get_rank()
    
    N, M = X.shape
    
    # Local (per process)
    # Split rows into MPI processes
    rows = np.array_split(X, world, axis=0)[rank]
    means = np.sum(rows, axis=0) / rows.shape[0]
    means = comm.bcast(means, root=0)

    # Partial / Local Covariance Matrix
    partial_cov = np.zeros((M,M))
    for i in range(M):
        for j in range(M):
            partial_cov[i,j] = np.sum((rows[:, i] - means[i]) * (rows[:, j] - means[j])) / (N - 1)
    
    # Main / Global Covariance Matrix
    cov = comm.gather(partial_cov, root=0)
    if rank == 0:
        covariance_matrix = np.sum(cov, axis=0)
    else:
        covariance_matrix = None
    
    covariance_matrix = comm.bcast(covariance_matrix, root=0)
    return covariance_matrix
    

def run():
    comm = MPI.COMM_WORLD
    X, y = load_data()
    cov = parallel_covariance_matrix(X, comm)

    if comm.Get_rank() == 0:
        print("Parallel implementation:\n")
        print(cov)
        print("NumPy build-in:\n")
        print(np.cov(X, rowvar=False))


if __name__ == "__main__":
    run()