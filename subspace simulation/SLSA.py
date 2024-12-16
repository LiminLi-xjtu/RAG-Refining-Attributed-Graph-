import numpy as np
from scipy.sparse import eye  # Import the eye function from the SciPy library for creating sparse identity matrices
from scipy.sparse.linalg import eigs  # Import the eigs function from the SciPy library for computing eigenvalues and eigenvectors of sparse matrices
import scipy.sparse as sp
from utiles import clustering_metrics, spectral_clusteringA_mean, spectral_clusteringX_mean, kmeans_clustering_mean
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="Graph is not fully connected", category=UserWarning)
warnings.filterwarnings("ignore", message="Array is not symmetric")


def SLSA(A, gnd, eta, n_class, theta, tau, max_iter=500, Z=None):
    """
    Simultaneously sparse and low-rank approximation (SLSA)
    """
    n = A.shape[0]  # Get the number of rows in the input matrix A
    scale = np.sqrt(n_class) / np.linalg.norm(A, 'fro')  # Compute a scaling factor based on the number of classes and the Frobenius norm of A
    A = (A + A.T) * (scale / 2)
    A_abs = np.abs(A)

    if Z is None:
        Z = A  # initialize Z to be equal to A

    error = []
    acc = []
    nmi = []
    F1 = []

    err = 1  # Initialize the error to be greater than the convergence tolerance
    iter = 0  # Initialize the iteration counter


    ac, _, nm, _, f1, _ = spectral_clusteringA_mean(A, gnd, n_iter=50, k=3)

    while err > tau and iter < max_iter:  # While the error is greater than the convergence tolerance and the maximum number of iterations has not been reached,

        iter += 1  # increment the iteration counter
        Z_old = Z  # store the current value of Z

        # Update U
        eigenvectors = np.linalg.eigh(Z)[1]
        U = eigenvectors[:, -n_class:]

        # Update Z
        Z = (1 / (1 + theta)) * (A + (theta * U) @ U.T)
        D = Z * (Z > 0)


        # Update Z to T_eta(Z,D) by truncating Z:
        Z = shrink(Z, D, eta, n)
        err = np.linalg.norm(Z - Z_old, 'fro')

        ac, _, nm, _, f1, _ = spectral_clusteringA_mean(Z, gnd, n_iter=1, k=3)
        acc.append(ac)
        nmi.append(nm)
        F1.append(f1)

    return Z, U, acc, nmi, F1



def shrink(Z, D, eta, n):  # Define a helper function for truncating a matrix
    if eta <= 1:  # If eta is less than or equal to one,
        return np.diag(np.diag(Z))  # return a diagonal matrix with the same diagonal elements as Z
    eta_1 = int(eta / 2)  # Compute half of eta rounded down to an integer
    dd = np.triu(D, 1).flatten()  # Extract the upper triangular part of D above its main diagonal and flatten it into a one-dimensional array
    dd.sort()  # Sort dd in ascending order
    dd = dd[::-1]  # Reverse dd to obtain it in descending order
    return Z * ((np.eye(n) + (D >= dd[eta_1])) != 0)


