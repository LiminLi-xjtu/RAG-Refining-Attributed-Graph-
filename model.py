import numpy as np  # Import the NumPy library for numerical computing
from scipy import linalg
import scipy.sparse as sp
from scipy.sparse import eye, issparse, triu  # Import the eye function from the SciPy library for creating sparse identity matrices
from scipy.sparse.linalg import eigs  # Import the eigs function from the SciPy library for computing eigenvalues and eigenvectors of sparse matrices
import scipy.io as sio
from utiles import clustering_metrics, update_local_best, filter_features_sparse, spectral_clustering
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import time




def RAG(A, X, gnd, eta, alpha, beta, lamb, t, max_iter=10):
    """
    Attributed Graph Refinement via low rank approximation and subspace learning (RAG)
    """
    gamma = 0.001
    n = A.shape[0]
    k = len(np.unique(gnd))

    if isinstance(A, np.ndarray):
        Anorm = np.linalg.norm(A, ord=None)
    else:
        Anorm = np.linalg.norm(A.A, ord=None)

    scale = np.sqrt(k) / Anorm
    As = (A + A.T) * (scale / 2)


    # filter X
    I = np.eye(n)
    X = filter_features_sparse(A, X, t)
    if issparse(X):
        X = X.todense()

    Xnorm = np.linalg.norm(X, ord=None)
    scaleX = np.sqrt(n) / Xnorm
    X = X * scaleX
    alpha_XX = alpha * np.dot(X, X.T)


    Z = 0
    Y = As
    acc = []
    nmi = []
    F1 = []

    y_A = SpectralClustering(n_clusters=k, assign_labels="discretize", random_state=23, affinity='precomputed').fit(As).labels_
    cm = clustering_metrics(gnd, y_A)
    ac, nm, f1 = cm.evaluationClusterModelFromLabel()
    acc.append(ac)
    # print('ORIGINAL A : acc_mean: {:.4f}'.format(ac),
    #     'nmi_mean: {:.4f}'.format(nm),
    #     'f1_mean: {:.4f}'.format(f1))

    local_best_acc = 0.0
    local_best_nmi = 0.0
    local_best_f1 = 0.0
    local_best_iter = 0

    err = 1
    iter = 0
    tau = 0.005

    while err > tau and iter < max_iter:  # While the error is greater than the convergence tolerance and the maximum number of iterations has not been reached,
        iter += 1
        Z_old = Z  # store the current value of Z
        err_old = err
        start_time = time.time()

        # Update U
        U = eigs(Y, k=k, which='LM')[1]  # compute the eigenvectors of Z corresponding to its largest eigenvalues
        UU = np.real(np.dot(U, U.T))

        # Update Z
        AAA = (beta + 1 + gamma) * np.eye(n) + alpha_XX
        BBB = As + beta * UU + alpha_XX + gamma * Z_old - lamb * I
        Y = linalg.solve(AAA, BBB)
        Y = Y.astype(float)

        D = Y * (Y > 0)

        # Update Z to T_eta(Z,D) by truncating Z:
        Z = shrink(Y, D, eta, n)

        err = np.linalg.norm(Z - Z_old, ord=None)
        err = err / np.linalg.norm(Z, ord=None)

        end_time = time.time()
        print(f"Time spent in iteration {iter}: {round(end_time - start_time, 4)} s")

        Z_sym = (np.abs(Z) + np.abs(Z.T)) / 2
        predict_labels = spectral_clustering(Z_sym, k=k, random_state=23)
        cm = clustering_metrics(gnd, predict_labels)
        ac, nm, f1 = cm.evaluationClusterModelFromLabel()
        acc.append(ac)
        nmi.append(nm)
        F1.append(f1)

        local_best_acc, local_best_nmi, local_best_f1, local_best_iter = update_local_best(
            err, err_old, acc, nmi, F1, iter,
            local_best_acc, local_best_nmi, local_best_f1, local_best_iter,
            max_iter, tau)
        if iter == max_iter or err < tau:
            print('local_best_acc: {:.4f}'.format(local_best_acc),
                  'local_best_nmi: {:.4f}'.format(local_best_nmi),
                  'local_best_f1: {:.4f}'.format(local_best_f1))
    return Z, acc, nmi, F1



def shrink(Z, D, eta, n):  # Define a helper function for truncating a matrix
    eta_1 = int(eta / 2)  # Compute half of eta rounded down to an integer
    D_upper = triu(D, k=1).tocoo()
    dd = D_upper.data
    dd.sort()  # Sort dd in ascending order
    dd = dd[::-1]  # Reverse dd to obtain it in descending order
    entry = ((eye(n) + (D >= dd[eta_1])) != 0)
    return Z * entry