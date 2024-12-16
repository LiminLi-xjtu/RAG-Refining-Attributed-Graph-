import numpy as np
from scipy import linalg
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from utiles import clustering_metrics, update_local_best, spectral_clusteringA_mean, filter_features, spectral_clusteringX_mean, kmeans_clustering_mean
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def RAG(A, X, eta, n_class, alpha, beta, lamb, tau, t, max_iter=10):
    """
    Refined Attributed Graph(RAG)
    """

    n = A.shape[0]
    norm = np.linalg.norm(A, ord=None)
    scale = np.sqrt(n_class) / norm
    As = (A + A.T) * (scale / 2)  # Scale and symmetrize the input matrix A

    # filter X
    I = np.eye(n)
    X = filter_features(A, X, t)

    Xnorm = np.linalg.norm(X, ord=None)
    scaleX = np.sqrt(n) / Xnorm
    X = X * scaleX
    alpha_XX = alpha * np.dot(X, X.T)

    gamma = 10
    Z = 0
    Y = As

    acc = []
    nmi = []
    F1 = []
    local_best_acc = 0.0
    local_best_nmi = 0.0
    local_best_f1 = 0.0
    local_best_iter = 0

    err = 1  # Initialize the error to be greater than the convergence tolerance
    iter = 0  # Initialize the iteration counter

    ac, _, nm, _, f1, _= spectral_clusteringA_mean(As, gnd, n_iter=50, k=3)
    # print('iter: {}'.format(iter), 'acc: {:.4f}'.format(ac), 'nmi: {:.4f}'.format(nm))

    while err > tau and iter < max_iter:  # While the error is greater than the convergence tolerance and the maximum number of iterations has not been reached,
        iter += 1
        Z_old = Z
        err_old = err

        # Update U
        U = eigs(Y, k=n_class, which='LM')[1]  # compute the eigenvectors of Z corresponding to its largest eigenvalues
        UU = np.real(np.dot(U, U.T))


        # Update Z
        AAA = (beta + 1 + gamma) * np.eye(n) + alpha_XX
        BBB = As + beta * UU + alpha_XX + gamma * Z_old - lamb * I

        Y = linalg.solve(AAA, BBB)
        Y = Y.astype(float)

        D = Y * (Y > 0)

        # Update Z to T_eta(Z,D) by truncating Z:
        Z = shrink(Y, D, eta, n)  # apply a truncation operation to Z using the helper function shrink
        Z = (np.abs(Z) + np.abs(Z.T)) / 2
        err = np.linalg.norm(Z - Z_old, ord=None)

        Z_sym = (np.abs(Z) + np.abs(Z.T)) / 2
        ac, _, nm, _, f1, _ = spectral_clusteringA_mean(Z_sym, gnd, n_iter=50, k=3)
        acc.append(ac)
        nmi.append(nm)
        F1.append(f1)

        local_best_acc, local_best_nmi, local_best_f1, local_best_iter = update_local_best(
            err, err_old, acc, nmi, F1, iter,
            local_best_acc, local_best_nmi, local_best_f1, local_best_iter,
            max_iter, tau)
        # if iter == max_iter or err < tau:
        #     print('local_best_acc: {:.4f}'.format(local_best_acc),
        #           'local_best_nmi: {:.4f}'.format(local_best_nmi),
        #           'local_best_f1: {:.4f}'.format(local_best_f1))

    return Z, U, acc, nmi, F1


def shrink(Z, D, eta, n):  # Define a helper function for truncating a matrix
    eta_1 = int(eta / 2)
    dd = np.triu(D, 1).flatten()  # Extract the upper triangular part of D above its main diagonal and flatten it into a one-dimensional array
    dd.sort()  # Sort dd in ascending order
    dd = dd[::-1]
    return Z * ((np.eye(n) + (D >= dd[eta_1])) != 0)



if __name__ == '__main__':
    data = np.load('xzdata.npz')
    features = data['features']
    sample_label = data['labels']
    X = features
    A = data['A1']    # choose the A(theta)
    # If theta=0.3, A = data['A1'].
    # If theta=0.6, A = data['A2'].
    # If theta=0.9, A = data['A3'].

    gnd = sample_label
    k = 3

    Z, U, rag_acc, rag_nmi, rag_F1 = RAG(A, X, eta=10000, n_class=k, alpha=1, beta=0.1, lamb=1, tau=0.0001, t=1, max_iter=30)

    print('RAG_acc: {:.4f}'.format(rag_acc[-1]),
         'RAG_nmi: {:.4f}'.format(rag_nmi[-1]),
          'RAG_F1: {:.4f}'.format(rag_F1[-1]))
