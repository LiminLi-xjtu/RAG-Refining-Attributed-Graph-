import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from model import RAG

def has_nonpositive_element(X):
    return (X < 0).any()

def replace_negative_with_zero(matrix):
    matrix[matrix < 0] = 0
    return matrix



if __name__ == '__main__':
    # Load data
    dataset = 'cora'
    stage = 2
    gamma = 0.001

    if dataset == 'cora':  # A=Z1,X=X
        features_struct = np.load('data/cora_sorted.npz')
        X = features_struct['features']
        gnd = features_struct['labels']
        if stage == 2:
            A = np.load('data/coraZ1.npz')
            A = A['adjacency']
            a = has_nonpositive_element(A)
            if a == True:
                A = (np.abs(A) + np.abs(A.T)) / 2
            k = 7
            gnd = gnd - 1
            Z, acc, nmi, F1 = RAG(A, X, gnd, k, eta=550000, n_class=k, alpha=100, beta=1, lamb=0, t=4,
                                  tau=0.005, max_iter=50)
            # maxiteration=43,eta=550000, n_class=k, theta=1, alpha=100, beta=1,lamb=0,t=4
        if stage == 3:
            A = np.load('data/coraZ2.npz')
            A = A['adjacency']
            a = has_nonpositive_element(A)
            if a == True:
                A = (np.abs(A) + np.abs(A.T)) / 2
            k = 7
            gnd = gnd - 1
            Z, acc, nmi, F1 = RAG(A, X, gnd, k, eta=650000, n_class=k, alpha=100, beta=1, lamb=0, t=4,
                                  tau=0.005, max_iter=50)
            # maxiteration=2,eta=600000, n_class=k, theta=1, alpha=100, beta=1,lamb=0,t=4



    if dataset == 'citeseer':
        features_struct = np.load('data/citeseer_sorted.npz')
        X = features_struct['features']
        gnd = features_struct['labels']
        if stage == 2:
            A = np.load('data/citeseerZ1.npz')
            A = A['adjacency']
            a = has_nonpositive_element(A)
            if a == True:
                # A = (np.abs(A) + np.abs(A.T)) / 2
                A = replace_negative_with_zero(A)
            k = 6
            gnd = gnd - 1
            Z, acc, nmi, F1 = RAG(A, X, gnd, k, eta=600000, n_class=k, alpha=4, beta=1, lamb=0.4, t=4, tau=0.005,
                                  max_iter=50)
            # max_iter=41,eta=600000, alpha=4, beta=1, lamb=0.4,t=4
        if stage == 3:
            A = np.load('data/citeseerZ2.npz')
            A = A['adjacency']
            a = has_nonpositive_element(A)
            if a == True:
                # A = (np.abs(A) + np.abs(A.T)) / 2
                A = replace_negative_with_zero(A)
            k = 6
            gnd = gnd - 1
            Z, acc, nmi, F1 = RAG(A, X, gnd, k, eta=600000, n_class=k, alpha=4, beta=1, lamb=0.4, t=4, tau=0.005,
                                  max_iter=50)
        # max_iter=28,eta=600000, alpha=4, beta=1, lamb=0.4,t=4



    if (dataset == 'acm'):
        data = sio.loadmat('data/{}.mat'.format(dataset))
        k = 3
        X = data['fea']
        if stage == 2:
            AA = np.load('data/acmZ1.npz')
            A = AA['adjacency']
            a = has_nonpositive_element(A)
            if a == True:
                A = replace_negative_with_zero(A)
            gnd = data['gnd'][0, :]
            Z, acc, nmi, F1 = RAG(A, X, gnd, k, eta=2800000, n_class=k, alpha=200, beta=1, lamb=0, t=1,
                                  tau=0.05, max_iter=10)
            # maxiter=3 , eta=2800000, alpha=200, beta=1, lamb=0,t=1
        if stage == 3:
            AA = np.load('data/acmZ2.npz')
            A = AA['adjacency']
            a = has_nonpositive_element(A)
            if a == True:
                A = replace_negative_with_zero(A)
            gnd = data['gnd'][0, :]
            Z, acc, nmi, F1 = RAG(A, X, gnd, k, eta=2800000, n_class=k, alpha=200, beta=1, lamb=0, t=1,
                                  tau=0.05, max_iter=100)
            # maxiter=91 , eta=2800000, alpha=200, beta=1, lamb=0,t=1



    if (dataset == 'dblp'):
        data = sio.loadmat('data/{}.mat'.format(dataset))
        k = 4
        X = data['fea']
        if stage == 2:
            AA = np.load('data/dblpZ1.npz')
            A = AA['adjacency']
            a = has_nonpositive_element(A)
            if a == True:
                A = replace_negative_with_zero(A)
            gnd = data['gnd'][0, :]
            Z, acc, nmi, F1 = RAG(A, X, gnd, k, eta=700000, n_class=k, alpha=5, beta=0.5, lamb=0.1, t=1,
                                  tau=0.005, max_iter=8)
            # maxiter=8 , eta=800000, alpha=5, beta=0.5, lamb=0.1,t=1



