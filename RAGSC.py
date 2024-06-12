import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from model import RAG



if __name__ == '__main__':
    # Load data
    dataset = 'citeseer'
    gamma = 0.001


    if dataset == 'cora':
        data = sio.loadmat('data/{}.mat'.format(dataset))
        X = data['fea']
        A = data['W']
        gnd = data['gnd']
        gnd = gnd.T
        gnd = gnd - 1
        gnd = gnd[0, :]
        k = 7
        Z, acc, nmi, F1 = RAG(A, X, gnd, k, eta=300000, n_class=k, alpha=120, beta=1, lamb=0.5, t=16, tau=0.005,
                              max_iter=30)

        plt.figure(figsize=(8, 6))
        plt.xlabel('Iteration (Cora)', fontsize=24)
        plt.ylabel('Accuracy & NMI & F1', fontsize=24)
        # plt.ylim((0.5, 1))
        plt.plot(acc, color='red', label='Accuracy', linewidth=4)
        plt.plot(nmi, color='green', label='NMI', linewidth=4)
        plt.plot(F1, color='blue', label='F1', linewidth=4)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(fontsize=22)
        plt.tight_layout()
        plt.show()



    if dataset == 'citeseer':
        data = sio.loadmat('data/{}.mat'.format(dataset))
        X = data['fea']
        A = data['W']
        gnd = data['gnd']
        gnd = gnd.T
        gnd = gnd - 1
        gnd = gnd[0, :]
        k = 6
        Z, acc, nmi, F1 = RAG(A, X, gnd, k, eta=600000, n_class=k, alpha=41, beta=0.01, lamb=0, t=8,
                              tau=0.005, max_iter=30)


    if dataset == 'wiki':
        data = sio.loadmat('data/{}.mat'.format(dataset))
        k = 17
        X = data['fea']
        A = data['W']
        gnd = data['gnd']
        gnd = gnd.T
        gnd = gnd - 1
        gnd = gnd[0, :]
        Z, acc, nmi, F1 = RAG(A, X, gnd, k, eta=600000, n_class=k, alpha=150, beta=1, lamb=3, t=4, tau=0.005,
                              max_iter=100)
        # max_iter=63,eta=600000, theta=1, alpha=150, beta=1, lamb=3,t=4

    if (dataset == 'acm'):
        data = sio.loadmat('data/{}.mat'.format(dataset))
        k = 3
        X = data['fea']
        A = data['W']
        gnd = data['gnd'][0, :]
        Z, acc, nmi, F1 = RAG(A, X, gnd, k, eta=2800000, n_class=k, alpha=200, beta=1, lamb=0, t=5, tau=0.005,
                              max_iter=30)
        # maxiter=21 , eta=2800000, theta=1, alpha=200, beta=1, lamb=0,t=5

    if (dataset == 'dblp'):
        data = sio.loadmat('data/{}.mat'.format(dataset))
        k = 4
        X = data['fea']
        A = data['W']
        gnd = data['gnd'][0, :]
        Z, acc, nmi, F1 = RAG(A, X, gnd, k, eta=1000000, n_class=k, alpha=50, beta=0.5, lamb=0.1, t=1,
                              tau=0.005, max_iter=10)
        # maxiter=5, eta=1000000, theta=1, alpha=50, beta=0.5, lamb=0.1, t=1





# cora-sorted version, conveniently used to visualize matrix block diagonal effects.
    sort = False
    if dataset == 'cora' and sort == True:
        features_struct = np.load('data/cora_sorted.npz')
        X = features_struct['features']
        gnd = features_struct['labels']
        A = features_struct['adjacency']
        k = 7
        gnd = gnd - 1
        Z, acc, nmi, F1 = RAG(A, X, gnd, k, eta=300000, n_class=k, alpha=115, beta=1, lamb=0.5, t=16, tau=0.005,
                              max_iter=30)  # max_iter=7,eta=300000, theta=1, alpha=115, beta=1, lamb=0.5, t=16


    if dataset == 'citeseer'and sort == True:
        features_struct = np.load('data/citeseer_sorted.npz')
        X = features_struct['features']
        gnd = features_struct['labels']
        A = features_struct['adjacency']
        k = 6
        gnd = gnd - 1
        Z, acc, nmi, F1 = RAG(A, X, gnd, k, eta=600000, n_class=k, alpha=42, beta=0.0125, lamb=0, t=8,
                              tau=0.005, max_iter=50)
        # max_iter=1,eta=600000,theta=1, alpha=80, beta=100, lamb=1,k=8










