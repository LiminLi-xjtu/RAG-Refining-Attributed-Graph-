import numpy as np
from SLSA import SLSA
from utiles import spectral_clusteringA_mean, filter_features,spectral_clusteringX_mean, kmeans_clustering_mean, gaussian_kernel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = np.load('xzdata.npz')
    features = data['features']
    X = features
    A1 = gaussian_kernel(X, 0.1)
    A2 = data['A1']   # choose the A(theta)
    # If theta=0.3, A = data['A1']. If theta=0.6, A = data['A2']. If theta=0.9, A = data['A3'].
    sample_label = data['labels']
    gnd = sample_label
    k = 3

    best_spectral_i = -1
    best_spectral_acc = 0
    best_spectral_nmi = 0
    best_spectral_f1 = 0

    best_slsa_i = -1
    best_slsa_acc = 0
    best_slsa_nmi = 0
    best_slsa_f1 = 0

    for i in range(11):
        print('\ni: {}'.format(i))
        A = i * 0.1 * A1 + (1-i*0.1) * A2
        sp_acc_mean, acc_std, sp_nmi_mean, nmi_std, sp_f1_mean, f1_std = spectral_clusteringA_mean(A, gnd, n_iter=50, k=3)
        Z, U, acc, nmi, F1 = SLSA(A, gnd, eta=10000, n_class=k, theta=1, tau=0.0001, max_iter=50, Z=None)
        print('Spectral(A+X)_acc:{:.4f}'.format(sp_acc_mean))
        print('Spectral(A+X)_nmi:{:.4f}'.format(sp_nmi_mean))
        print('Spectral(A+X)_f1:{:.4f}\n'.format(sp_f1_mean))
        print('SLSA(A+X)_acc:{:.4f}'.format(acc[-1]))
        print('SLSA(A+X)_nmi:{:.4f}'.format(nmi[-1]))
        print('SLSA(A+X)_f1:{:.4f}'.format(F1[-1]))

        if sp_acc_mean > best_spectral_acc:
            best_spectral_i = i
            best_spectral_acc = sp_acc_mean
            best_spectral_nmi = sp_nmi_mean
            best_spectral_f1 = sp_f1_mean

        if acc[-1] > best_slsa_acc:
            best_slsa_i = i
            best_slsa_acc = acc[-1]
            best_slsa_nmi = nmi[-1]
            best_slsa_f1 = F1[-1]

    print("\nBest Results:")
    print(
        f"Spectral Best i: {best_spectral_i}, ACC: {best_spectral_acc:.4f}, NMI: {best_spectral_nmi:.4f}, F1: {best_spectral_f1:.4f}")
    print(f"SLSA Best i: {best_slsa_i}, ACC: {best_slsa_acc:.4f}, NMI: {best_slsa_nmi:.4f}, F1: {best_slsa_f1:.4f}")