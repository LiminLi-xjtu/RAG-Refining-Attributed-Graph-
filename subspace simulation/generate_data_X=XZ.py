import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from datetime import datetime


def modify_off_diagonal(A, theta, block_sizes):
    """
      Modifies the block diagonal matrix A to change the partial 0's to 1's in non-diagonal blocks.

    Parameters:
        A (numpy.ndarray): the input block diagonal matrix (n x n) with elements 0 or 1.
        theta (float): The number of new 1's to the original 1's in A (e.g. 1/3).
        block_sizes (list[int]): size of each diagonal block, expressed as a list.

    Returns:
        numpy.ndarray: The modified matrix.
    """
    # Verify that the input matrix is a square matrix
    if A.shape[0] != A.shape[1]:
        raise ValueError("A Must be a square")
    if sum(block_sizes) != A.shape[0]:
        raise ValueError("The sum of block_sizes must equal the dimension of A")

    n = A.shape[0]

    # Initialize the index matrix for diagonal blocks
    diagonal_indices = np.zeros((n, n), dtype=bool)
    start_idx = 0

    # Mark the position of the diagonal blocks
    for size in block_sizes:
        end_idx = start_idx + size
        diagonal_indices[start_idx:end_idx, start_idx:end_idx] = True
        start_idx = end_idx

    off_diagonal_indices = ~diagonal_indices

    # Total number of 1's in the original A matrix
    original_ones_count = np.sum(A)

    # non-diagonal zeros need to be changed to ones
    target_flip_count = int(original_ones_count * theta)

    # Find a position in a non-diagonal block with a value of 0
    zero_indices = np.argwhere((A == 0) & off_diagonal_indices)

    # Randomly pick the position of 0 in a partially non-diagonal block
    np.random.shuffle(zero_indices)
    selected_indices = zero_indices[:target_flip_count]

    # Update the A matrix to change the selected position from 0 to 1
    for idx in selected_indices:
        A[idx[0], idx[1]] = 1

    return A





if __name__ == '__main__':
    n = [50, 80, 100]
    SIGMA = 0.1 * np.eye(5)
    mu1 = np.array([1, 1, 0, 0, 0])
    mu2 = np.array([0, 0, 1, 0, 0])
    mu3 = np.array([0, 0, 0, 1, 1])

    # Generate Gaussian distributed data
    r1 = np.random.multivariate_normal(mu1, SIGMA, n[0])
    r2 = np.random.multivariate_normal(mu2, SIGMA, n[1])
    r3 = np.random.multivariate_normal(mu3, SIGMA, n[2])
    X = np.vstack((r1, r2, r3))
    y = np.concatenate([np.ones(n[0]), 2 * np.ones(n[1]), 3 * np.ones(n[2])])

    # PLOT X0
    pca = PCA(n_components=2)
    Y = pca.fit_transform(X).T
    ids1 = np.where(y == 1)[0]
    ids2 = np.where(y == 2)[0]
    ids3 = np.where(y == 3)[0]
    plt.figure(figsize=(6, 6))
    plt.scatter(Y[0, ids1], Y[1, ids1], color='r', label='Class 1', s=50)
    plt.scatter(Y[0, ids2], Y[1, ids2], color='b', label='Class 2', s=50)
    plt.scatter(Y[0, ids3], Y[1, ids3], color='g', label='Class 3', s=50)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.legend(fontsize=20)
    plt.axis('equal')
    plt.show()

    # Construct the block diagonal matrix
    A = np.block([
        [np.ones((n[0], n[0])), np.zeros((n[0], n[1] + n[2]))],
        [np.zeros((n[1], n[0])), np.ones((n[1], n[1])), np.zeros((n[1], n[2]))],
        [np.zeros((n[2], n[0] + n[1])), np.ones((n[2], n[2]))]
    ])

    # Construct KNN graphs A0
    options_k = 5
    knn_graph = kneighbors_graph(X, n_neighbors=options_k, mode='connectivity')
    A0 = knn_graph.toarray() * A

    plt.figure(figsize=(6, 6))
    plt.imshow(A0, cmap='Greys', interpolation='nearest')
    plt.tick_params(axis='both', labelsize=22)
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().yaxis.set_ticks_position('left')
    # plt.colorbar()
    plt.show()


    # Iterative updating of data X
    X1 = X.copy()
    for i in range(2):
        X2 = (X1.T @ A0).T / options_k
        X1 = X2

    Y = PCA(n_components=2).fit_transform(X2)
    plt.figure(figsize=(6, 6))
    plt.scatter(Y[ids1, 0], Y[ids1, 1], color='r', label='Class 1', s=50)
    plt.scatter(Y[ids2, 0], Y[ids2, 1], color='b', label='Class 2', s=50)
    plt.scatter(Y[ids3, 0], Y[ids3, 1], color='g', label='Class 3', s=50)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.legend(fontsize=20)
    plt.show()

    # binary_knn_graph = kneighbors_graph(X2, n_neighbors=options_k, mode='connectivity')
    # AX = binary_knn_graph.toarray()


    A_1 = modify_off_diagonal(A0.copy(), theta=0.3, block_sizes=n)
    # PLOT A(0.3)
    plt.figure(figsize=(6, 6))
    plt.imshow(A_1, cmap='Greys', interpolation='nearest')
    plt.tick_params(axis='both', labelsize=22)
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().yaxis.set_ticks_position('left')
    # plt.colorbar()
    plt.show()

    A_2 = modify_off_diagonal(A0.copy(), theta=0.6, block_sizes=n)
    # PLOT A(0.6)
    plt.figure(figsize=(6, 6))
    plt.imshow(A_2, cmap='Greys', interpolation='nearest')
    plt.tick_params(axis='both', labelsize=22)
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().yaxis.set_ticks_position('left')
    # plt.colorbar()
    plt.show()

    A_3 = modify_off_diagonal(A0.copy(), theta=0.9, block_sizes=n)
    # PLOT A(0.9)
    plt.figure(figsize=(6, 6))
    plt.imshow(A_3, cmap='Greys', interpolation='nearest')
    plt.tick_params(axis='both', labelsize=22)
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().yaxis.set_ticks_position('left')
    # plt.colorbar()
    plt.show()


    # save data (We have saved at xzdata.npz)

    # current_time = datetime.now().strftime("%Y%m%d_%H%M")
    # filename = f"xzdata_{current_time}.npz"
    # np.savez(filename, features=X2, labels=y, A0=A0, A1=A_1, A2=A_2, A3=A_3)

    # A1: theta = 0.3, A2: theta = 0.6, A3: theta = 0.9