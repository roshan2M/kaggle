import pandas as pd
import numpy as np

class PCA(object):
    def __init__(self, X):
        """Apply Principal Component Analysis (PCA)

        Uses eigenvalue decomposition of the correlation matrix between features
        - eigenvectors represent the directions of the principal components
        - eigenvalues represent variances of those principal components
        """
        pass

    def normalize(self):
        return np.divide(X - np.mean(X), np.std(X))

    def eigen_pairs(self, X):
        eigenVal, eigenVec = np.linalg.eigh(X.cov())
        return [(np.abs(eigenVal[i]), eigenVec[:,i]) for i in range(len(eigenVal))].sort(key=lambda e : e[0], reverse=True)

    def principal_components(self, pairs, X, n):
        pcaT = [np.dot(X, pairs[i][1].reshape(-1, 1)).reshape(1, -1).T for i in range(n)]
        return np.array(zip(*pcaT)).reshape(-1, n)
