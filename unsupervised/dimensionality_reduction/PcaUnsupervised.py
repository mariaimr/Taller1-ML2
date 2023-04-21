import logging

import numpy as np
from numpy.linalg import svd

from unsupervised.dimensionality_reduction.BaseEstimator import BaseEstimator


class PcaUnsupervised(BaseEstimator):
    y_required = False

    def __init__(self, n_components, solver="svd"):
        """Principal component analysis (PCA) implementation.
        Transforms a dataset of possibly correlated values into n linearly
        uncorrelated components. The components are ordered such that the first
        has the largest possible variance and each following component as the
        largest possible variance given the previous components. This causes
        the early components to contain most of the variability in the dataset.
        Parameters
        ----------
        n_components : int
        solver : str, default 'svd'
            {'svd', 'eigen'}
        """
        self.solver = solver
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=0)
        self._decompose(X)

    def _decompose(self, X):
        # Mean centering
        X_centered = X - self.mean

        if self.solver == "svd":
            _, s, Vh = svd(X_centered, full_matrices=True)
        elif self.solver == "eigen":
            s, Vh = np.linalg.eig(np.cov(X_centered, rowvar=False))

        s_squared = s ** 2
        variance_ratio = s_squared / s_squared.sum()
        logging.info("Explained variance ratio: %s" % (variance_ratio[0: self.n_components]))
        self.components = Vh[:self.n_components].T

    def fit_transform(self, X):
        self.fit(X)
        X_centered = X - self.mean
        X_pca = X_centered @ self.components
        img_reconstructed = (X_pca @ self.components.T) + self.mean
        return img_reconstructed
