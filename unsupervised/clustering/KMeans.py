import random

import numpy as np


class KMeans:
    def __init__(self, n_clusters, max_iters=100, tol=1e-5):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.label = None
        random.seed(1111)

    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for i in range(self.max_iters):
            distances = self._calc_distances(X)

            # Assign each point to the nearest centroid
            self.label = np.argmin(distances, axis=0)

            # Calculate the mean of each group of points and update centroids
            new_centroids = np.zeros((self.n_clusters, X.shape[1]))
            for j in range(self.n_clusters):
                new_centroids[j] = X[self.label == j].mean(axis=0)

            # Check for convergence
            if np.sum(np.abs(new_centroids - self.centroids)) < self.tol:
                break

            self.centroids = new_centroids

        return self.centroids, self.label

    def _calc_distances(self, X):
        return np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))  # L2

    def pairwise_distances(self, X, Y=None):
        if Y is None:
            Y = X

        n_samples_X, n_features_X = X.shape
        n_samples_Y, n_features_Y = Y.shape

        assert n_features_X == n_features_Y, "Input matrices must have the same number of features"

        distances = np.zeros((n_samples_X, n_samples_Y))
        for i in range(n_samples_X):
            for j in range(n_samples_Y):
                distances[i, j] = np.sqrt(np.sum((X[i] - Y[j]) ** 2))  # L2

        return distances
