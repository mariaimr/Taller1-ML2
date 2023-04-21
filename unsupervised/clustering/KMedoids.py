import numpy as np
from scipy.spatial.distance import cdist


class KMedoids:
    def __init__(self, n_clusters, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.medoids = None
        self.label = None

    def fit(self, X):
        # Select random n_clusters medoids from the dataset
        self.medoids = X[np.random.choice(range(len(X)), self.n_clusters, replace=False)]
        # Iterate until the assignment of points to clusters does not change
        for i in range(self.max_iters):
            # Calculate the distance between each point and the medoids
            D = self.dist_matrix(X, self.medoids)
            # Assign each point to the nearest medoid
            self.label = np.argmin(D, axis=1)
            # Calculate the total cost of the current assignment
            cost = np.sum(np.min(D, axis=1))
            # Select a new medoid for each cluster
            for j in range(self.n_clusters):
                # Obtain the points assigned to the j-th cluster
                cluster_points = X[self.label == j]
                # Calculate the distance between each point in the cluster and all other points in the cluster
                cluster_distances = self.dist_matrix(cluster_points, cluster_points)
                # Select the point with the lowest mean distance as the new medoid of the cluster
                new_medoid_idx = np.argmin(np.mean(cluster_distances, axis=1))
                self.medoids[j] = cluster_points[new_medoid_idx]

            # If the assignment of the points to the clusters did not change, terminate
            new_labels = np.argmin(self.dist_matrix(X, self.medoids), axis=1)
            if np.array_equal(self.label, new_labels):
                break

        return self.medoids, self.label

    def dist_matrix(self, X, medoids):
        return cdist(X, medoids, metric='euclidean')
