import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score

from unsupervised.clustering.KMeans import KMeans
from unsupervised.clustering.KMedoids import KMedoids


def plot_data(X, y):
    picture_name = "data_toy.jpg"
    path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(path, "../../resources")
    fig, ax = plt.subplots(figsize=(9, 6))

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.tight_layout()
    plt.savefig(os.path.join(resources_path, picture_name))
    return os.path.join(resources_path, picture_name)


def calculate_distances(X):
    centroids, labels = KMeans(n_clusters=4).fit(X)
    distances = KMeans.pairwise_distances(centroids, centroids)
    n_rows, n_cols = distances.shape
    distances_cluster = []
    for i in range(n_rows):
        for j in range(i + 1, n_cols):
            distances_cluster.append(
                {"Distance between cluster #" + str(i + 1) + " and cluster #" + str(j + 1): round(distances[i, j], 3)})
    return distances_cluster


def plot_silhouette_coefficients_k_means(X):
    picture_name = "SilhouetteKMeans.jpg"
    path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(path, "../../resources")

    # Define a list of cluster numbers to be tested
    n_clusters_list = range(2, 6)

    # Create subplots for the silhouette coefficient plot and the cluster plot
    fig, axes = plt.subplots(2, len(n_clusters_list), figsize=(15, 8))
    for i, n_clusters in enumerate(n_clusters_list):
        # Initialize the K-Means model with the current number of clusters.
        cluster_centroids, labels = KMeans(n_clusters=n_clusters).fit(X)

        # Calculate the silhouette coefficient for the clusters
        silhouette_avg = silhouette_score(X, labels)

        # Create a scatter plot with the data points colored by the cluster labels.
        axes[0, i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        axes[0, i].scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], marker="o", c="white", alpha=1, s=200,
                           edgecolor="k")
        for idx, c in enumerate(cluster_centroids):
            axes[0, i].scatter(c[0], c[1], marker="$%d$" % idx, alpha=1, s=50, edgecolor="k")
        axes[0, i].set_title(f'Clusters = {n_clusters}\nSilhouette = {silhouette_avg:.2f}')

        # Create a bar chart of the silhouette coefficient for the clusters.
        cluster_silhouette_avg = silhouette_score(X[labels >= 0], labels[labels >= 0])
        sample_silhouette_values = silhouette_samples(X, labels)
        y_lower = 10
        for j in range(n_clusters):
            cluster_j_silhouette_values = sample_silhouette_values[labels == j]
            cluster_j_silhouette_values.sort()
            size_cluster_j = cluster_j_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j
            color = plt.cm.viridis(float(j) / n_clusters)
            axes[1, i].fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_j_silhouette_values, facecolor=color,
                                     edgecolor=color, alpha=0.7)
            axes[1, i].text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))
            y_lower = y_upper + 10
        axes[1, i].axvline(x=cluster_silhouette_avg, color="red", linestyle="--")
        axes[1, i].set_yticks([])
        axes[1, i].set_xlim([-0.1, 1])
        axes[1, i].set_ylim([0, len(X) + (n_clusters + 1) * 10])
        axes[1, i].set_title(f'Clusters = {n_clusters}\nAvg Silhouette = {cluster_silhouette_avg:.2f}')

    plt.tight_layout()
    plt.savefig(os.path.join(resources_path, picture_name))
    return os.path.join(resources_path, picture_name)


def plot_silhouette_coefficients_k_medoids(X):
    picture_name = "SilhouetteKMedoids.jpg"
    path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(path, "../../resources")

    #  Define a list of cluster numbers to be tested
    n_clusters_list = range(2, 6)

    # Create subplots for the silhouette coefficient plot and the cluster plot
    fig, axes = plt.subplots(2, len(n_clusters_list), figsize=(15, 8))
    for i, n_clusters in enumerate(n_clusters_list):
        # Initialize the K-Means model with the current number of clusters.
        cluster_medoids, labels = KMedoids(n_clusters=n_clusters).fit(X)

        # Calculate the silhouette coefficient for the clusters
        silhouette_avg = silhouette_score(X, labels)

        # Create a scatter plot with the data points colored by the cluster labels.
        axes[0, i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        axes[0, i].scatter(cluster_medoids[:, 0], cluster_medoids[:, 1], marker="o", c="white", alpha=1, s=200,
                           edgecolor="k")
        for idx, c in enumerate(cluster_medoids):
            axes[0, i].scatter(c[0], c[1], marker="$%d$" % idx, alpha=1, s=50, edgecolor="k")
        axes[0, i].set_title(f'Clusters = {n_clusters}\nSilhouette = {silhouette_avg:.2f}')

        #  Create a bar chart of the silhouette coefficient for the clusters.
        cluster_silhouette_avg = silhouette_score(X[labels >= 0], labels[labels >= 0])
        sample_silhouette_values = silhouette_samples(X, labels)
        y_lower = 10
        for j in range(n_clusters):
            cluster_j_silhouette_values = sample_silhouette_values[labels == j]
            cluster_j_silhouette_values.sort()
            size_cluster_j = cluster_j_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j
            color = plt.cm.viridis(float(j) / n_clusters)
            axes[1, i].fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_j_silhouette_values, facecolor=color,
                                     edgecolor=color, alpha=0.7)
            axes[1, i].text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))
            y_lower = y_upper + 10
        axes[1, i].axvline(x=cluster_silhouette_avg, color="red", linestyle="--")
        axes[1, i].set_yticks([])
        axes[1, i].set_xlim([-0.1, 1])
        axes[1, i].set_ylim([0, len(X) + (n_clusters + 1) * 10])
        axes[1, i].set_title(f'Clusters = {n_clusters}\nAvg Silhouette = {cluster_silhouette_avg:.2f}')

    plt.tight_layout()
    plt.savefig(os.path.join(resources_path, picture_name))
    return os.path.join(resources_path, picture_name)
