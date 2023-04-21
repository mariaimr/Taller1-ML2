import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn_extra.cluster import KMedoids

from clustering.Data import create_scattered_data_noisy_circles, create_scattered_data_noisy_moons, \
    create_scattered_data_blobs, create_scattered_data_without_structure, create_anisotropic_distributed_data, \
    create_scattered_data_blobs_varied_variances


def plot_scattered_data():
    picture_name = "ScatteredData.jpg"
    path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(path, "../resources")

    X_noisy_circles, y_noisy_circles = create_scattered_data_noisy_circles()
    X_noisy_moons, y_noisy_moons = create_scattered_data_noisy_moons()
    X_blobs, y_blobs = create_scattered_data_blobs()
    X_without_structure, y_without_structure = create_scattered_data_without_structure()
    X_anisotropic_distributed, y_anisotropic_distributed = create_anisotropic_distributed_data()
    X_blobs_varied_variances, y_blobs_varied_variances = create_scattered_data_blobs_varied_variances()

    _, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(11, 11))
    ax1.scatter(X_noisy_circles[:, 0], X_noisy_circles[:, 1], c=y_noisy_circles, cmap='viridis')
    ax1.set_title('Noisy Circles Data')
    ax2.scatter(X_noisy_moons[:, 0], X_noisy_moons[:, 1], c=y_noisy_moons, cmap='viridis')
    ax2.set_title('Noisy Moons Data')
    ax3.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, cmap='viridis')
    ax3.set_title('Blobs Data')
    ax4.scatter(X_without_structure[:, 0], X_without_structure[:, 1], c=y_without_structure, cmap='viridis')
    ax4.set_title('Data Without Structure')
    ax5.scatter(X_anisotropic_distributed[:, 0], X_anisotropic_distributed[:, 1], c=y_anisotropic_distributed, cmap='viridis')
    ax5.set_title('Data Anisotropic Distributed')
    ax6.scatter(X_blobs_varied_variances[:, 0], X_blobs_varied_variances[:, 1], c=y_blobs_varied_variances, cmap='viridis')
    ax6.set_title('Blobs Varied Variances')

    plt.tight_layout()
    plt.savefig(os.path.join(resources_path, picture_name))
    return os.path.join(resources_path, picture_name)


def plot_scattered_with_cluster_methods(X, y):
    picture_name = "ClustersScatteredData.jpg"
    path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(path, "../resources")

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=1, random_state=0)
    kmeans_labels = kmeans.fit_predict(X)

    # Perform k-medoids clustering
    kmedoids = KMedoids(n_clusters=3, random_state=0)
    kmedoids_labels = kmedoids.fit_predict(X)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=0.3, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X)

    # Perform Spectral Clustering
    spectral = SpectralClustering(n_clusters=3, random_state=0, affinity='nearest_neighbors')
    spectral_labels = spectral.fit_predict(X)

    # Visualize the clustering results
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', marker='o', edgecolors='k')
    plt.title("K-Means Clustering")

    plt.subplot(2, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=kmedoids_labels, cmap='viridis', marker='o', edgecolors='k')
    plt.title("K-Medoids Clustering")

    plt.subplot(2, 2, 3)
    plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis', marker='o', edgecolors='k')
    plt.title("DBSCAN Clustering")

    plt.subplot(2, 2, 4)
    plt.scatter(X[:, 0], X[:, 1], c=spectral_labels, cmap='viridis', marker='o', edgecolors='k')
    plt.title("Spectral Clustering")

    plt.tight_layout()
    plt.savefig(os.path.join(resources_path, picture_name))
    return os.path.join(resources_path, picture_name)

def plot_scattered_with_k_means(X):
    picture_name = "ClusteringKmeans.jpg"
    path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(path, "../resources")
    # Definir una lista de números de clusters para probar
    n_clusters_list = range(2, 6)

    # Crear figuras subplots para el diagrama de coeficiente de silueta y la gráfica de clusters
    fig, axes = plt.subplots(2, len(n_clusters_list), figsize=(15, 8))
    for i, n_clusters in enumerate(n_clusters_list):
        # Inicializar el modelo de K-Means con el número de clusters actual
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit_predict(X)

        labels = kmeans.labels_
        cluster_centroids = kmeans.cluster_centers_

        # Calcular el coeficiente de silueta para los clusters
        silhouette_avg = silhouette_score(X, labels)

        # Crear una gráfica de dispersión con los puntos de datos coloreados por las etiquetas de los clusters
        axes[0, i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        axes[0, i].scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], marker="o", c="white", alpha=1, s=200,
                           edgecolor="k")
        for idx, c in enumerate(cluster_centroids):
            axes[0, i].scatter(c[0], c[1], marker="$%d$" % idx, alpha=1, s=50, edgecolor="k")
        axes[0, i].set_title(f'Clusters = {n_clusters}\nSilhouette = {silhouette_avg:.2f}')

        # Crear un diagrama de barras del coeficiente de silueta para los clusters
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

    fig.suptitle('K-Means Clustering')
    plt.tight_layout()
    plt.savefig(os.path.join(resources_path, picture_name))
    return os.path.join(resources_path, picture_name)


def plot_scattered_with_k_medoids(X):
    picture_name = "ClusteringKmeans.jpg"
    path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(path, "../resources")
    # Definir una lista de números de clusters para probar
    n_clusters_list = range(2, 6)

    # Crear figuras subplots para el diagrama de coeficiente de silueta y la gráfica de clusters
    fig, axes = plt.subplots(2, len(n_clusters_list), figsize=(15, 8))
    for i, n_clusters in enumerate(n_clusters_list):
        # Inicializar el modelo de K-Means con el número de clusters actual

        # Perform k-medoids clustering
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
        kmedoids.fit_predict(X)

        labels = kmedoids.labels_
        cluster_medoids = kmedoids.cluster_centers_

        # Calcular el coeficiente de silueta para los clusters
        silhouette_avg = silhouette_score(X, labels)

        # Crear una gráfica de dispersión con los puntos de datos coloreados por las etiquetas de los clusters
        axes[0, i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        axes[0, i].scatter(cluster_medoids[:, 0], cluster_medoids[:, 1], marker="o", c="white", alpha=1, s=200,
                           edgecolor="k")
        for idx, c in enumerate(cluster_medoids):
            axes[0, i].scatter(c[0], c[1], marker="$%d$" % idx, alpha=1, s=50, edgecolor="k")
        axes[0, i].set_title(f'Clusters = {n_clusters}\nSilhouette = {silhouette_avg:.2f}')

        # Crear un diagrama de barras del coeficiente de silueta para los clusters
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

    fig.suptitle('K-Medoids Clustering')
    plt.tight_layout()
    plt.savefig(os.path.join(resources_path, picture_name))
    return os.path.join(resources_path, picture_name)


def plot_scattered_with_dbscan(X):
    picture_name = "ClusteringDBSCAN.jpg"
    path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(path, "../resources")
    fig, ax = plt.subplots(figsize=(9, 6))
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    dbscan.fit_predict(X)
    # Obtener los labels y la métrica de la silueta
    labels = dbscan.labels_
    silhouette = silhouette_score(X, labels)

    # Visualizar los resultados
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title('DBSCAN clustering with silhouette score: {}'.format(silhouette))

    plt.tight_layout()
    plt.savefig(os.path.join(resources_path, picture_name))
    return os.path.join(resources_path, picture_name)


def plot_scattered_with_spectral(X):
    picture_name = "ClusteringSpectral.jpg"
    path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(path, "../resources")
    # Definir una lista de números de clusters para probar
    n_clusters_list = range(2, 6)

    # Crear figuras subplots para el diagrama de coeficiente de silueta y la gráfica de clusters
    fig, axes = plt.subplots(1, len(n_clusters_list), figsize=(15, 8))
    for i, n_clusters in enumerate(n_clusters_list):

        # Perform Spectral Clustering
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=0, affinity='nearest_neighbors')
        spectral.fit_predict(X)

        labels = spectral.labels_
       # cluster_medoids = spectral.cluster_centers_

        # Calcular el coeficiente de silueta para los clusters
        silhouette_avg = silhouette_score(X, labels)
        sample_silhouette_values = silhouette_samples(X, labels)

        # Crear una gráfica de dispersión con los puntos de datos coloreados por las etiquetas de los clusters
        colors = plt.cm.Spectral(spectral.labels_.astype(float) / spectral.n_clusters)
        axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        #for idx, c in enumerate(cluster_medoids):
        #axes[0, i].scatter(c[0], c[1], marker="$%d$" % idx, alpha=1, s=50, edgecolor="k")
        axes[i].set_title(f'Clusters = {n_clusters}\nSilhouette = {silhouette_avg:.2f}')

        """"# Crear un diagrama de barras del coeficiente de silueta para los clusters
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
        axes[1, i].set_title(f'Clusters = {n_clusters}\nAvg Silhouette = {cluster_silhouette_avg:.2f}')"""

    fig.suptitle('Spectral Clustering')
    plt.tight_layout()
    plt.savefig(os.path.join(resources_path, picture_name))
    return os.path.join(resources_path, picture_name)