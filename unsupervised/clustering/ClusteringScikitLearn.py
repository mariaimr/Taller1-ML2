import os
import warnings
from itertools import islice, cycle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids

from unsupervised.clustering.Data import create_scattered_data_noisy_circles, create_scattered_data_noisy_moons, \
    create_scattered_data_blobs, create_scattered_data_without_structure, create_anisotropic_distributed_data, \
    create_scattered_data_blobs_varied_variances


def plot_scattered_data():
    picture_name = "ScatteredData.jpg"
    path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(path, "../../resources")

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
    ax5.scatter(X_anisotropic_distributed[:, 0], X_anisotropic_distributed[:, 1], c=y_anisotropic_distributed,
                cmap='viridis')
    ax5.set_title('Data Anisotropic Distributed')
    ax6.scatter(X_blobs_varied_variances[:, 0], X_blobs_varied_variances[:, 1], c=y_blobs_varied_variances,
                cmap='viridis')
    ax6.set_title('Blobs Varied Variances')

    plt.tight_layout()
    plt.savefig(os.path.join(resources_path, picture_name))
    return os.path.join(resources_path, picture_name)


def plot_scattered_with_cluster_methods():
    picture_name = "ClusteringPlot.jpg"
    path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(path, "../../resources")
    plt.figure(figsize=(20, 13))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
    )

    plot_num = 1

    default_base = {
        "quantile": 0.3,
        "eps": 0.3,
        "damping": 0.9,
        "preference": -200,
        "n_neighbors": 3,
        "n_clusters": 3,
        "min_samples": 7,
        "xi": 0.05,
        "min_cluster_size": 0.1,
    }

    datasets = [
        (create_scattered_data_noisy_circles(), {
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.08,
        }),
        (create_scattered_data_noisy_moons(),  {
            "damping": 0.75,
            "preference": -220,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.1,
        }),
        (create_scattered_data_blobs(), {
            "eps": 0.18,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.01,
            "min_cluster_size": 0.2,
        }),
        (create_anisotropic_distributed_data(), {
            "eps": 0.15,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2,
        }),
        (create_scattered_data_blobs_varied_variances(), {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),
        (create_scattered_data_without_structure(), {})
    ]

    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        X, y = dataset

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # ============
        # Create cluster objects
        # ============
        kmeans = KMeans(n_clusters=params["n_clusters"])
        kmedoids = KMedoids(n_clusters=params["n_clusters"])
        spectral = SpectralClustering(n_clusters=params["n_clusters"],
                                      eigen_solver='arpack',
                                      affinity="nearest_neighbors")
        dbscan = DBSCAN(eps=params["eps"])

        clustering_algorithms = (
            ("KMeans", kmeans),
            ("KMedoids", kmedoids),
            ("Spectral Clustering", spectral),
            ("DBSCAN", dbscan),
        )

        for name, algorithm in clustering_algorithms:
            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the "
                            + "connectivity matrix is [0-9]{1,2}"
                            + " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding"
                            + " may not work as expected.",
                    category=UserWarning,
                )
            y_pred = algorithm.fit_predict(X)

            plt.subplot(len(datasets), len(clustering_algorithms), plot_num)

            colors = np.array(
                list(
                    islice(
                        cycle(
                            ["#40E0D0", "#EE82EE", "#800080", "#3CB371", "#E9967A", "#DDA0DD", "#800080", "#EE82EE",
                             "#9ACD32"]), int(max(y_pred) + 1),
                    )
                )
            )
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

            # Calculate the silhouette coefficient for the clusters
            silhouette_avg = silhouette_score(X, y_pred)
            plt.title(f'{name}\nSilhouette Average = {silhouette_avg:.2f}')
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plot_num += 1
    plt.tight_layout()
    plt.savefig(os.path.join(resources_path, picture_name))
    return os.path.join(resources_path, picture_name)
