import os

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score

from unsupervised.KMeans import KMeans
from unsupervised.KMedoids import KMedoids


def plot_data(X, y):
    picture_name = "data_toy.jpg"
    path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(path, "../resources")
    fig, ax = plt.subplots(figsize=(9, 6))
    centroids, labels = KMeans(n_clusters=4).fit(X)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='D')
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


def plot_silhouette_coefficients_k_means1(X):
    picture_name = "SilhouetteKMeans.jpg"
    path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(path, "../resources")
    range_n_clusters = range(2, 6)

    # Create a subplot with 1 row and 2 columns
    fig, axes = plt.subplots()
    fig.set_size_inches(18, 7)

    for id in range(1, 2):
        for n_clusters in range_n_clusters:

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            axes.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            axes.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusters with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            cluster_centroids, cluster_labels = KMeans(n_clusters=n_clusters).fit(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is :",
                silhouette_avg,
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                axes.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                axes.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            axes.set_title("The silhouette plot for the various clusters.")
            axes.set_xlabel("The silhouette coefficient values")
            axes.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            axes.axvline(x=silhouette_avg, color="red", linestyle="--")

            axes.set_yticks([])  # Clear the yaxis labels / ticks
            axes.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            axes.scatter(
                X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
            )

            # Labeling the clusters
            centers = cluster_centroids
            # Draw white circles at cluster centers
            axes.scatter(
                centers[:, 0],
                centers[:, 1],
                marker="o",
                c="white",
                alpha=1,
                s=200,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                axes.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

            axes.set_title("The visualization of the clustered data.")
            axes.set_xlabel("Feature space for the 1st feature")
            axes.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(
                "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
                % n_clusters,
                fontsize=14,
                fontweight="bold",
            )

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(resources_path, picture_name))
    return os.path.join(resources_path, picture_name)


def plot_silhouette_coefficients_k_means(X, y):
    global color, size_cluster_j
    picture_name = "SilhouetteKMeans.jpg"
    path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(path, "../resources")
    k_values = range(2, 6)

    fig, axes = plt.subplots(4, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i, k in enumerate(k_values):
        centroids, labels = KMeans(n_clusters=k).fit(X)

        # Calculate the mean silhouette coefficient off all samples (labels)
        silhouette_avg = silhouette_score(X, labels)
        # Calculate the silhouette coefficients for each sample
        silhouette_vals = silhouette_samples(X, labels)
        unique_labels = np.unique(labels)
        silhouette_vals = [silhouette_vals[labels == j] for j in unique_labels]

        # Create a horizontal bar chart for silhouette coefficient
        y_lower = 12

        for j in range(k):
            # Add the silhouette coefficient for each sample in the cluster
            cluster_silhouette_values = silhouette_vals[j]
            cluster_silhouette_values.sort()
            size_cluster_j = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j
            color = plt.cm.get_cmap("Spectral")(float(j) / k)
            if i == 0:
                axes[i].text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))
                axes[i].barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
                             color=color)
            else:
                if i % 2 == 0:
                    axes[i + 2].text(-0.05, y_lower + 0.5 * size_cluster_j, str(j + 1))
                    axes[i + 2].barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
                                     color=color)
                else:
                    axes[i * 2].text(-0.05, y_lower + 0.5 * size_cluster_j, str(j + 1))
                    axes[i * 2].barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
                                     color=color)
            y_lower = y_upper + 10

        if i == 0:
            axes[i].set_xlabel('Silhouette Coefficient')
            axes[i].set_ylabel('Samples')
            axes[i].set_title(f'k = {k}, Average Silhouette Coefficient = {silhouette_avg:.2f}')
            axes[i].axvline(x=silhouette_avg, color="black",
                            linestyle="--")  # Vertical line for the average coefficient

            axes[i + 1].scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
            axes[i + 1].scatter(centroids[:, 0], centroids[:, 1],marker="o", c="white", alpha=1, s=200,edgecolor="k")
            for idx, c in enumerate(centroids):
                axes[i + 1].scatter(c[0], c[1], marker="$%d$" % idx, alpha=1, s=50, edgecolor="k")
            axes[i + 1].set_title("The visualization of the clustered data.")
        else:
            if i % 2 == 0:
                axes[i + 2].set_xlabel('Silhouette Coefficient')
                axes[i + 2].set_ylabel('Samples')
                axes[i + 2].set_title(f'k = {k}, Average Silhouette Coefficient = {silhouette_avg:.2f}')
                axes[i + 2].axvline(x=silhouette_avg, color="black", linestyle="--")
                axes[i + 3].scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
                axes[i + 3].scatter(centroids[:, 0], centroids[:, 1], marker="o", c="white", alpha=1, s=200,
                                    edgecolor="k")
                for idx, c in enumerate(centroids):
                    axes[i + 3].scatter(c[0], c[1], marker="$%d$" % idx, alpha=1, s=50, edgecolor="k")
                axes[i + 3].set_title("The visualization of the clustered data.")
                axes[i + 3].text(-0.05, y_lower + 0.5 * len(cluster_silhouette_values), str(i+1))
            else:
                axes[i * 2].set_xlabel('Silhouette Coefficient')
                axes[i * 2].set_ylabel('Samples')
                axes[i * 2].set_title(f'k = {k}, Average Silhouette Coefficient = {silhouette_avg:.2f}')
                axes[i * 2].axvline(x=silhouette_avg, color="black", linestyle="--")
                axes[i * 2 + 1].scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
                axes[i * 2 + 1].scatter(centroids[:, 0], centroids[:, 1], marker="o", c="white", alpha=1, s=200,
                                    edgecolor="k")
                for idx, c in enumerate(centroids):
                    axes[i * 2 + 1].scatter(c[0], c[1], marker="$%d$" % idx, alpha=1, s=50, edgecolor="k")
                axes[i * 2 + 1].set_title("The visualization of the clustered data.")
                axes[i * 2 + 1].text(-0.05, y_lower + 0.5 * len(cluster_silhouette_values), str(i + 1))

    plt.subplots_adjust(hspace=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(resources_path, picture_name))
    return os.path.join(resources_path, picture_name)


def plot_silhouette_coefficients_k_medoids(X):
    picture_name = "SilhouetteKMedoids.jpg"
    path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(path, "../resources")
    np.random.seed(1)
    k_values = range(2, 6)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, k in enumerate(k_values):
        medoids, labels = KMedoids(n_clusters=k).fit(X)

        # Calculate the mean silhouette coefficient off all samples (labels)
        silhouette_avg = silhouette_score(X, labels)
        # Calculate the silhouette coefficients for each sample
        silhouette_vals = silhouette_samples(X, labels)
        unique_labels = np.unique(labels)
        silhouette_vals = [silhouette_vals[labels == j] for j in unique_labels]

        # Create a horizontal bar chart for silhouette coefficient
        y_lower = 10

        for j in range(k):
            # Add the silhouette coefficient for each sample in the cluster
            cluster_silhouette_values = silhouette_vals[j]
            cluster_silhouette_values.sort()
            size_cluster_j = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j
            color = plt.cm.get_cmap("Spectral")(float(j) / k)
            axes[i].barh(range(y_lower, y_upper), cluster_silhouette_values, height=1.0,
                         color=color)
            y_lower = y_upper + 10

        axes[i].set_xlabel('Silhouette Coefficient')
        axes[i].set_ylabel('Samples')
        axes[i].set_title(f'k = {k}, Average Silhouette Coefficient = {silhouette_avg:.2f}')
        axes[i].axvline(x=silhouette_avg, color="black", linestyle="--")  # Vertical line for the average coefficient

    plt.tight_layout()
    plt.savefig(os.path.join(resources_path, picture_name))
    return os.path.join(resources_path, picture_name)
