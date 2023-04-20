#  Apply k-means, k-medoids, DBSCAN and Spectral Clustering from Scikit-Learn
import os

from matplotlib import pyplot as plt

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

    #fig, ax = plt.subplots(figsize=(9, 6))
    #centroids, labels = KMeans(n_clusters=4).fit(X)

    _, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(11, 11))
    ax1.scatter(X_noisy_circles[:, 0], X_noisy_circles[:, 1], c=y_noisy_circles, cmap='summer')
    ax1.set_title('Noisy Circles Data')
    ax2.scatter(X_noisy_moons[:, 0], X_noisy_moons[:, 1], c=y_noisy_moons, cmap='summer')
    ax2.set_title('Noisy Moons Data')
    ax3.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, cmap='summer')
    ax3.set_title('Blobs Data')
    ax4.scatter(X_without_structure[:, 0], X_without_structure[:, 1], c=y_without_structure, cmap='summer')
    ax4.set_title('Data Without Structure')
    ax5.scatter(X_anisotropic_distributed[:, 0], X_anisotropic_distributed[:, 1], c=y_anisotropic_distributed, cmap='summer')
    ax5.set_title('Data Anisotropic Distributed')
    ax6.scatter(X_blobs_varied_variances[:, 0], X_blobs_varied_variances[:, 1], c=y_blobs_varied_variances, cmap='summer')
    ax6.set_title('Blobs Varied Variances')


    #plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='D')
    plt.tight_layout()
    plt.savefig(os.path.join(resources_path, picture_name))
    return os.path.join(resources_path, picture_name)


