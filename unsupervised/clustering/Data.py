import numpy as np
from sklearn import datasets
from sklearn.datasets import make_blobs

n_samples = 500
random_state = 170


def create_toy_data():
    np.random.seed(123)
    return make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=4,
        cluster_std=1,
        center_box=(-10.0, 10.0),
        shuffle=True,
        random_state=1)


def create_scattered_data_noisy_circles():
    np.random.seed(123)
    return datasets.make_circles(
        n_samples=n_samples,
        factor=0.5,
        noise=0.05)


def create_scattered_data_noisy_moons():
    np.random.seed(123)
    return datasets.make_moons(
        n_samples=n_samples,
        noise=0.05)


def create_scattered_data_blobs():
    np.random.seed(123)
    return datasets.make_blobs(
        n_samples=n_samples,
        random_state=8)


def create_scattered_data_without_structure():
    np.random.seed(123)
    return np.random.rand(n_samples, 2), None


def create_anisotropic_distributed_data():
    np.random.seed(123)
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    return X_aniso, y


def create_scattered_data_blobs_varied_variances():
    np.random.seed(123)
    return make_blobs(
        n_samples=n_samples,
        cluster_std=[1.0, 2.5, 0.5],
        random_state=random_state
    )
