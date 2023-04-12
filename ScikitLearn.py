import os
import time

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE


def load_nmist_dataset():
    mnist = fetch_openml('mnist_784')
    X, y = mnist["data"], mnist["target"]

    X_numbers_0_8 = X[(y == '0') | (y == '8')]
    y_numbers_0_8 = y[(y == '0') | (y == '8')]
    X_numbers_0_8 = X_numbers_0_8.to_numpy()
    X_numbers_0_8 = X_numbers_0_8.reshape(X_numbers_0_8.shape[0], -1)

    split = int(X_numbers_0_8.shape[0] * 0.8)
    X_train, y_train = X_numbers_0_8[:split], y_numbers_0_8[:split]
    X_test, y_test = X_numbers_0_8[split:], y_numbers_0_8[split:]
    return X_train, y_train, X_test, y_test


def train_logistic_regression_model(X_train, y_train, X_test, y_test):
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(X_train, y_train)
    return logistic_regression_model.score(X_test, y_test)


def plot_dimension_reduction_scikit_learn():
    picture_name = "DimensionReductionScikitLearn.jpg"
    path = os.path.dirname(os.path.abspath(__file__))
    resources_path = os.path.join(path, "resources")

    X_train, y_train, X_test, y_test = load_nmist_dataset()

    train_color = ['c' if i == '0' else 'm' for i in y_train]
    test_color = ['c' if i == '0' else 'm' for i in y_test]

    svd_train_components, svd_test_components = SVD_scikit_learn(X_train, X_test)
    pca_train_components, pca_test_components = PCA_scikit_learn(X_train, X_test)
    tsne_train_components, tsne_test_components = TSNE_scikit_learn(X_train, X_test)

    _, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(11, 11))
    ax1.scatter(pca_train_components[:, 0], pca_train_components[:, 1], c=train_color)
    ax1.set_title('PCA train components')
    ax2.scatter(pca_test_components[:, 0], pca_test_components[:, 1], c=test_color)
    ax2.set_title('PCA test components')
    ax3.scatter(tsne_train_components[:, 0], tsne_train_components[:, 1], c=train_color)
    ax3.set_title('TSNE train components')
    ax4.scatter(tsne_test_components[:, 0], tsne_test_components[:, 1], c=test_color)
    ax4.set_title('TSNE test components')
    ax5.scatter(svd_train_components[:, 0], svd_train_components[:, 1], c=train_color)
    ax5.set_title('SVD train components')
    ax6.scatter(svd_test_components[:, 0], svd_test_components[:, 1], c=test_color)
    ax6.set_title('SVD test components')

    plt.tight_layout()
    plt.savefig(os.path.join(resources_path, picture_name))
    return os.path.join(resources_path, picture_name)


def PCA_scikit_learn(X_train, X_test):
    start_time = time.time()
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.fit_transform(X_test)
    print(f"Execution time PCA (s): {time.time() - start_time}")
    return X_train_pca, X_test_pca


def TSNE_scikit_learn(X_train, X_test):
    start_time = time.time()
    tsne = TSNE(n_components=2)
    X_train_tsne = tsne.fit_transform(X_train)
    X_test_tsne = tsne.fit_transform(X_test)
    print(f"Execution time TSNE (s): {time.time() - start_time}")
    return X_train_tsne, X_test_tsne


def SVD_scikit_learn(X_train, X_test):
    start_time = time.time()
    svd = TruncatedSVD(n_components=2)
    X_train_svd = svd.fit_transform(X_train)
    X_test_svd = svd.fit_transform(X_test)
    print(f"Execution time SVD (s): {time.time() - start_time}")
    return X_train_svd, X_test_svd
