import numpy as np


def fit_svd(matrix):
    u_matrix, sigma_matrix, v_matrix = np.linalg.svd(matrix)
    sigma_matrix = np.diag(sigma_matrix)
    return u_matrix, sigma_matrix, v_matrix


class SvdUnsupervised:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, matrix):
        u_matrix, sigma_matrix, v_matrix = np.linalg.svd(matrix)
        sigma_matrix = np.diag(sigma_matrix)
        img_reconstructed = np.matmul(u_matrix[:, :self.n_components],
                                      sigma_matrix[:self.n_components, :self.n_components])
        img_reconstructed = np.matmul(img_reconstructed, v_matrix[:self.n_components, :])
        return img_reconstructed

