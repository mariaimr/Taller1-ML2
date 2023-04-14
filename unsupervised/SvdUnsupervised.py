import numpy as np


class SvdUnsupervised:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, matrix):
        u_matrix, sigma_matrix, v_matrix = np.linalg.svd(matrix)
        sigma_matrix = np.diag(sigma_matrix)
        return u_matrix, sigma_matrix, v_matrix

    def fit_transform(self, matrix):
        u_matrix, sigma_matrix, v_matrix = self.fit(matrix)
        img_reconstructed = u_matrix[:, :self.n_components] \
                            @ sigma_matrix[:self.n_components, :self.n_components] \
                            @ v_matrix[:self.n_components, :]
        return img_reconstructed
