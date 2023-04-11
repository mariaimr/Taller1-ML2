import numpy as np


class Matrix:
    def __init__(self, rows, cols):
        self.rows = int(rows)
        self.cols = int(cols)

    @classmethod
    def create(cls, rows, cols):
        rows = int(rows)
        cols = int(cols)
        matrix = np.random.randint(1, 100, (rows, cols))
        return np.array(matrix)

    @classmethod
    def calculate_rank(cls, matrix):
        return np.linalg.matrix_rank(matrix)

    @classmethod
    def calculate_trace(cls, matrix):
        return np.trace(matrix)

    @classmethod
    def calculate_determinant(cls, matrix):
        if matrix.shape[0] == matrix.shape[1]:
            return int(np.linalg.det(matrix))
        else:
            return "Determinant is only defined for square matrices"

    @classmethod
    def calculate_inverse(cls, matrix):
        if matrix.shape[0] == matrix.shape[1]:
            if cls.calculate_determinant(matrix) != 0:
                return np.linalg.inv(matrix).tolist()
            else:
                "Inverse cannot be calculated because the matrix determinant is 0"
        else:
            return "Inverse is only defined for square matrices"

    @classmethod
    def get_eigen_values_vectors(cls, matrix):
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors
        }
