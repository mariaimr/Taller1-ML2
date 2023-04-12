import time

import numpy as np
import random
from fastapi import FastAPI
from fastapi.responses import FileResponse

import ScikitLearn
import Unsupervised
from Matrix import Matrix
from Picture import Picture
from Unsupervised import descompose_my_picture, plot_singular_values, \
    difference_my_picture_and_aproximation, plot_dimension_reduction_methods_from_scratch

app = FastAPI()


@app.post("/matrix-perations")
async def matrix_operations(rows: int = random.randint(1, 10), cols: int = random.randint(1, 10)):
    matrix = Matrix(rows, cols)
    matrix_data = matrix.create(rows, cols)
    eigen_transposeA_A = matrix.get_eigen_values_vectors(matrix_data.T @ matrix_data)
    eigen_A_transposeA = matrix.get_eigen_values_vectors(matrix_data @ matrix_data.T)
    return {
        "data": matrix_data.tolist(),
        "rank": int(matrix.calculate_rank(matrix_data)),
        "trace": matrix.calculate_trace(matrix_data),
        "determinant": matrix.calculate_determinant(matrix_data),
        "inverse": matrix.calculate_inverse(matrix_data),
        "eigenvalues A'A": np.array_str(eigen_transposeA_A.get("eigenvalues"), precision=4),
        "Shape eigenvalues A'A": eigen_transposeA_A.get("eigenvalues").shape,
        "eigenvectors A'A": np.array_str(eigen_transposeA_A.get("eigenvectors"), precision=4),
        "Shape eigenvectors A'A": eigen_transposeA_A.get("eigenvectors").shape,
        "eigenvalues AA'": np.array_str(eigen_A_transposeA.get("eigenvalues"), precision=4),
        "Shape eigenvalues AA'": eigen_A_transposeA.get("eigenvalues").shape,
        "eigenvectors AA'": np.array_str(eigen_A_transposeA.get("eigenvectors"), precision=4),
        "Shape eigenvectors AA'": eigen_A_transposeA.get("eigenvectors").shape,
        "Conclusion": "if the matrix A is square, the eigenvalues of A'A are equal to those of AA' and the "
                      "eigenvectors are different. If matrix A is rectangular, both eigenvalues and eigenvectors are "
                      "different for A'A and AA'"
    }


@app.get("/edit-my-picture")
async def edit_and_plot_my_picture():
    picture = Picture()
    picture.edit_my_picture()
    my_picture = picture.get_my_picture()
    return FileResponse(my_picture, media_type="image/jpg")


@app.get("/calculate-picture-average")
async def calculate_and_plot_picture_average():
    picture = Picture()
    avg_picture = picture.get_picture_average()
    return FileResponse(avg_picture, media_type="image/jpg")


@app.get("/calculate-distance-from-my-picture-to-average")
async def calculate_distance_my_picture_to_average():
    picture = Picture()
    distance = picture.calculate_distance_my_picture_to_avg()
    return {"Distance from my picture to the average (MSE)": np.round(distance, 3)}


@app.get("/plot-singular-values")
async def plot_my_picture_singular_values():
    picture = Picture()
    my_picture = picture.edit_my_picture()
    singular_values_picture = plot_singular_values(my_picture)
    return FileResponse(singular_values_picture, media_type="image/jpg")


@app.get("/apply-svd-over-my-picture")
async def apply_svd_over_my_picture():
    picture = Picture()
    my_picture = picture.edit_my_picture()
    svd_picture = descompose_my_picture(my_picture)
    return FileResponse(svd_picture, media_type="image/jpg")


@app.get("/calculate-distance-from-my-picture-to-approximation")
async def calculate_difference_between_my_picture_and_approximation():
    picture = Picture()
    my_picture = picture.edit_my_picture()
    difference = difference_my_picture_and_aproximation(my_picture)
    return difference


@app.get("/train-mnist-dataset-with-logistic-regression")
async def train_mnist_dataset_with_scikit_learn():
    X_train, y_train, X_test, y_test = ScikitLearn.load_nmist_dataset()
    start_time = time.time()
    score = ScikitLearn.train_logistic_regression_model(X_train, y_train, X_test, y_test)
    return {"Score logistic regression": score,
            "Execution time (s)": time.time() - start_time}


@app.get("/plot-new-features-generated-methods-from-scratch")
async def plot_two_features_generated_methods_from_scratch():
    plot = plot_dimension_reduction_methods_from_scratch()
    return FileResponse(plot, media_type="image/jpg")


@app.get("/train-with-logistic-regression-svd")
async def train_mnist_dataset_with_svd_unsupervised():
    X_train, y_train, X_test, y_test = ScikitLearn.load_nmist_dataset()
    X_train, X_test = Unsupervised.SVD_from_scratch(X_train, X_test)
    start_time = time.time()
    score = ScikitLearn.train_logistic_regression_model(X_train, y_train, X_test, y_test)
    return {"Score logistic regression with SVD": score,
            "Execution time (s)": time.time() - start_time}


@app.get("/train-with-logistic-regression-pca")
async def train_mnist_dataset_with_pca_unsupervised():
    X_train, y_train, X_test, y_test = ScikitLearn.load_nmist_dataset()
    X_train, X_test = Unsupervised.PCA_from_scratch(X_train, X_test)
    start_time = time.time()
    score = ScikitLearn.train_logistic_regression_model(X_train, y_train, X_test, y_test)
    return {"Score logistic regression with PCA": score,
            "Execution time (s)": time.time() - start_time}


@app.get("/train-with-logistic-regression-tsne")
async def train_mnist_dataset_with_tsne_unsupervised():
    X_train, y_train, X_test, y_test = ScikitLearn.load_nmist_dataset()
    X_train, X_test = Unsupervised.TSNE_from_scratch(X_train, y_train, X_test, y_test)
    start_time = time.time()
    score = ScikitLearn.train_logistic_regression_model(X_train, y_train[:500], X_test, y_test[:500])
    return {"Score logistic regression with TSNE": score,
            "Execution time (s)": time.time() - start_time}


@app.get("/plot-new-features-generated-by-scikit-learn")
async def plot_two_features_generated_by_scikit_learn():
    plot = ScikitLearn.plot_dimension_reduction_scikit_learn()
    return FileResponse(plot, media_type="image/jpg")


@app.get("/train-with-svd-scikit-learn")
async def train_mnist_dataset_with_svd_scikit_learn():
    X_train, y_train, X_test, y_test = ScikitLearn.load_nmist_dataset()
    X_train, X_test = ScikitLearn.SVD_scikit_learn(X_train, X_test)
    start_time = time.time()
    score = ScikitLearn.train_logistic_regression_model(X_train, y_train, X_test, y_test)
    return {"Score logistic regresion with SVD": score,
            "Execution time (s)": time.time() - start_time}


@app.get("/train-with-pca-scikit-learn")
async def train_mnist_dataset_with_pca_scikit_learn():
    X_train, y_train, X_test, y_test = ScikitLearn.load_nmist_dataset()
    X_train, X_test = ScikitLearn.PCA_scikit_learn(X_train, X_test)
    start_time = time.time()
    score = ScikitLearn.train_logistic_regression_model(X_train, y_train, X_test, y_test)
    return {"Score logistic regression with PCA": score,
            "Execution time (s)": time.time() - start_time}


@app.get("/train-with-tsne-scikit-learn")
async def train_mnist_dataset_with_tsne_scikit_learn():
    X_train, y_train, X_test, y_test = ScikitLearn.load_nmist_dataset()
    X_train, X_test = ScikitLearn.TSNE_scikit_learn(X_train, X_test)
    start_time = time.time()
    score = ScikitLearn.train_logistic_regression_model(X_train, y_train, X_test, y_test)
    return {"Score logistic regression with TSNE": score,
            "Execution time (s)": time.time() - start_time}
