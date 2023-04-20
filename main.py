import random
import time

import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse

from clustering import ClusteringUnsupervised, ClusteringScikitLearn
from clustering.Data import create_toy_data
from matrix.Matrix import Matrix
from picture.Picture import Picture
from scikitlearn import ScikitLearn
from unsupervised import Unsupervised
from unsupervised.Unsupervised import descompose_my_picture, plot_singular_values, \
    difference_my_picture_and_approximation, plot_dimension_reduction_methods_from_scratch

app = FastAPI()


@app.post("/1-matrix-operations")
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


@app.get("/2-required-download-face-pictures-from-google-drive")
async def download_required_pictures():
    picture = Picture()
    picture.load_pictures()
    return {}


@app.get("/2-edit-my-picture")
async def edit_and_plot_my_picture():
    picture = Picture()
    picture.edit_my_picture()
    my_picture = picture.get_my_picture()
    return FileResponse(my_picture, media_type="image/jpg")


@app.get("/2-calculate-picture-average")
async def calculate_and_plot_picture_average():
    picture = Picture()
    avg_picture = picture.get_picture_average()
    return FileResponse(avg_picture, media_type="image/jpg")


@app.get("/3-calculate-distance-from-my-picture-to-average")
async def calculate_distance_my_picture_to_average():
    picture = Picture()
    distance = picture.calculate_distance_my_picture_to_avg()
    return {"Distance from my picture to the average (MSE)": np.round(distance, 3)}


@app.get("/4-plot-singular-values")
async def plot_my_picture_singular_values():
    picture = Picture()
    my_picture = picture.edit_my_picture()
    singular_values_picture = plot_singular_values(my_picture)
    return FileResponse(singular_values_picture, media_type="image/jpg")


@app.get("/4-apply-svd-over-my-picture")
async def apply_svd_over_my_picture():
    picture = Picture()
    my_picture = picture.edit_my_picture()
    svd_picture = descompose_my_picture(my_picture)
    return FileResponse(svd_picture, media_type="image/jpg")


@app.get("/4-calculate-distance-from-my-picture-to-approximation")
async def calculate_difference_between_my_picture_and_approximation():
    picture = Picture()
    my_picture = picture.edit_my_picture()
    difference = difference_my_picture_and_approximation(my_picture)
    return difference


@app.get("/5-train-mnist-dataset-with-logistic-regression")
async def train_mnist_dataset_with_scikit_learn():
    X_train, y_train, X_test, y_test = ScikitLearn.load_nmist_dataset()
    start_time = time.time()
    model = ScikitLearn.train_logistic_regression_model(X_train, y_train)
    return {"Score logistic regression": model.score(X_test, y_test),
            "Execution time (s)": time.time() - start_time}


@app.get("/6-plot-new-features-generated-methods-from-scratch")
async def plot_two_features_generated_methods_from_scratch():
    plot = plot_dimension_reduction_methods_from_scratch()
    return FileResponse(plot, media_type="image/jpg")


@app.get("/6-train-logistic-regression-unsupervised")
async def train_mnist_dataset_unsupervised():
    X_train, y_train, X_test, y_test = ScikitLearn.load_nmist_dataset()

    X_train_svd = Unsupervised.SVD_from_scratch(X_train)
    X_test_svd = Unsupervised.SVD_from_scratch(X_test)
    start_time_model_svd = time.time()
    model_svd = ScikitLearn.train_logistic_regression_model(X_train_svd, y_train)
    time_model_svd = time.time() - start_time_model_svd

    X_train_pca = Unsupervised.PCA_from_scratch(X_train)
    X_test_pca = Unsupervised.PCA_from_scratch(X_test)
    start_time_model_pca = time.time()
    model_pca = ScikitLearn.train_logistic_regression_model(X_train_pca, y_train)
    time_model_pca = time.time() - start_time_model_pca

    X_train_tsne = Unsupervised.TSNE_from_scratch(X_train)
    X_test_tsne = Unsupervised.TSNE_from_scratch(X_test)
    start_time_model_tsne = time.time()
    model_tsne = ScikitLearn.train_logistic_regression_model(X_train_tsne, y_train[:500])
    time_model_tsne = time.time() - start_time_model_tsne

    return {
        "Score logistic regression with SVD": model_svd.score(X_test_svd, y_test),
        "Execution time fit model with SVD (s)": time_model_svd,
        "Score logistic regression with PCA": model_pca.score(X_test_pca, y_test),
        "Execution time fit model with PCA (s)": time_model_pca,
        "Score logistic regression with TSNE": model_tsne.score(X_test_tsne, y_test[:500]),
        "Execution time fit model with TSNE (s)": time_model_tsne
    }


@app.get("/7-plot-new-features-generated-by-scikit-learn")
async def plot_two_features_generated_by_scikit_learn():
    plot = ScikitLearn.plot_dimension_reduction_scikit_learn()
    return FileResponse(plot, media_type="image/jpg")


@app.get("/7-train-with-scikit-learn")
async def train_mnist_dataset_with_scikit_learn():
    X_train, y_train, X_test, y_test = ScikitLearn.load_nmist_dataset()

    X_train_svd = ScikitLearn.SVD_scikit_learn(X_train)
    X_test_svd = ScikitLearn.SVD_scikit_learn(X_test)
    start_time_model_svd = time.time()
    model_svd = ScikitLearn.train_logistic_regression_model(X_train_svd, y_train)
    time_model_svd = time.time() - start_time_model_svd

    X_train_pca = ScikitLearn.PCA_scikit_learn(X_train)
    X_test_pca = ScikitLearn.PCA_scikit_learn(X_test)
    start_time_model_pca = time.time()
    model_pca = ScikitLearn.train_logistic_regression_model(X_train_pca, y_train)
    time_model_pca = time.time() - start_time_model_pca

    X_train_tsne = ScikitLearn.TSNE_scikit_learn(X_train)
    X_test_tsne = ScikitLearn.TSNE_scikit_learn(X_test)
    start_time_model_tsne = time.time()
    model_tsne = ScikitLearn.train_logistic_regression_model(X_train_tsne, y_train)
    time_model_tsne = time.time() - start_time_model_tsne

    return {
        "Score logistic regression with SVD": model_svd.score(X_test_svd, y_test),
        "Execution time fit model with SVD (s)": time_model_svd,
        "Score logistic regression with PCA": model_pca.score(X_test_pca, y_test),
        "Execution time fit model with PCA (s)": time_model_pca,
        "Score logistic regression with TSNE": model_tsne.score(X_test_tsne, y_test),
        "Execution time fit model with TSNE (s)": time_model_tsne
    }


@app.post("/11-classify-mnist-image-svd")
async def classify_image_svd(file: UploadFile):
    image = ScikitLearn.read_image(file)
    X_train, y_train, X_test, y_test = ScikitLearn.load_nmist_dataset()
    X_train_red = Unsupervised.SVD_from_scratch(X_train)
    model = ScikitLearn.train_logistic_regression_model(X_train_red, y_train)
    return {"predicted label": model.predict(image)[0]}


@app.post("/11-classify-mnist-image-pca")
async def classify_image_pca(file: UploadFile):
    image = ScikitLearn.read_image(file)
    X_train, y_train, X_test, y_test = ScikitLearn.load_nmist_dataset()
    X_train_red = Unsupervised.PCA_from_scratch(X_train)
    model = ScikitLearn.train_logistic_regression_model(X_train_red, y_train)
    return {"predicted label": model.predict(image)[0]}


@app.post("/11-classify-mnist-image-tsne")
async def classify_image_tsne(file: UploadFile):
    image = ScikitLearn.read_image(file)
    X_train, y_train, X_test, y_test = ScikitLearn.load_nmist_dataset()
    X_train_red = Unsupervised.TSNE_transform_from_scratch(X_train)
    model = ScikitLearn.train_logistic_regression_model(X_train_red, y_train[:500])
    return {"predicted label": model.predict(image)[0]}


@app.get("/plot-toy-data")
async def plot_data_dummy():
    X, y = create_toy_data()
    plot = ClusteringUnsupervised.plot_data(X, y)
    return FileResponse(plot, media_type="image/jpg")


@app.get("/calculate-distance-between-clusters")
async def calculate_clusters_distance():
    X, y = create_toy_data()
    centroids_distances = ClusteringUnsupervised.calculate_distances(X)
    return centroids_distances


@app.get("/calculate-the-silhouette-plots-and-coefficients-k-means")
async def calculate_the_silhouette_plot_k_means():
    X, y = create_toy_data()
    plot = ClusteringUnsupervised.plot_silhouette_coefficients_k_means(X, y)
    return FileResponse(plot, media_type="image/jpg")


@app.get("/calculate-the-silhouette-plots-and-coefficients-k-medoids")
async def calculate_the_silhouette_plot_k_medoids():
    X, y = create_toy_data()
    plot = ClusteringUnsupervised.plot_silhouette_coefficients_k_medoids(X)
    return FileResponse(plot, media_type="image/jpg")


@app.get("/plot-scattered-data")
async def plot_scattered_data():
    plot = ClusteringScikitLearn.plot_scattered_data()
    return FileResponse(plot, media_type="image/jpg")
