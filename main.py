import numpy as np
import random
from fastapi import FastAPI
from fastapi.responses import FileResponse

from Matrix import Matrix
from Picture import Picture
from Unsupervised import descompose_my_picture, plot_singular_values, \
    difference_my_picture_and_aproximation

app = FastAPI()

'''
What is the rank and trace of A?
- What is the determinant of A?
- Can you invert A? How?
- How are eigenvalues and eigenvectors of A’A and AA’ related? What interesting differences can you notice between both?
'''


@app.post("/matrixOperations")
async def matrix_operations(rows: int = random.randint(1, 20), cols: int = random.randint(1, 20)):
    matrix = Matrix(rows, cols)
    matrix_data = matrix.create(rows, cols)
    eigen_transponseA_A = matrix.get_eigen_values_vectors(np.dot(matrix_data.T, matrix_data))
    eigen_A_transponseA = matrix.get_eigen_values_vectors(np.dot(matrix_data, matrix_data.T))
    return {
        "data": matrix_data.tolist(),
        "rank": int(matrix.calculate_rank(matrix_data)),
        "trace": int(matrix.calculate_trace(matrix_data)),
        "determinant": matrix.calculate_determinant(matrix_data),
        "inverse": matrix.calculate_inverse(matrix_data),
        "eigenvalues A'A": np.array_str(eigen_transponseA_A.get("eigenvalues"), precision=4),
        "eigenvectors A'A": np.array_str(eigen_transponseA_A.get("eigenvectors"), precision=4),
        "eigenvalues AA'": np.array_str(eigen_A_transponseA.get("eigenvalues"), precision=4),
        "eigenvectors AA'": np.array_str(eigen_A_transponseA.get("eigenvectors"), precision=4)
    }


@app.get("/editMyPicture")
async def edit_and_plot_my_picture():
    picture = Picture()
    picture.edit_my_picture()
    my_picture = picture.get_my_picture()
    return FileResponse(my_picture, media_type="image/jpg")


@app.get("/calculatePictureAverage")
async def calculate_and_plot_picture_average():
    picture = Picture()
    avg_picture = picture.get_picture_average()
    return FileResponse(avg_picture, media_type="image/jpg")


@app.get("/calculateDistanceFromMyPictureToAverage")
async def calculate_distance_my_picture_to_average():
    picture = Picture()
    distance = picture.calculate_distance_my_picture_to_avg()
    return {"Distance from my picture to the average (MSE)": np.round(distance, 3)}


@app.get("/plotSingularValues")
async def plot_my_picture_singular_values():
    picture = Picture()
    my_picture = picture.edit_my_picture()
    singular_values_picture = plot_singular_values(my_picture)
    return FileResponse(singular_values_picture, media_type="image/jpg")


@app.get("/applySvdOverMyPicture")
async def apply_svd_over_my_picture():
    picture = Picture()
    my_picture = picture.edit_my_picture()
    svd_picture = descompose_my_picture(my_picture)
    return FileResponse(svd_picture, media_type="image/jpg")


@app.get("/calculateDistanceFromMyPictureToAproximation")
async def calculate_difference_between_my_picture_and_aproximation():
    picture = Picture()
    my_picture = picture.edit_my_picture()
    difference = difference_my_picture_and_aproximation(my_picture)
    return difference
