from fastapi import FastAPI
from fastapi.responses import FileResponse

from unsupervised.clustering import ClusteringScikitLearn, ClusteringUnsupervised
from unsupervised.clustering.ClusteringScikitLearn import plot_scattered_with_cluster_methods
from unsupervised.clustering.Data import create_toy_data

app = FastAPI()


@app.get("/5-b-plot-toy-data")
async def plot_data_dummy():
    X, y = create_toy_data()
    plot = ClusteringUnsupervised.plot_data(X, y)
    return FileResponse(plot, media_type="image/jpg")


@app.get("/5-b-calculate-distance-between-clusters")
async def calculate_clusters_distance():
    X, y = create_toy_data()
    centroids_distances = ClusteringUnsupervised.calculate_distances(X)
    return centroids_distances


@app.get("/5-c-calculate-the-silhouette-plots-and-coefficients-k-means")
async def calculate_the_silhouette_plot_k_means():
    X, y = create_toy_data()
    plot = ClusteringUnsupervised.plot_silhouette_coefficients_k_means(X)
    return FileResponse(plot, media_type="image/jpg")


@app.get("/5-c-calculate-the-silhouette-plots-and-coefficients-k-medoids")
async def calculate_the_silhouette_plot_k_medoids():
    X, y = create_toy_data()
    plot = ClusteringUnsupervised.plot_silhouette_coefficients_k_medoids(X)
    return FileResponse(plot, media_type="image/jpg")


@app.get("/6-a-plot-scattered-data")
async def plot_scattered_data():
    plot = ClusteringScikitLearn.plot_scattered_data()
    return FileResponse(plot, media_type="image/jpg")


@app.get("/6-b-plot-scattered-with-scikitlearn_noisy_circles")
async def plot_scattered_with_scikitlearn_cluster():
    plot = plot_scattered_with_cluster_methods()
    return FileResponse(plot, media_type="image/jpg")
