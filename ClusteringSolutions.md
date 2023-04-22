## Lab2 ML II 
### Clustering
  
#### **1.** Research about the **Spectral Clustering** method, and answer the following questions:
Spectral Clustering is a popular machine learning method used for clustering data points. It is a graph-based clustering technique that leverages the spectral properties of data to group similar data points together.
- **a.** In which cases might it be more useful to apply?
>- It's useful when working with data that have complex or non-linear cluster shapes.
>- For high dimensional data, since the algorithm embeds the graph in a lower dimensional space, it can help reduce dimensionality and improve clustering efficiency on data with many features.
>- For data that do not follow a linear distribution and have more complex relationships.
>- For data with presence of noise or outliers, because it is based on the global structure of the graph, instead of depending on the position of the data points in the original space.
- **b.** What are the mathematical fundamentals of it?
> Spectral Clustering is a graph-based clustering method that utilizes graph theory, linear algebra, and spectral decomposition to identify clusters in data. It involves graph representation, graph construction, graph embedding, and clustering using lower-dimensional representations obtained from the eigenvectors of the graph Laplacian matrix.
- **c.** What is the algorithm to compute it?
> **1. Input:** Affinity matrix representing pairwise similarities or distances between data points.   
> **2. Compute Graph Laplacian:** Use the affinity matrix to compute the graph Laplacian matrix, which is a symmetric and positive semidefinite matrix.   
> **3. Eigenvector Decomposition:** Perform eigenvalue decomposition on the graph Laplacian matrix to obtain the eigenvectors and eigenvalues.  
> **4. Dimensionality Reduction:** Select a subset of the eigenvectors corresponding to the desired number of clusters or subspace dimensionality.  
> **5. Cluster Assignment:** Apply a clustering algorithm, such as k-means, spectral clustering, or other clustering techniques, to the reduced-dimensional embedding to assign data points to clusters.  
> **6. Post-Processing:** Optionally, perform post-processing steps, such as refining cluster assignments, handling outliers, or incorporating domain-specific constraints or prior knowledge.    
> **7. Interpretation and Evaluation:** Interpret the clustering results in the context of the specific problem domain and evaluate the quality of the clustering using appropriate evaluation metrics.   
> **8. Output:** Final cluster assignments or embeddings for further analysis or visualization.
- **d.** Does it hold any relation to some of the concepts previously mentioned in class? Which, and how?
> It is related to the PCA dimensionality reduction method, since Spectral Clustering also uses the decomposition of eigenvectors and eigenvalues to reduce the dimensionality of the data. 

#### **2.** Research about the **DBSCAN** method, and answer the following questions:
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a popular algorithm for unsupervised clustering  that is based on the density of points in space to identify clusters or groups. It is a non-parametric clustering algorithm, which means that it does not require the specification of a pre-defined number of clusters.
- **a.** In which cases might it be more useful to apply?
> - It's useful when the number of clusters is unknown or can vary.
> - When dealing with datasets with a large number of data points, as it is designed to be scalable and efficient.
> - When dealing with data with varying densities, as it can effectively identify clusters of different shapes, sizes, and densities.
> - When noise detection is important.
> - When dealing with spatial or geometric data such as GPS coordinates, image data, or other types of data with inherent spatial relationships.
- **b.** What are the mathematical fundamentals of it?
> - DBSCAN is based on local density, distance and relationship to neighboring points in a data set. 
- **c.** Is there any relation between DBSCAN and Spectral Clustering? If so, what is it?
> No, Both DBSCAN and Spectral Clustering are used for clustering data, they have different mathematical foundations and operate on different principles. DBSCAN is based on local density and distance, while Spectral Clustering is based on spectral graph theory.

#### **3.** What is the elbow method in clustering? And which flaws does it pose to assess quality?
> The elbow method is a popular technique used in clustering to help determine the optimal number of clusters to use for a given dataset. It involves plotting the variance explained (or other cluster quality metric) as a function of the number of clusters, and identifying the "elbow" point on the plot, which is typically the point where the variance explained starts to level off after a steep decrease. The number of clusters corresponding to the elbow point is often chosen as the optimal number of clusters for the dataset.
>
> Despite its popularity, the elbow method has some limitations and potential flaws in assessing the quality of clustering results:
> - The identification of the elbow point is often subjective and relies on visual inspection of the plot, which may vary depending on the individual's perception and interpretation.
> - It is dataset-dependent,the optimal number of clusters may vary depending on the specific dataset being analyzed.
> - The elbow method assumes that all clusters have equal sizes, and the variance explained (or other quality metric) is an appropriate measure of cluster quality.

#### **5.b.** How many clusters are there? How far are they from one another?
***Run endpoints "/5-b-plot-toy-data" and "/5-b-calculate-distance-between-clusters" to see the plots and distances***
> There are 4 clusters. The distance was calculated using the Euclidean distance between all the centroids obtained by K-means.
>> - "Distance between cluster #1 and cluster #2": 8.861
>> - "Distance between cluster #1 and cluster #3": 11.846
>> - "Distance between cluster #1 and cluster #4": 13.718
>> - "Distance between cluster #2 and cluster #3": 3.983
>> - "Distance between cluster #2 and cluster #4": 5.039
>> - "Distance between cluster #3 and cluster #4": 5.163
 
#### **5.d.** What number of K got the best silhouette score? What can you say about the figures? Is this the expected result?
***Run endpoints "/5-c-calculate-the-silhouette-plots-and-coefficients-k-means" and "/5-c-calculate-the-silhouette-plots-and-coefficients-k-medoids" to see the plots***
> K=2 got the best result for both k-means and k-medoids with an Average Silhouette of 0.7050
> From the figures it can be seen that for the test data, the best result is obtained for k=2, followed by k=4. In general, the k-means and k-medoids clustering algorithms have a good performance for data in blob form or distributed in groups.

#### **6.b.** Apply k-means, k-medoids, DBSCAN and Spectral Clustering from Scikit-Learn over each dataset and compare the results of each algorithm with respect to each dataset.
***Run endpoint "/6-a-plot-scattered-data" to see the plot***
> - K-means and K-medoids obtained better results on blob data and acceptable scores on Anistropic data.
> - DBSCAN: performed poorly with unstructured data and standard deviation blobs in its clusters, but was able to identify outliers. 
> - Spectral Clustering: performed very well for the blobs with standard deviation and in general has good behavior, except for the clustering in the blobs when their grouping is concentrated.
> - DBSCAN and Spectral clustering, Visually it can be seen that they are very good at separating data in the form of noisy moons and noisy circles, but being nonlinear algorithms, the silhouette coefficient metric does not score well.
