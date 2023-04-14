### **Answers to theoretical questionsA**
  
#### **8.** What strategies do you know (or can think of) in order to make PCA more robust?
>  - Scaling the data: it can help to avoid the dominance of one variable over the others. This can be done using techniques such as standardization, normalization, and so on.
>  - Handling outliers: outliers can have a significant effect on the principal components and can distort the results of the analysis. This can be done removing  them from the data or using robust PCA method.
>  - Dealing with missing data: one way to handle missing data is to impute them using techniques such as mean imputation, median imputation, or interpolation.
>>  Robust PCA  is a variation of PCA that is designed to be more robust to outliers and corruptions in the data. In RPCA, the principal components are calculated by decomposing the data matrix into two parts: a *low-rank* matrix and a *sparse* matrix. The low-rank matrix contains the principal components, while the sparse matrix contains the outliers or corruptions. This decomposition is achieved by solving an optimization problem (with methods such as ADMM, ALM, PCP...) that minimizes the rank of the low-rank matrix subject to a constraint on the sparsity of the sparse matrix.

#### **9.** What are the underlying mathematical principles behind UMAP? What is it useful for?
> Uniform Manifold Approximation and Projection (UMAP) is a nonlinear dimensionality reduction method used to visualize and analyze large, high-dimensional data sets.
> 
> The mathematical principles underlying UMAP combine techniques from graph theory, machine learning, algebraic topology and Riemannian geometry to construct a visual representation of high-dimensional data in a low-dimensional space. This representation allows a clearer visualization of the structures and patterns present in the data, making it easier to analyze and understand.
#### **10.** What are the underlying mathematical principles behind LDA? What is it useful for?
> Linear Discriminant Analysis (LDA) is a supervised machine learning algorithm used for classification and dimensionality reduction.
> 
> The underlying mathematical principles behind LDA involve linear algebra, statistics  and probability theory.
> LDA uses the covariance matrix of the data to determine the linear combination of features that best separates the classes, and then projects the data into this feature space to reduce dimensionality and visualize relationships between classes.