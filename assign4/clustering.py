import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from time import time
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import ParameterGrid
from kneed import KneeLocator

def data_acquisition():
    """Fetch the Travel Review Rating dataset from the UCI ML Repository and preprocess it.
    Returns
    -------
    X : pandas DataFrame
        The features of the dataset.
    y : pandas Series
        The target values of the dataset
    """
    # Using UCI ML Repository's Mice Protein 2022 dataset
    # Fetch the dataset
    mice_protein = fetch_ucirepo(id=342)

    # Data as pandas DataFrame
    X = mice_protein.data.features
    y = mice_protein.data.targets
    
    print("Number of features", X.shape[1])
    print("Unique target values", len(pd.unique(y.squeeze())))
    print("Distribution of target values", y.squeeze().value_counts())
    return X, y.squeeze()


def data_preprocessing(X, y):
    """Preprocess the data by replacing missing values, removing categorical features and normalizing numerical features.

    Parameters
    ----------
    X : pandas DataFrame
        The features of the dataset.
    y : pandas Series
        The target values of the dataset.
    Returns
    -------
    x_num_pca : ndarray of shape (n_samples, n_features)
        The preprocessed numerical features after PCA.
    y_num : ndarray of shape (n_samples,)
        The numerical target values corresponding to the numerical features.
    """
    # Remove all categorical features because they are not suitable for KMeans clustering
    x_num = X.select_dtypes(include=["number"])
    print("Number of numerical features", len(x_num.columns))
        
    # Replace missing values with the KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    x_num_complete = imputer.fit_transform(x_num.to_numpy())
    
    # Using QuantileTransformer to normalize the data
    x_num_complete = QuantileTransformer(n_quantiles=50, random_state=0).fit_transform(x_num_complete)
   
    # Normalize all features using standard scaler
    non_categorical_feature = RobustScaler().fit_transform(x_num_complete)

    
    # The feature is redundant and correlated, so we can use PCA to reduce the dimensionality
    pca = PCA(n_components=0.75)
    x_num_pca = pca.fit_transform(non_categorical_feature)
    x_num_pca_norm = RobustScaler().fit_transform(x_num_pca)
    print("Number of numerical features after PCA", x_num_pca.shape[1])
    
    le = LabelEncoder()
    y_num = le.fit_transform(y)
    return x_num_pca_norm, y_num


def cohesion(centers, labels, x):
    """Calculate the cohesion of the clustering.

    Parameters
    ----------
    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers.
    labels : ndarray of shape (n_samples,)
        The labels of the samples.
    x : ndarray of shape (n_samples, n_features)
        The samples.

    Returns
    -------
    float
        The cohesion of the clustering.
    """
    return np.sum(
        np.linalg.norm(centers - np.mean(x, axis=0)[np.newaxis, :], axis=1)
        ** 2
        * np.bincount(labels)
    )


def bench_clustering_algorithm(algorithm, name, data, labels):
    """Benchmark to evaluate different clustering methods.

    Parameters
    ----------
    algorithm : clustering algorithm
        A :class:`~sklearn.cluster` instance with the initialization
        already set.
    name : str
        Name given to the algorithm or strategy.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = algorithm.fit(data)
    fit_time = time() - t0
    results = [
        name,
        fit_time,
        estimator.inertia_ if hasattr(estimator, "inertia_") else 0,
        cohesion(estimator.cluster_centers_, estimator.labels_, data) if hasattr(estimator, "cluster_centers_") else 0,
    ]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator.labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    print(set(estimator.labels_))
    num_label = len(set(estimator.labels_)) if -1 not in set(estimator.labels_) else len(set(estimator.labels_)) - 1
    results += [
        metrics.silhouette_score(
            data,
            estimator.labels_,
            metric="euclidean",
            sample_size=300,
        ) if num_label > 1 else 0
    ]

    # Show the results
    formatter_result = "{:9s}\t{:.3f}s\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    print(formatter_result.format(*results))


def visualize_cluster(data, label, algorithms):
    """Visualize the clustering results with different algorithms using PCA for feature reduction.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        The data to cluster. Data are already subsampled and preprocessed.
    label : ndarray of shape (n_samples,)
        The target values of the dataset
    algorithms : list of tuples
        A list of tuples where each tuple contains the name and the clustering algorithm
        instance.
    """
    reduced_data = PCA(n_components=2).fit_transform(data)
    
    for i, (name, algorithm) in enumerate(algorithms):
        y_pred = algorithm.labels_ if algorithm is not None else label
        plt.subplot(int(len(algorithms) / 2) , 2, i + 1)
        plt.title(name)
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_pred)
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
    plt.show()

def visualize_kmeans(data, kmeans):
    """Visualize the clustering results with KMeans using PCA for feature reduction.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        The data to cluster. Data are already subsampled and preprocessed.
    kmeans : KMeans instance
        The KMeans instance that has been fitted to the data.
    """
    reduced_data = PCA(n_components=2).fit_transform(data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
    )

    # Obtain labels for each point in mesh. Use last trained model.
    kmeans.fit(reduced_data)
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(
        "Visualization of KMeans on Mice Protein dataset.\n"
        "Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    kmeans.fit(data)
    
def visualize_dendrogram(model, **kwargs):
    """Visualize the dendrogram of the agglomerative clustering.

    Parameters
    ----------
    model : AgglomerativeClustering instance
        The AgglomerativeClustering instance that has been fitted to the data.
    **kwargs : dict
        Additional keyword arguments to pass to `dendrogram
    """
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    plt.title("Hierarchical Clustering Dendrogram")
    dendrogram(linkage_matrix, **kwargs)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    
def visualize_dbscan(data, db):
    """Visualize the clustering results with DBSCAN using PCA for feature reduction.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        The data to cluster. Data are already subsampled and preprocessed.
    db : DBSCAN instance
        The DBSCAN instance that has been fitted to the data.
    """
    reduced_data = PCA(n_components=2).fit_transform(data)
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of noise points: %d" % n_noise_)

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = reduced_data[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = reduced_data[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()

# Define a function to subsample the data, keep the label distribution same as original data, only keep 5% of the data
def subsample_data(X, y):
    """Subsample the data and keep the label distribution same as original data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The features of the dataset.
    y : ndarray of shape (n_samples,)
        The target values of the dataset

    Returns
    -------
    X_sub : ndarray of shape (n_samples, n_features)
        The subsampled features of the dataset.
    y_sub : ndarray of shape (n_samples,)
        The subsampled target values of the dataset
    """
    # Subsample the data, keep the label distribution same as original data
    X_sub = []
    y_sub = []
    for label in np.unique(y):
        X_label = X[y == label]
        y_label = y[y == label]
        n_samples = int(0.05 * len(X_label))
        X_sub.append(X_label[:n_samples])
        y_sub.append(y_label[:n_samples])
    X_sub = np.vstack(X_sub)
    y_sub = np.hstack(y_sub)
    return X_sub, y_sub

# Assign target value from string to numerical value
def assign_target_value(y):
    """Assign target value from string to numerical value.

    Parameters
    ----------
    y : pandas Series
        The target values of the dataset

    Returns
    -------
    y_num : ndarray of shape (n_samples,)
        The numerical target values corresponding to the numerical features.
    """
    y_num = y.copy()
    for i, label in enumerate(pd.unique(y)):
        y_num[y == label] = i
    return y_num.to_numpy()

# Determine eps value from elbow method
def determine_eps_value(data, min_samples=5):
    """Determine the eps value from elbow method.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered as a core point.

    Returns
    -------
    eps : float
        The eps value for DBSCAN clustering.
    """
    # Compute the distance of each point to its closest neighbor
    from sklearn.neighbors import NearestNeighbors

    # Define k (min_samples - 1)
    k = min_samples - 1  # Example: min_samples = 5
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(data)
    distances, _ = neigh.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    indices = np.arange(len(distances))

    # Use kneed to find the elbow point
    kneed = KneeLocator(indices, distances, curve="convex", direction="increasing")
    elbow_point = kneed.knee

    # Plot the result
    plt.plot(indices, distances, label='k-th nearest neighbor distance')
    plt.axvline(x=elbow_point, color='r', linestyle='--', label=f'Elbow at {elbow_point}')
    plt.xlabel('Points sorted by distance')
    plt.ylabel('k-th nearest neighbor distance')
    plt.title('kNN Distance Plot with Elbow')
    plt.legend()
    plt.show()
    print("Elbow point", elbow_point, " distance", distances[elbow_point])
    return distances[elbow_point]

def grid_search_dbscan(data, eps, min_samples=5):
    """Grid search to find the best eps value for DBSCAN clustering.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    eps: float
        best eps found from elbow method
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered as a core point.

    Returns
    -------
    best_eps : float
        The best eps value for DBSCAN clustering.
    best_min_samples : int
        The best min_samples value for DBSCAN clustering.
    """
    # Use sklearn ParamGrid to find the best eps and min_samples
    param_grid = {'eps': np.linspace(eps / 2, eps * 2, 10), 'min_samples': np.linspace(int(min_samples/ 2), min_samples * 2, 8, dtype=int)}

    best_params = None
    best_score = -1
    for params in ParameterGrid(param_grid):
        db = DBSCAN(eps=params['eps'], min_samples=params['min_samples']).fit(data)
        unique_labels = set(db.labels_)
        len_labels = len(unique_labels) - (1 if -1 in unique_labels else 0)
        if len_labels > 1:
            score = metrics.silhouette_score(data, db.labels_, metric='euclidean', sample_size=300)
            if score > best_score:
                best_params = params
                best_score = score

    print(f"Best Params: {best_params}, Best Silhouette Score: {best_score}")
    return best_params['eps'], best_params['min_samples']

# Main function to run the clustering algorithms
def main():
    X, y = data_acquisition()
    x_num_pca, y_num = data_preprocessing(X, y)
    num_cluster = len(set(y_num))
    
    # Generate different clustering algorithms
    kmeans = KMeans(init="k-means++", n_clusters=num_cluster, n_init=10)
    
    # Determine eps value
    min_samples = 5
    eps = determine_eps_value(x_num_pca, min_samples)
    print("Elbow eps value", eps)
    eps_modifier = 1.35
    db = DBSCAN(eps=eps * eps_modifier, min_samples=min_samples, n_jobs=-1)
    agglomerative = AgglomerativeClustering(n_clusters=num_cluster, connectivity=None, linkage="ward", compute_distances=True)

    # KMeans clustering
    print("Benchmarking kmeans & dbscan & hierarchal clustering")
    print(82 * "_")
    print("Init\tTime\tSeparation\tCohesion\tHomogeneity\tCompleteness\tV-measure\tARI\tAMI\tSilhouette")
    bench_clustering_algorithm(kmeans, "k-means", x_num_pca, y_num)
    bench_clustering_algorithm(db, "DBSCAN", x_num_pca, y_num)
    bench_clustering_algorithm(agglomerative, "agglomerative clustering", x_num_pca, y_num)
    print(82 * "_")

    # Visualize the clustering results with KMeans
    # visualize_kmeans(x_num_pca, kmeans)
    
    # Visualize the clustering results with DBSCAN
    # visualize_dbscan(x_num_pca, db)
   
    # Visualize the dendrogram of the agglomerative clustering
    # visualize_dendrogram(agglomerative, truncate_mode='level', p=3)
    
    # Visualize the clustering results
    algorithms = [
        ("KMeans", kmeans),
        ("DBSCAN", db),
        ("Agglomerative Clustering", agglomerative),
        ("Ground Truth", None),
    ]  
    visualize_cluster(x_num_pca, y_num, algorithms)  

if __name__ == "__main__":
    main()
