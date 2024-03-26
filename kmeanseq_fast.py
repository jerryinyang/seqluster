import numpy as np 
from sklearn.cluster import KMeans # noqa
from dtaidistance import dtw # noqa
from collections import deque # noqa


class SeqKMeans: 

    def __init__(self, n_clusters, random_state=14) -> None:
        self.n_clusters = self.k = n_clusters

        # Store KMeans model and properties
        self._kmeans = None
        self._random_state = random_state
        self._model_fitted = False

        # Store the base data
        self._base_mean_distance = None
        self._base_cluster_counts = None

        # Store the batch data
        self._batch_centroids = None # Stores the mean data point of the batch for each cluster
        self._batch_distance_sums = None # For computing intra-cluster mean pairwaise distance
        self._batch_cluster_counts = None # Stores the count of data points for each cluster 


    def fit(self, X, max_iterations=300):
        kmeans = KMeans(n_clusters=self.n_clusters, 
                        init='k-means++', 
                        n_init=10, 
                        random_state=self._random_state, 
                        max_iter=max_iterations)
        kmeans.fit(X)

        # Store the kmeans model
        self._kmeans = kmeans

        # Get the centroids 
        labels_ = kmeans.labels_
        centroids_ = kmeans.cluster_centers_

        # Store the base stats
        # Calculate distances of each point to its assigned centroid
        distances = np.linalg.norm(X - centroids_[labels_], axis=1)

        # Calculate the average intra-cluster distances per cluster
        self._base_mean_distance = np.array([np.mean(distances[labels_ == label]) for label in range(self.k)])
        self._base_cluster_counts = np.bincount(labels_)

        self._model_fitted = True
        return kmeans.labels_
    

    def predict(self, X):
        # Make X a numpy array
        if type(X) is not np.ndarray:
            X = np.array(X)

        # Predict the labels for the new data
        labels = self._kmeans.predict(X)

        # Perform Centroid Updates
        self.sequential_learn(X, labels)

        return labels


    def sequential_learn(self, X, labels):
        # Collect batch data
        # Compute/monitor the current batch's centroids
        
        return

if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt  # noqa

    # region 1. Prepare the data s
    prices = pd.read_parquet('prices.parquet')

    raw_data = prices["close"].dropna(axis=0)
    raw_data = raw_data.to_numpy()

    # Generate windows
    window_size = 24

    X = []
    
    for index in range(window_size, len(raw_data)):
        start_index = index - window_size
        end_index = index

        X.append(raw_data[start_index : end_index])

    # Split X Data
    split_percent = .7
    split_index = int(round(len(X) * split_percent))
    
    X, X_test = np.array(X[:split_index]), np.array(X[split_index:])
    X = np.random.permutation(X)
    X_test = np.random.permutation(X_test)

    # endregion

    kmeans = SeqKMeans(4)
    labels = kmeans.fit(X)
    print(labels)
 