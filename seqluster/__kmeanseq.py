from collections import deque  # noqa

import numpy as np
from dtaidistance import dtw  # noqa
from seqluster.__kmeans import KMeansPlus

from seqluster.utils import clear_terminal, debug  # noqa


class KMeansSeq(KMeansPlus): 
    """
    The model uses a kmeans++ clustering model, and sequentially updates the centroid definitions.
    """ 

    def __init__(self, k, confluence_metric=None, batch_size=100, learning_rate=.00001, change_threshold_std=5) -> None:
        super().__init__(k)

        self.k = k
        self.confluence_metric = confluence_metric
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Parameters for tracking change in 
        self.change_threshold_std = change_threshold_std

        self._distance_distr_count = 0
        self._distance_distr_mean = 0
        self._distance_distr_std = 0
        self._distance_distr_variance = 0
        self._distance_distr_stable = False

        self._distance_delta_threshold = 0

        # Stores Cluster Centroids
        self.centroids = None

        # Store the base data
        self._base_mean_distance = None
        self._base_cluster_counts = None

        # Store the batch data
        self._batch_centroids = None # Stores the mean data point of the batch for each cluster
        self._batch_distance_sums = None # For computing intra-cluster mean pairwaise distance
        self._batch_cluster_counts = None # Stores the count of data points for each cluster 


    def fit(self, X, max_iterations=200):
        labels = super().fit(X, max_iterations)

        # Get the centroids 
        self.centroids = kmeans.centroids
        print('Staring Centroids : \n', self.centroids)

        # Store the base stats
        # Calculate distances of each point to its assigned centroid
        distances = np.linalg.norm(X - self.centroids[labels], axis=1)

        # Calculate the average intra-cluster distances per cluster
        self._base_mean_distance = np.array([np.mean(distances[labels == label]) for label in range(self.k)])
        self._base_cluster_counts = np.bincount(labels)

        return labels


    def predict(self, X, seq_learn=False):
        # Assert that model has been trained
        assert self.centroids is not None, "Model has not been trained!"

        # Calculate the distance between that point and all centroids
        X = np.atleast_2d(X)
        labels = []
        
        for point in X:
            distances = KMeansPlus.distance_euclidean(point, self.centroids)
            label = np.argmin(distances) # Get the index of the minimum distance from a centroid
            
            labels.append(label) # Assign that index/label/cluster to that data point

        # Enable sequential learning from new data
        if seq_learn:
            self.sequential_learn(X, np.array(labels))

        return labels


    def sequential_learn(self, X, labels):
        """
        Perform Sequential Learning Operations
        """
        # For each new data point, add them to the current batch sum for its respective cluster
        cluster_ids = range(self.k)

        # 1:  Collect data in batches
        # Initialize the array with the same shape as a centroid
        if self._batch_distance_sums is None:
            self._batch_centroids = np.zeros_like(self.centroids)
            self._batch_distance_sums = np.zeros(self.k)
            self._batch_cluster_counts = np.zeros(self.k)

        # 2. Compute the new data's average intra-cluster distance
        labels_count = np.bincount(labels, minlength=self.k)
        centroid_point_distances = np.linalg.norm(X - self.centroids[labels], axis=1)
        intracluster_distance_sums = np.array([np.sum(centroid_point_distances[labels == label], axis=0) for label in cluster_ids])

        # 3. Compute the centroid data point for each cluster
        # Add new cluster sums to the stored batch cluster sums  
        _cluster_sums = np.array([np.sum(X[labels == label], axis=0) for label in cluster_ids])
        batch_cluster_sums = self._batch_centroids * self._batch_cluster_counts[:, np.newaxis]
        cluster_centroids = batch_cluster_sums + _cluster_sums

        # 4. Update batch with new data
        self._batch_cluster_counts += labels_count # Counts
        self._batch_distance_sums += intracluster_distance_sums
        for label in cluster_ids:
            if self._batch_cluster_counts[label] > 0:
                self._batch_centroids[label] = cluster_centroids[label] / self._batch_cluster_counts[label]


        # print('Tested Batch Centroid : \n' , self._batch_centroids)
        # 5. Evaluate and Update the centroids
        self.evaluate_centroids()


    def update_basedata(self, cluster_label, centroid, count, mean_distance):
        """
        Updates the centroid of a cluster, specified by the cluster labe
        """
        try:
            self.centroids[cluster_label] = centroid
            self._base_cluster_counts[cluster_label] = count
            self._base_mean_distance [cluster_label]= mean_distance
        except Exception as e:
            raise e
        
        # Reset the batch states
        self._batch_centroids[cluster_label] = 0
        self._batch_distance_sums[cluster_label] = 0
        self._batch_cluster_counts[cluster_label] = 0

        return True
    

    def evaluate_centroids(self):
        """
        Evaluate the centroids with the current batch's data. Default trigger is maximum batch size
        """

        # Learning Rate Factors
        alpha = 1 - self.learning_rate
        beta = self.learning_rate

        # 1. Calculate new base data, and the percentage change from current base data 
        new_count = self._base_cluster_counts + self._batch_cluster_counts
        new_count_weighted = (alpha * self._base_cluster_counts) + (beta * self._batch_cluster_counts)

        new_mean_distances = ((alpha * self._base_mean_distance * self._base_cluster_counts) + (beta * self._batch_distance_sums)) / new_count
        distance_delta = new_mean_distances / self._base_mean_distance # 1 - [this] gives the percentage change from the current base data

        # Calculate the threshold change in intra-cluster distance to trigger a centroid update
        previous_delta_threshold = self._distance_delta_threshold

        mean_distance_delta = np.mean(distance_delta)
        self._distance_delta_threshold = self.change_threshold_std * mean_distance_delta

        self._distance_distr_count += 1
        delta = mean_distance_delta - self._distance_distr_mean
        self._distance_distr_mean += delta / self._distance_distr_count
        self._distance_distr_variance += delta * (mean_distance_delta - self._distance_distr_mean)
        self._distance_distr_std = np.sqrt(self._distance_distr_variance / self._distance_distr_count)

        # Update the distance delta threshold
        if not self._distance_distr_std == 0:
            self._distance_delta_threshold = self.change_threshold_std * self._distance_distr_std

            # Check for stability of standard deviation calculation
            # The change in standard deviation should be less than / equal to 1%
            if abs((self._distance_delta_threshold / previous_delta_threshold) - 1) <= 0.01:
                self._distance_distr_stable = True

        new_centroid = (
            (alpha * self.centroids * self._base_cluster_counts[:, None]) +  # Broadcasting
            (beta * self._batch_centroids * self._batch_cluster_counts[:, None])
        ) / new_count_weighted[:, None]  # Broadcasting



        print("Standard Deviation Stability : ", self._distance_distr_stable)

        # Update the base data
        for label in range(self.k):
            cluster_label = label
            centroid = new_centroid[label]

            count = new_count[label]
            mean_distance = new_mean_distances[label]

            dist_change = abs(distance_delta[label] - 1)

            # Check conditions for cluster centroid update
            # Intra-cluster distance increases above a thresholds
            if (self._distance_distr_stable and (dist_change >= self._distance_delta_threshold)):        
                # old_centroid = self.centroids[label]
                self.update_basedata(cluster_label, centroid, count, mean_distance)
                print(f'Centroid Updated at index {label}') # ; from {old_centroid} to {self.centroids[label]}')

        print(f"Centroid : \n{self.centroids}")
        return 


    def _compute_confluence_metric(self):
        """
        Calculate the confluence metric value for each cluster.

        Options :
        INTRA-CLUSTER SIMILARITIES
        - Average pairwise distance within clusters (DTW)
        - Silhouette Coefficient
        - Information-based measures like mutual information
        """

        return
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt  # noqa
    import pandas as pd

    clear_terminal()
    np.random.seed(14)

    data = pd.read_csv("/Users/jerryinyang/Code/seqluster/data.csv")

    X = data[['x', 'y']].to_numpy()
    X = np.random.permutation(X)

    kmeans = KMeansSeq(4, learning_rate=0.001)
    labels = kmeans.fit(X)

    for iter in range(500):
        random_indices = np.random.randint(0, len(X), size=400)
        test_data = X[random_indices]

        print(f"Iteration {iter + 1}-------------------------")
        kmeans.predict(test_data, seq_learn=True)
        print('\n\n')

    # plt.scatter(X[:,0], X[:,1], c=labels)
    # plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='fuchsia', marker='*', s=200)
    # plt.show()
        
