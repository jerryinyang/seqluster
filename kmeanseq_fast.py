import numpy as np 
from sklearn.cluster import KMeans # noqa
from dtaidistance import dtw # noqa
from collections import deque # noqa

from utils import clear_terminal, debug# noqa

class SeqKMeans:
    """
    The model uses a kmeans++ clustering model, and sequentially updates the centroid definitions.
    """ 

    def __init__(self, k, confluence_metric=None, batch_size=100, learning_rate=.001, centroid_update_threshold_std=5, random_state=14, verbose=False) -> None:
        # Stores Cluster Centroids
        self.k = k
        self.confluence_metric = confluence_metric
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Store KMeans variables
        self.kmeans = None

        # Parameters for tracking change 
        self.centroid_update_threshold_std = centroid_update_threshold_std

        # Variables for monitoring the confluence metric change distribution
        self._distance_distr_count = 0
        self._distance_distr_mean = 0
        self._distance_distr_std = 0
        self._distance_distr_variance = 0
        self._distance_distr_stable = False
        self._distance_delta_threshold = 0 # Actual threshold value

        # Store the base data
        self._base_mean_distance = None
        self._base_cluster_counts = None

        # Store the batch data
        self._batch_centroids = None # Stores the mean data point of the batch for each cluster
        self._batch_distance_sums = None # For computing intra-cluster mean pairwaise distance
        self._batch_cluster_counts = None # Stores the count of data points for each cluster 

        self._random_state = random_state
        self._verbose = verbose


    def fit(self, X, max_iterations=200):
        kmeans = KMeans(n_clusters=self.n_clusters, 
                        init='k-means++', 
                        n_init=10, 
                        random_state=self._random_state, 
                        max_iter=max_iterations)
        kmeans.fit(X)

        labels = kmeans.labels_

        # Store the centroids 
        self.kmeans = kmeans
        self.centroids = kmeans.cluster_centers_
        self._verbose_output('Staring Centroids : \n', self.centroids)

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
        labels = self.kmeans.predict(X)
        
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
        self._distance_delta_threshold = self.centroid_update_threshold_std * mean_distance_delta

        self._distance_distr_count += 1
        delta = mean_distance_delta - self._distance_distr_mean
        self._distance_distr_mean += delta / self._distance_distr_count
        self._distance_distr_variance += delta * (mean_distance_delta - self._distance_distr_mean)
        self._distance_distr_std = np.sqrt(self._distance_distr_variance / self._distance_distr_count)

        # Update the distance delta threshold
        if not self._distance_distr_std == 0:
            self._distance_delta_threshold = self.centroid_update_threshold_std * self._distance_distr_std

            # Check for staibility of standard deviation calculation
            # The change in standard deviation should be less than / equal to 1%
            if abs((self._distance_delta_threshold / previous_delta_threshold) - 1) <= 0.01:
                self._distance_distr_stable = True

        # Calculate the new centroids
        new_centroid = (
            (alpha * self.centroids * self._base_cluster_counts[:, np.newaxis]) +  # Broadcasting
            (beta * self._batch_centroids * self._batch_cluster_counts[:, np.newaxis])
        ) / new_count_weighted[:, np.newaxis] # Broadcasting
        
        self._verbose_output("Standard Deviation Stability Status: ", self._distance_distr_stable)

        # Update the base data
        for label in range(self.k):
            cluster_label = label
            centroid = new_centroid[label]
            count = new_count[label]
            mean_distance = new_mean_distances[label]

            dist_change = abs(distance_delta[label] - 1)

            # Allow Clusters/Centroid updates
            if not self._distance_distr_stable:
                continue

            # Check conditions for cluster centroid update
            # Intra-cluster distance increases above a thresholds
            if dist_change >= self._distance_delta_threshold:        
                self.update_basedata(cluster_label, centroid, count, mean_distance)
                self._verbose_output(f'Centroid Updated at index {label}')

        self._verbose_output(f"Centroid : \n{self.centroids}")
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
   

    def _verbose_output(self, *args):
        if self._verbose:
            for content in args:
                print(content)


    @property
    def centroids(self):
        return self.kmeans.cluster_centers_
    
    @centroids.setter
    def centroids(self, value):
        self.kmeans.cluster_centers_ = value

    @property
    def n_clusters(self):
        return self.k


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt  # noqa

    clear_terminal()
    np.random.seed(14)

    data = pd.read_csv("/Users/jerryinyang/Code/seqluster/data.csv")

    X = data[['x', 'y']].to_numpy()
    X = np.random.permutation(X)

    kmeans = SeqKMeans(4, learning_rate=0.00000001, verbose=False)
    labels = kmeans.fit(X)

    for iter in range(20):
        random_indices = np.random.randint(0, len(X), size=400)
        test_data = X[random_indices]

        # print(f"Iteration {iter + 1}–––––––––––––––––––––––––––––––––––––––––––––")
        kmeans.predict(test_data, seq_learn=True)
        # print('\n\n')

    print(f'Final Centroids : \n{kmeans.centroids}')

    # plt.scatter(X[:,0], X[:,1], c=labels)
    # plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='fuchsia', marker='*', s=200)
    # plt.show()
        



# if __name__ == "__main__":

    # import pandas as pd
    # import matplotlib.pyplot as plt  # noqa
    
    # np.random.seed(14)

    # # region 1. Prepare the data s
    # prices = pd.read_parquet('prices.parquet')

    # raw_data = prices["close"].dropna(axis=0)
    # raw_data = raw_data.to_numpy()

    # # Generate windows
    # window_size = 5

    # X = []
    
    # for index in range(window_size, len(raw_data)):
    #     start_index = index - window_size
    #     end_index = index

    #     X.append(raw_data[start_index : end_index])

    # # Split X Data
    # split_percent = .7
    # split_index = int(round(len(X) * split_percent))
    
    # X = np.random.permutation(X)
    # X, X_test = np.array(X[:split_index]), np.array(X[split_index:])
    # X_test = np.random.permutation(X_test)

    # # endregion

    # kmeans = SeqKMeans(8)
    # labels = kmeans.fit(X)

    # print(kmeans.predict(X_test, seq_learn=True))

    # # cluster1 = X[labels == 1]

    # # print(cluster1.shape)

    # # plt.plot(cluster1[:10])
    # # plt.show()
        