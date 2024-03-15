import numpy as np
from kmeans import KMeansPlus
from dtaidistance import dtw

class KMeansSeq(KMeansPlus): 
    """
    The model uses a kmeans++ clustering model, and sequentially updates the centroid definitions.
    """ 

    def __init__(self, k, confluence_metric=None, learning_rate=None, batch_size=None) -> None:
        super().__init__(k)

        self.k = k
        self.confluence_metric = confluence_metric
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.centroids = None

        # Store the data for the current batch
        self._data = None

        # Store the base metrics
        self._base_mean_distance = None
        self._base_cluster_counts = None

        self._batch_distance_sums = None
        self._batch_point_counts = None


    def fit(self, X, max_iterations=200):
        labels = super().fit(X, max_iterations)

        # Get the centroids 
        self.centroids = kmeans.centroids

        # Store the base stats
        # Calculate distances of each point to its assigned centroid
        distances = np.linalg.norm(X - self.centroids[labels], axis=1)

        # Calculate the average intra-cluster distances per cluster
        self._base_mean_distance = np.array([np.mean(distances[labels == label]) for label in range(self.k)])
        self._base_cluster_counts = np.bincount(labels)


    def predict(self, X, seq_learn=False):
        # Calculate the distance between that point and all centroids
        X = np.atleast_2d(X)
        labels = []
        
        for point in X:
            distances = KMeansPlus.distance_euclidean(point, self.centroids)
            label = np.argmin(distances) # Get the index of the minimum distance from a centroid
            
            labels.append(label) # Assign that index/label/cluster to that data point

        # Enable sequential learning from new data
        if seq_learn:
            # 1:  Collect data in batches
            # Initialize the array with the same shape as a centroid
            if self._batch_distance_sums is None:
                self._batch_distance_sums = np.zeros(self.k)
                self._batch_point_counts = np.zeros(self.k)

            # For each new data point, add them to the current batch sum for its respective cluster
            labels = np.array(labels)
            distances = np.linalg.norm(X - self.centroids[labels], axis=1)
            sums = np.array([np.sum(distances[labels == label]) for label in range(self.k)])
            
            print(sums)
            exit()
            # for label in np.unique(labels):

            #     # Select only data points of that label
            #     mask = labels == label
            #     points = X[mask]

            #     summed = np.sum(points, axis=0)
            #     summed = summed.T.reshape(-1)

            #     # Update the sums
            #     self._batch_sums[label] += summed
            #     self._batch_counts[label] += len(points)

            #     self.evaluate_centroids(label)

        return labels
    

    def update_centroid(self, centroid, sum, count):
        """
        Computes the adjusted centroid values, based on recent batch data
        """
        return ((1 - self.learning_rate) * centroid) + (self.learning_rate * (sum/ count))


    def evaluate_centroids(self, label):
        """
        Evaluate the centroids with the current batch's data
        """

        # Compute the new centroid
        centroid = self.centroids[label]
        sum = self._batch_sums[label]
        count = self._batch_counts[label]
        
        # Evaluate the difference in the confluence metric, if it exceeds the threshold
        new_centroid = self.update_centroid(centroid, sum, count)

        # For threshold centroid difference, compare current centroid with the calculated centroid
        pass
        
    
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
    import pandas as pd
    import matplotlib.pyplot as plt  # noqa

    np.random.seed(14)

    data = pd.read_csv("/Users/jerryinyang/Code/seqluster/data.csv")

    X = data[['x', 'y']].to_numpy()
    X = np.random.permutation(X)

    kmeans = KMeansSeq(4)
    
    labels = kmeans.fit(X)

    kmeans.predict(X[-50:], seq_learn=True)
    # print(kmeans._current_batch_sums)
    # print(kmeans._current_batch_counts)

    # plt.scatter(X[:,0], X[:,1], c=labels)
    # plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='fuchsia', marker='*', s=200)
    # plt.show()