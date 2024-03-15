import numpy as np
from kmeans import KMeansPlus
from dtaidistance import dtw


class KMeansSeq(KMeansPlus): 
    """
    The model uses a kmeans++ clustering model, and sequentially updates the centroid definitions.
    """ 

    def __init___(self, k, confluence_metric=None, learning_rate=None, batch_size=None) -> None:
        super().__init__(k)

        self.k = k
        self.confluence_metric = confluence_metric
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.centroids = None

        # Store the data for the current batch
        self._data = None

        self._current_batch_counts = None
        self._current_batch_sums = None


    def fit(self, X, max_iterations=200):
        _ = super().fit(X, max_iterations)

        # Get the centroids 
        self.centroids = kmeans.centroids


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
            pass

        return labels


    def update_centroid(self):
        return


    def evaluate_centroids(self):
        """
        Evaluate the centroids with the current batch's data
        """

        # For each centroid
        for cluster in range(self.k):
            # Compute the new centroid
            new_data_sum = self._current_batch_sums

            # Evaluate the difference in the confluence metric, if it exceeds the threshold
        
        
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

    data = pd.read_csv("/Users/jerryinyang/Code/seqluster/data.csv")

    X = data[['x', 'y']].to_numpy()

    kmeans = KMeansSeq(4)
    
    labels = kmeans.fit(X)
    print(kmeans.centroids)

    # plt.scatter(X[:,0], X[:,1], c=labels)
    # plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='fuchsia', marker='*', s=200)
    # plt.show()