from typing import Literal

import matplotlib.pyplot as plt  # noqa
import numpy as np
from dtaidistance import dtw
from sklearn.cluster import KMeans


class SeqKMeans:
    """
    The model uses a kmeans++ clustering model, and sequentially updates the centroid definitions.
    """ 

    def __init__(self, 
                 n_clusters:int=8, 
                 learning_rate:float=.001, 
                 distance_metric:Literal['euclidean', 'dtw']='euclidean', 
                 centroid_update_threshold_std:float=5, 
                 fit_method:Literal['full', 'sequential']='sequential',
                 training_splits:int=20,
                 random_state:int=14, 
                 verbose:bool=False) -> None:
        
        # Stores Cluster Centroids
        self.__k = n_clusters
        self.__learning_rate = learning_rate
        self.__distance_metric = distance_metric
        self.__fit_method = fit_method
        self.__training_splits = training_splits

        # Store KMeans variables
        self.__kmeans = None

        # Parameters for tracking change 
        self.__centroid_update_threshold_std = centroid_update_threshold_std

        # Variables for monitoring the confluence metric change distribution
        self.__distance_distr_count = 0
        self.__distance_distr_mean = 0
        self.__distance_distr_std = 0
        self.__distance_distr_variance = 0
        self.__distance_distr_stable = False
        self.__distance_delta_threshold = 0 # Actual threshold value

        # Store the base data
        self.__base_mean_distance = None
        self.__base_cluster_counts = None

        # Store the batch data
        self.__batch_centroids = None # Stores the mean data point of the batch for each cluster
        self.__batch_distance_sums = None # For computing intra-cluster mean pairwaise distance
        self.__batch_cluster_counts = None # Stores the count of data points for each cluster 

        self.__random_state = random_state
        self.__verbose = verbose


    def fit(self, X, max_iterations=200):

        # Initialize the kmeans++ model
        kmeans = KMeans(n_clusters=self.n_clusters, 
                        init='k-means++', 
                        n_init='auto', 
                        random_state=self.__random_state, 
                        max_iter=max_iterations)

        # Different methods for fitting the model to training data
        if self.__fit_method == 'full':
            self.labels_ = self._fit_full(kmeans, X)
        
        else:
            self.labels_ = self._fit_sequential(kmeans, X, training_splits=self.__training_splits)

        return self.labels_


    def predict(self, X, seq_learn=True):
        # Assert that model has been trained
        assert self.centroids is not None, "Model has not been trained!"

        # Calculate the distance between that point and all centroids
        X = np.atleast_2d(X)
        labels = self.__kmeans.predict(X)
        
        # Enable sequential learning from new data
        if seq_learn:
            self._sequential_learn(X, np.array(labels))

        return labels


    def _fit_full(self, kmeans, X):
        """
        Fits the KMeans model with all the training data at once.
        """
        # Fit the model
        kmeans.fit(X)
        labels = kmeans.labels_

        # Store the model, centroids 
        self.__kmeans = kmeans
        self.centroids = kmeans.cluster_centers_
        self.__verbose_output('Staring Centroids : \n', self.centroids)

        # Store the base stats
        # Calculate distances of each point to its assigned centroid
        distances = np.linalg.norm(X - self.centroids[labels], axis=1)

        # Calculate the average intra-cluster distances per cluster
        self.__base_mean_distance = np.array([np.mean(distances[labels == label]) for label in range(self.__k)])
        self.__base_cluster_counts = np.bincount(labels)

        return labels
    

    def _fit_sequential(self, kmeans, X, training_splits=100):
        """
        Sequentially fit the KMeans model to the training data.
        """
        
        data = np.array(X)

        # 1. Split the data into init and sequential train batches
        init_split_index = int(round(.4 * len(data)))
        data_train_init = data[:init_split_index]
        data_train_batch = data[init_split_index:]

        # Shuffle the init train data, to initialize centroids
        # data_train_init = np.random.permutation(data_train_init)

        # 3. Fit the model with the init split data
        kmeans.fit(data_train_init)
        labels = kmeans.labels_

        # Initialize the base data
        # Store the centroids 
        self.__kmeans = kmeans
        self.centroids = kmeans.cluster_centers_
        self.__verbose_output('Staring Centroids : \n', self.centroids)

        # Store the base stats
        # Calculate distances of each point to its assigned centroid
        distances = np.linalg.norm(data_train_init - self.centroids[labels], axis=1)

        # Calculate the average intra-cluster distances per cluster
        self.__base_mean_distance = np.array([np.mean(distances[labels == label]) for label in range(self.__k)])
        self.__base_cluster_counts = np.bincount(labels)

        # 4. Sequentially train the model with the training batch 
        batches = np.array_split(data_train_batch, training_splits)

        for _data in batches:
            self.predict(_data, seq_learn=True)

        # Force Centroid updates with remaining data and clear up batch stores 
        self.__evaluate_centroids(force_update=True)

        # Generate labels for for the entire training data
        labels = self.predict(X)
        
        return labels


    def _sequential_learn(self, X, labels):
        """
        Perform Sequential Learning Operations
        """
        # For each new data point, add them to the current batch sum for its respective cluster
        cluster_ids = range(self.__k)

        # 1:  Collect data in batches
        # Initialize the array with the same shape as a centroid
        if self.__batch_distance_sums is None:
            self.__batch_centroids = np.zeros_like(self.centroids)
            self.__batch_distance_sums = np.zeros(self.__k)
            self.__batch_cluster_counts = np.zeros(self.__k)

        # 2. Compute the new data's average intra-cluster distance
        labels_count = np.bincount(labels, minlength=self.__k)
        centroid_point_distances = self.__compute_distances(X, labels)
        intracluster_distance_sums = np.array([np.sum(centroid_point_distances[labels == label], axis=0) for label in cluster_ids])

        # 3. Compute the centroid data point for each cluster
        # Add new cluster sums to the stored batch cluster sums  
        _cluster_sums = np.array([np.sum(X[labels == label], axis=0) for label in cluster_ids])
        batch_cluster_sums = self.__batch_centroids * self.__batch_cluster_counts[:, np.newaxis]
        cluster_centroids = batch_cluster_sums + _cluster_sums

        # 4. Update batch with new data
        self.__batch_cluster_counts += labels_count # Counts
        self.__batch_distance_sums += intracluster_distance_sums
        for label in cluster_ids:
            if self.__batch_cluster_counts[label] > 0:
                self.__batch_centroids[label] = cluster_centroids[label] / self.__batch_cluster_counts[label]

        # print('Tested Batch Centroid : \n' , self._batch_centroids)
        # 5. Evaluate and Update the centroids
        self.__evaluate_centroids()


    def __update_basedata(self, cluster_label, centroid, count, mean_distance):
        """
        Updates the centroid of a cluster, specified by the cluster labe
        """
        try:
            self.centroids[cluster_label] = centroid
            self.__base_cluster_counts[cluster_label] = count
            self.__base_mean_distance [cluster_label]= mean_distance
        except Exception as e:
            raise e
        
        # Reset the batch states
        self.__batch_centroids[cluster_label] = 0
        self.__batch_distance_sums[cluster_label] = 0
        self.__batch_cluster_counts[cluster_label] = 0

        return True
    

    def __evaluate_centroids(self, force_update:bool=False):
        """
        Evaluate the centroids with the current batch's data. Default trigger is maximum batch size
        """

        # Learning Rate Factors
        alpha = 1 - self.__learning_rate
        beta = self.__learning_rate

        # 1. Calculate new base data, and the percentage change from current base data 
        new_count = self.__base_cluster_counts + self.__batch_cluster_counts
        new_count_weighted = (alpha * self.__base_cluster_counts) + (beta * self.__batch_cluster_counts)

        new_mean_distances = ((alpha * self.__base_mean_distance * self.__base_cluster_counts) + (beta * self.__batch_distance_sums)) / new_count
        distance_delta = new_mean_distances / self.__base_mean_distance # 1 - [this] gives the percentage change from the current base data

        # Calculate the threshold change in intra-cluster distance to trigger a centroid update
        previous_delta_threshold = self.__distance_delta_threshold

        mean_distance_delta = np.mean(distance_delta)
        self.__distance_delta_threshold = self.__centroid_update_threshold_std * mean_distance_delta

        self.__distance_distr_count += 1
        delta = mean_distance_delta - self.__distance_distr_mean
        self.__distance_distr_mean += delta / self.__distance_distr_count
        self.__distance_distr_variance += delta * (mean_distance_delta - self.__distance_distr_mean)
        self.__distance_distr_std = np.sqrt(self.__distance_distr_variance / self.__distance_distr_count)

        # Update the distance delta threshold
        if not self.__distance_distr_std == 0:
            self.__distance_delta_threshold = self.__centroid_update_threshold_std * self.__distance_distr_std

            # Check for staibility of standard deviation calculation
            # The change in standard deviation should be less than / equal to 1%
            if abs((self.__distance_delta_threshold / previous_delta_threshold) - 1) <= 0.01:
                self.__distance_distr_stable = True

        # Calculate the new centroids
        new_centroid = (
            (alpha * self.centroids * self.__base_cluster_counts[:, np.newaxis]) +  # Broadcasting
            (beta * self.__batch_centroids * self.__batch_cluster_counts[:, np.newaxis])
        ) / new_count_weighted[:, np.newaxis] # Broadcasting
        
        self.__verbose_output("Standard Deviation Stability Status: ", self.__distance_distr_stable)

        # Update the base data
        for label in range(self.__k):
            cluster_label = label
            centroid = new_centroid[label]
            count = new_count[label]
            mean_distance = new_mean_distances[label]

            dist_change = abs(distance_delta[label] - 1)

            # Allow Clusters/Centroid updates
            # Check conditions for cluster centroid update
            # Intra-cluster distance increases above a thresholds
            # Force Cluster Update : Update all cluster centers and reset the sequential learning batch data
            if force_update or \
                (self.__distance_distr_stable and (dist_change >= self.__distance_delta_threshold)):        
                self.__update_basedata(cluster_label, centroid, count, mean_distance)
                self.__verbose_output(f'Centroid Updated at index {label}')
                
        self.__verbose_output(f"Centroid : \n{self.centroids}")
        return         


    def __compute_distances(self, X, labels):

        if self.__distance_metric == "euclidean":
            return np.linalg.norm(X - self.centroids[labels], axis=1)
        
        elif self.__distance_metric == "dtw":
            distances = []
            
            for index in range(len(X)):
                label = labels[index]
                distances.append((dtw.distance_fast(X[index], self.centroids[label])))

            return np.array(distances)

        return None


    def __verbose_output(self, *args):
        if self.__verbose:
            for content in args:
                print(content)


    def __sort_centroids(self, centroids):
        return np.sort(centroids, axis=0)

    @property
    def cluster_centers_(self):
        return self.__kmeans.cluster_centers_
    
    
    @cluster_centers_.setter
    def cluster_centers_(self, value):
        self.__kmeans.cluster_centers_ = value

    @property
    def centroids(self):
        return self.__kmeans.cluster_centers_
    

    @centroids.setter
    def centroids(self, value):
        self.__kmeans.cluster_centers_ = value


    @property
    def n_clusters(self):
        return self.__k


    @property
    def labels_(self):
        return self.__kmeans.labels_
    

    @labels_.setter
    def labels_(self, value):
        self.__kmeans.labels_ = value



if __name__ == "__main__":
    import pandas as pd
    
    np.random.seed(14)

    data = pd.read_csv("/Users/jerryinyang/Code/seqluster/data.csv")

    X = data[['x', 'y']].to_numpy()
    X = np.random.permutation(X)

    kmeans = SeqKMeans(4, 
                       learning_rate=0.01, 
                       centroid_update_threshold_std=3, 
                       fit_method='full', 
                       verbose=False)
    labels_seq = kmeans.fit(X)

    # Get all 1 data
    # cluster_1_seq = X[labels_norm == 1]
    # plt.scatter(X[:,0], X[:,1], c=labels_seq)
    # plt.show()


    # for iter in range(1000):
    #     random_indices = np.random.randint(0, len(X), size=400)
    #     test_data = X[random_indices]

    #     # print(f"Iteration {iter + 1}")
    #     kmeans.predict(test_data, seq_learn=True)
    #     # print('\n\n')

    # print(f'Final Centroids : \n{kmeans.centroids}')

    # plt.scatter(X[:,0], X[:,1], c=labels)
    # plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='fuchsia', marker='*', s=200)
    # plt.show()
        