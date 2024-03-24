import numpy as np 
from sklearn.cluster import KMeans # noqa
from dtaidistance import dtw # noqa
from collections import deque # noqa


class SeqKMeans: 

    def __init__(self, n_clusters, random_state=14) -> None:
        self.n_clusters = n_clusters

        self._kmeans = None
        self._random_state = random_state
        self._model_fitted = False

    def fit(self, X, max_iterations=300):
        kmeans = KMeans(self.n_clusters, init='k-means++', n_init='auto', random_state=self._random_state)
        kmeans.fit(X)

        # Store the kmeans model
        self._kmeans = kmeans

        self._model_fitted = True
        return kmeans.labels_
    
    def predict(self, X):
        return self._kmeans.predict(X)


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
