# kmeans
import numpy as np

# np.random.seed(14)

class KMeans:
    def __init__(self, k:int) -> None:
        self.k = k
        self.centroids = None

    @staticmethod
    def distance_euclidean(point_a, point_b):
        return np.sqrt(np.sum((point_a - point_b) ** 2, axis=1))

    def fit(self, X, max_iterations=200):
        # Randomly initialize the centroids within the maximum and minimum values of each dimension
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0),
                                          size = (self.k, X.shape[1]))
        
        print("Starting Centroids : \n", self.centroids, '\n')
        
        for _ in range(max_iterations):
            labels = [] 
            previous_centroid = self.centroids.copy()

            # Calculate the distances between each data point and all the centroids
            for point in X:
                distances = KMeans.distance_euclidean(point, self.centroids)
                label = np.argmin(distances) # Get the index of the minimum distance from a centroid
                
                labels.append(label) # Assign that index/label/cluster to that data point

            # For each centroid, calculate the mean point
            labels = np.array(labels)
            
            for label in range(self.k):
                # Get the points assigned to this label
                mask = labels == label

                if not any(mask):
                    continue
                points = X[mask]
                
                # Calculate the mean array
                self.centroids[label] =  np.mean(points, axis=0)

            # Early Break, when the change in distances is less that 1e-4
            if np.max(abs(self.centroids - previous_centroid)) <= 1e-20:
                print(f'Converged after {_ + 1} iterations')
                print("Final Centroids : \n", np.sort(self.centroids, axis=0), '\n')
                
                break
        
        return labels


class KMeansPlus:
    def __init__(self, k:int) -> None:
        self.k = k
        self.centroids = None

    @staticmethod
    def distance_euclidean(point_a, point_b):
        return np.sqrt(np.sum((point_a - point_b) ** 2, axis=1))

    def fit(self, X, max_iterations=200):
        # Randomly initialize the centroids within the maximum and minimum values of each dimension
        # 1. Initialize a random plot
        self.centroids = [np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0))]

        distance_matrix = []
        for point in X:
            # Calculate the distance between the centroid and that point
            distance = KMeans.distance_euclidean(point, self.centroids)
            distance_matrix.append(distance)

        # Normalize the distance matrix values
        distance_matrix = np.log(distance_matrix)
        distance_matrix = distance_matrix / np.sum(distance_matrix)
        distance_matrix = np.reshape(distance_matrix, -1 )

        # Select the remaining centroids
        centroid_index = np.random.choice(
            np.array(range(len(distance_matrix))),
            p=distance_matrix,
            size=self.k - 1
        )
        remaining_centroids = X[centroid_index]

        self.centroids = np.vstack((self.centroids, remaining_centroids))
    
        # print("Starting Centroids : \n", self.centroids, '\n')
        
        for _ in range(max_iterations):
            labels = [] 
            previous_centroid = self.centroids.copy()

            # Calculate the distances between each data point and all the centroids
            for point in X:
                distances = KMeans.distance_euclidean(point, self.centroids)
                label = np.argmin(distances) # Get the index of the minimum distance from a centroid
                
                labels.append(label) # Assign that index/label/cluster to that data point

            # For each centroid, calculate the mean point
            labels = np.array(labels)
            
            for label in range(self.k):
                # Get the points assigned to this label
                mask = labels == label

                if not any(mask):
                    continue
                points = X[mask]
                
                # Calculate the mean array
                self.centroids[label] =  np.mean(points, axis=0)

            # Early Break, when the change in distances is less that 1e-4
            if np.max(abs(self.centroids - previous_centroid)) <= 1e-20:
                print(f'Converged after {_ + 1} iterations')
                print("Final Centroids : \n", np.sort(self.centroids, axis=0), '\n')
                
                break
        
        return labels     


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    data = pd.read_csv("/Users/jerryinyang/Code/seqluster/data.csv")

    X = data[['x', 'y']].to_numpy()

    kmeans = KMeansPlus(4)
    
    labels = kmeans.fit(X)

    plt.scatter(X[:,0], X[:,1], c=labels)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='fuchsia', marker='*', s=200)
    plt.show()
