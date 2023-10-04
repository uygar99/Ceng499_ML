import numpy as np


class KMeans:
    def __init__(self, dataset, K=2):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        # each cluster is represented with an integer index
        # self.clusters stores the data points of each cluster in a dictionary
        self.clusters = {i: [] for i in range(K)}
        # self.cluster_centers stores the cluster mean vectors for each cluster in a dictionary
        self.cluster_centers = {i: None for i in range(K)}
        # you are free to add further variables and functions to the class
        self.previous_centers = {i: None for i in range(K)}
        self.terminate = False

    def calculateLoss(self):
        """Loss function implementation of Equation 1"""
        result = 0
        for i in self.clusters.keys():
            if self.clusters[i]:
                result += np.sum(np.linalg.norm(np.array(self.clusters[i]) - np.array(self.cluster_centers[i])) ** 2)
        return result

    def initialize_centroids(self):
        temp = self.dataset[np.random.choice(self.dataset.shape[0], self.K)]
        for i in range(self.K):
            self.cluster_centers[i] = temp[i].tolist()
        return

    def closest_centroid(self):
        distances = list()
        for data in self.dataset:
            dist_to_cluster_center = list()
            for cluster_center in self.cluster_centers:
                dist_to_cluster_center.append(np.linalg.norm(data - self.cluster_centers[cluster_center]))
            distances.append(dist_to_cluster_center)
        min_list = list()
        for distance in distances:
            a = np.array(distance).argmin()
            min_list.append(a)
        for i in range(len(self.dataset)):
            self.clusters[min_list[i]].append(self.dataset[i])

        for i in range(self.K):
            mean = np.mean(self.clusters[i], axis=0)
            self.previous_centers[i] = self.cluster_centers[i]
            self.cluster_centers[i] = mean
        return

    def run(self):
        """Kmeans algorithm implementation"""
        self.initialize_centroids()
        while not self.terminate:
            self.closest_centroid()
            self.terminate = True
            for i in range(0, self.K):
                diff = np.abs(np.max(self.cluster_centers[i] - self.previous_centers[i]))
                if diff > 0.02:
                    self.terminate = False
                    break
        return self.cluster_centers, self.clusters, self.calculateLoss()
