import numpy
import Distance

class KNN:
    def __init__(self, dataset, data_label, similarity_function, similarity_function_parameters=None, K=1):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics
        :param K: how many neighbors to consider, integer
        """
        self.K = K
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters

    def neighbours(self, instance):
        distances = list()
        i = 0
        for row in self.dataset:
            if self.similarity_function == Distance.Distance.calculateCosineDistance:
                dist = Distance.Distance.calculateCosineDistance(instance, row)
            elif self.similarity_function == Distance.Distance.calculateMinkowskiDistance:
                dist = Distance.Distance.calculateMinkowskiDistance(instance, row, self.similarity_function_parameters)
            elif self.similarity_function == Distance.Distance.calculateMahalanobisDistance:
                dist = Distance.Distance.calculateMahalanobisDistance(instance, row, self.similarity_function_parameters)
            distances.append((dist, self.dataset_label[i]))
            i = i+1
        distances.sort()
        neighbors = list()
        i = 0
        while i < self.K:
            neighbors.append(distances[i])
            i = i+1
        return neighbors      #  Returns the array of K neighbors

    def predict(self, instance):
        neighbour_list = self.neighbours(instance)
        array, count = numpy.unique(neighbour_list, return_counts=True)
        return array[count == count.max()][0]

