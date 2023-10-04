import math
from scipy.stats import entropy
import numpy as np


# In the decision tree, non-leaf nodes are going to be represented via TreeNode


class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}


# In the decision tree, leaf nodes are going to be represented via TreeLeafNode
class TreeLeafNode:
    def __init__(self, data, label):
        self.data = data
        self.labels = label


class DecisionTree:
    def __init__(self, dataset: list, labels, features, criterion="information gain"):
        """
        :param dataset: array of data instances, each data instance is represented via an Python array
        :param labels: array of the labels of the data instances
        :param features: the array that stores the name of each feature dimension
        :param criterion: depending on which criterion ("information gain" or "gain ratio") the splits are to be performed
        """
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None

    # further variables and functions can be added...
    def calculate_entropy__(self, dataset, labels):
        """
        :param dataset: array of the data instances
        :param labels: array of the labels of the data instances
        :return: calculated entropy value for the given dataset
        """
        value, counts = np.unique(labels, return_counts=True)
        entropy_value = -np.sum((counts / np.sum(counts)) * np.log2(counts / np.sum(counts)))
        return entropy_value

    def calculate_average_entropy__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an average entropy value is calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute an average entropy value is going to be calculated...
        :return: the calculated average entropy value for the given attribute
        """
        # print(attribute)
        # print("column stack")
        # print(self.data_with_label)
        average_entropy = 0.0
        # av_temp = 0.0
        index = self.features.index(attribute)
        # print(index)
        data_with_label = np.column_stack((dataset, labels))
        dataset_num = np.array(dataset)
        values, counts = np.unique(dataset_num[:, index], return_counts=True)
        effect = counts / np.sum(counts)
        # print(values)
        # i = 0
        for value in values:
            new_index = dataset_num[:, index] == value
            # print(labels)
            # print(data_with_label[new_index][:, -1])
            values_temp, counts_temp = np.unique(data_with_label[new_index][:, -1], return_counts=True)
            average_entropy += effect[np.where(values == value)][0] * np.sum(
                -counts_temp / np.sum(counts_temp) * np.log2(counts_temp / np.sum(counts_temp)))
            # av_temp += effect[i] * np.sum(effect_temp)
            # print(average_entropy)
            # print(av_temp)
            # i += 1
        return average_entropy

    def calculate_information_gain__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an information gain score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the information gain score is going to be calculated...
        :return: the calculated information gain score
        """
        calc_entropy = self.calculate_entropy__(dataset, labels)
        average_entropy = self.calculate_average_entropy__(dataset, labels, attribute)
        information_gain = calc_entropy - average_entropy
        return information_gain

    def calculate_intrinsic_information__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances on which an intrinsic information score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the intrinsic information score is going to be calculated...
        :return: the calculated intrinsic information score
        """
        dataset_num = np.array(dataset)
        index = self.features.index(attribute)
        values, counts = np.unique(dataset_num[:, index], return_counts=True)
        effect = counts / np.sum(counts)
        intrinsic_info = -np.sum(effect * np.log2(effect))
        return intrinsic_info

    def calculate_gain_ratio__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances with which a gain ratio is going to be calculated
        :param labels: array of labels of those instances
        :param attribute: for which attribute the gain ratio score is going to be calculated...
        :return: the calculated gain ratio score
        """
        """
            Your implementation
        """
        info_gain = self.calculate_information_gain__(dataset, labels, attribute)
        intrinsic_info = self.calculate_intrinsic_information__(dataset, labels, attribute)
        gain_ratio = info_gain / intrinsic_info
        return gain_ratio

    def ID3__(self, dataset, labels, used_attributes):
        """
        Recursive function for ID3 algorithm
        :param dataset: data instances falling under the current  tree node
        :param labels: labels of those instances
        :param used_attributes: while recursively constructing the tree, already used labels should be stored in used_attributes
        :return: it returns a created non-leaf node or a created leaf node
        """
        # print(used_attributes)
        if len(np.unique(labels)) == 1:
            return TreeLeafNode("", labels[0])
        if len(labels) == len(used_attributes):
            values, counts = np.unique(labels, return_counts=True)
            print(values[np.argmax(counts)])
            return TreeLeafNode("", values[np.argmax(counts)])
        non_used_features = [x for x in self.features if x not in used_attributes]
        max_gain = 0
        using_attribute = None
        # print(self.criterion)
        index = 0
        # i = 0
        for attribute in non_used_features:
            if self.criterion == "information gain":
                gain = self.calculate_information_gain__(dataset, labels, attribute)
            else:
                gain = self.calculate_gain_ratio__(dataset, labels, attribute)
            if gain > max_gain:
                using_attribute = attribute
                index = self.features.index(using_attribute)
                max_gain = gain
            # i += 1
        # used_attributes.append(using_attribute)
        merged_dataset = np.column_stack((dataset, labels))
        values, counts = np.unique(merged_dataset[:, index], return_counts=True)
        temp_node = TreeNode(using_attribute)
        for value in values:
            new_index = merged_dataset[:, index] == value
            # values_temp, counts_temp = np.unique(merged_dataset[new_index][:, -1], return_counts=True)
            # print(values_temp)
            # values_temp = list(map(int, values_temp))
            # print(values_temp)
            temp_node.subtrees[value] = self.ID3__(merged_dataset[new_index], merged_dataset[new_index][:, -1],
                                                   used_attributes + [using_attribute])

            # leaf = TreeLeafNode(dataset, values_temp[0])
            # temp_node.subtrees[value] = leaf
        return temp_node

    def predict(self, x):
        """
        :param x: a data instance, 1 dimensional Python array
        :return: predicted label of x

        If a leaf node contains multiple labels in it, the majority label should be returned as the predicted label
        """
        predicted_label = None
        # print(x)
        node = self.root
        while True:
            index = self.features.index(node.attribute)
            # print(node.attribute)
            # print(index)
            # print(value)
            node = node.subtrees[x[index]]
            if type(node) == TreeLeafNode:
                break
        predicted_label = node.labels
        # print(predicted_label)
        return int(predicted_label)

    def train(self):
        self.root = self.ID3__(self.dataset, self.labels, [])
        print("Training completed")
