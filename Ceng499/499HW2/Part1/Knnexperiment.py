import math
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from Distance import Distance
from Knn import KNN

dataset, labels = pickle.load(open("../data/part1_dataset.data", "rb"))
configurations = [(3, Distance.calculateCosineDistance), (5, Distance.calculateCosineDistance),
                  (10, Distance.calculateCosineDistance), (20, Distance.calculateCosineDistance),
                  (5, Distance.calculateMinkowskiDistance), (20, Distance.calculateMinkowskiDistance),
                  (30, Distance.calculateMinkowskiDistance), (35, Distance.calculateMinkowskiDistance),
                  (5, Distance.calculateMahalanobisDistance), (15, Distance.calculateMahalanobisDistance),
                  (25, Distance.calculateMahalanobisDistance), (3, Distance.calculateMahalanobisDistance)]

k_folded = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
c_acc = list()
iteration_count = 5

for c in configurations:
    i = 0
    iteration = list()

    while i < iteration_count:
        inside_list = list()

        for i_train, i_test in k_folded.split(dataset, labels):
            x_train, x_test = dataset[i_train], dataset[i_test]
            y_train, y_test = labels[i_train], labels[i_test]
            if c[1] == Distance.calculateMinkowskiDistance:
                model = KNN(x_train, y_train, c[1], 2, c[0])
            elif c[1] == Distance.calculateMahalanobisDistance:
                S_minus_1 = np.linalg.inv(np.cov(dataset.T))
                model = KNN(x_train, y_train, c[1], S_minus_1, c[0])
            else:
                model = KNN(x_train, y_train, c[1], None, c[0])
            predicted_list = list()
            for x in x_test:
                predicted = model.predict(x)
                predicted_list.append(predicted)
            accuracy = (np.sum(predicted_list == y_test) / len(y_test)) * 100
            inside_list.append(accuracy)    #  There is a value for each fold (10 folds)

        mean_inside = np.mean(inside_list)  #  Mean value for inside_list
        iteration.append(mean_inside)       #  Mean value appended the other list (5 iteration)
        i = i + 1

    c_acc.append(iteration)                 #  The iteration list appended to config accuracy list (its len is config length)

i = 0
best = 0
best_std = 0
temp_i = 0
for acc in c_acc:
    acc = np.array(acc)
    mean = np.mean(acc)
    if mean > best:
        temp_i = i
        best = mean
        best_std = np.std(acc)
    if abs(best-mean) < 0.001:  #    Just to make sure that we choose the more reliable one when means are equal
        if best_std > np.std(acc):
            temp_i = i
            best = mean
            best_std = np.std(acc)
    print(('Confidence interval %d. configuration: %.3f ' + '+-' + ' %.3f') % (i, np.mean(acc), 1.96 * np.std(acc) / (math.sqrt(len(acc)))))
    i += 1
print('Configuration %d has best mean accuracy & (if equal) standard deviation combination with %.3f ' % (temp_i, best) + '+-' + ' %.3f' % best_std)  #    The best accuracy (if some configs have same then lower standard deviation)
