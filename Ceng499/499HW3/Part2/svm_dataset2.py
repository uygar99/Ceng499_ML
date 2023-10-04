import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC

dataset, labels = pickle.load(open("../data/part2_dataset2.data", "rb"))

outer_loop_count = 5
split_count = 10

configs = {'C': [0.1, 1, 10, 100],
           'kernel': ['rbf', 'poly', 'linear', 'sigmoid']}

mean_scores = [None, None] * len(configs['C']) * len(configs['kernel'])

i = 0
while i < outer_loop_count:
    folded = RepeatedStratifiedKFold(n_splits=split_count)
    svm = SVC()
    dataset = StandardScaler().fit_transform(dataset, labels)
    configuration = GridSearchCV(svm, configs, cv=folded, scoring="accuracy").fit(dataset, labels)
    j = 0
    while j < len(configs['C']) * len(configs['kernel']):
        if mean_scores[j]:
            mean_scores[j][1].append(configuration.cv_results_["mean_test_score"][j])
        else:
            result = [configuration.cv_results_["params"][j], [configuration.cv_results_["mean_test_score"][j]]]
            mean_scores[j] = result
        j += 1
    i += 1

for i in range(len(configs['C']) * len(configs['kernel'])):
    x = mean_scores[i][1]
    print("C: " + str(mean_scores[i][0]['C']) + " Kernel: " + mean_scores[i][0]['kernel'],
          "Mean: " + str(np.mean(x)) + " Confidence interval: " +
          str([np.mean(x) - (1.95 * np.std(x) / np.sqrt(5)), np.mean(x) + (1.95 * np.std(x) / np.sqrt(5))]))
