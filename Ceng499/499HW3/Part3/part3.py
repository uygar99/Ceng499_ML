import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, cross_val_score,cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from DataLoader import DataLoader
from joblib import parallel_backend


dataset, labels = DataLoader.load_credit_with_onehot("../data/credit.data")
configs = [
    {"kneighborsclassifier__metric": ["euclidean", "manhattan"],
     "kneighborsclassifier__n_neighbors": [4, 6]},

    {"svc__C": [0.1, 1],
     "svc__kernel": ["poly", "rbf", "linear"]},

    {"decisiontreeclassifier__max_depth": [5, 7],
     "decisiontreeclassifier__criterion": ["gini", "entropy"]},


    {"randomforestclassifier__n_estimators": [7, 15],
     "randomforestclassifier__max_depth": [3, 5]}
]

knn_list = []
svm_list = []
dec_tree_list = []
random_forest_list = []

knn_f1 = []
svm_f1 = []
dec_tree_f1 = []
random_forest_f1 = []

# knn_best_params = {}
# svm_best_params = None
# dec_tree_best_params = None
# random_forest_best_params = None

with parallel_backend('threading', n_jobs=5):
    outer_cross_validation = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=np.random.randint(1, 1000))
    inner_cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=np.random.randint(1, 1000))

    knn_pipeline = make_pipeline(MinMaxScaler(), KNeighborsClassifier())
    knn = GridSearchCV(knn_pipeline, configs[0], scoring="accuracy", cv=inner_cross_validation)
    knn_val = cross_validate(knn, dataset, labels, cv=outer_cross_validation, scoring=['f1', 'accuracy'], return_estimator=True)
    knn_list.append(knn_val['test_accuracy'])
    knn_f1.append(knn_val['test_f1']),
    knn_best_params = knn_val['estimator'][0].best_params_
    # if knn_val['test_accuracy'] > knn_max['test_accuracy']:
    #     knn_max = knn_val
    #print("Confidence interval of KNN: ",)

    svm_pipeline = make_pipeline(MinMaxScaler(), SVC())
    svm = GridSearchCV(svm_pipeline, configs[1], scoring="accuracy", cv=inner_cross_validation)
    svm_val = cross_validate(svm, dataset, labels, cv=outer_cross_validation, scoring=['f1', 'accuracy'], return_estimator=True)
    svm_list.append(svm_val['test_accuracy'])
    svm_f1.append(svm_val['test_f1'])
    svm_best_params = svm_val['estimator'][0].best_params_

    dec_tree_pipeline = make_pipeline(MinMaxScaler(), DecisionTreeClassifier())
    dec_tree = GridSearchCV(dec_tree_pipeline, configs[2], scoring="accuracy", cv=inner_cross_validation)
    dec_tree_val = cross_validate(dec_tree, dataset, labels, cv=outer_cross_validation, scoring=['f1', 'accuracy'], return_estimator=True)
    dec_tree_list.append(dec_tree_val['test_accuracy'])
    dec_tree_f1.append(dec_tree_val['test_f1'])
    dec_tree_best_params = dec_tree_val['estimator'][0].best_params_

    average_test_accuracy_rf = []
    average_f1_rf = []
    best = []
    i = 0
    while i < 5:
        random_forest_pipeline = make_pipeline(MinMaxScaler(), RandomForestClassifier())
        random_forest = GridSearchCV(random_forest_pipeline, configs[3], scoring="accuracy", cv=inner_cross_validation)
        random_forest_val = cross_validate(random_forest, dataset, labels, cv=outer_cross_validation, scoring=['f1', 'accuracy'], return_estimator=True)
        random_forest_best_params = random_forest_val['estimator'][0].best_params_
        best.append(random_forest_best_params)
        if i == 0:
            average_test_accuracy_rf = random_forest_val['test_accuracy']
            average_f1_rf = random_forest_val['test_f1']
        else:
            average_test_accuracy_rf = np.add(average_test_accuracy_rf, random_forest_val['test_accuracy'])
            average_f1_rf = np.add(average_f1_rf, random_forest_val['test_f1'])

        #print(random_forest_val['test_accuracy'])
        i += 1
        if i == 5:
            average_test_accuracy_rf = np.divide(average_test_accuracy_rf, 5)
            average_f1_rf = np.divide(average_f1_rf, 5)
            random_forest_f1.append(average_f1_rf)
            random_forest_list.append(average_test_accuracy_rf)
        # average_test_accuracy_rf += random_forest_val['test_accuracy']
        # average_f1_rf += random_forest_val['test_f1']
    #rf_accuracy = np.mean(random_forest_list)

    knn_mean = np.mean(knn_list)
    knn_std = (1.95 * np.std(knn_list)) / np.sqrt(5)
    knn_mean_of_F1 = np.mean(knn_f1)
    knn_std_of_F1 = (1.95 * np.std(knn_f1)) / np.sqrt(5)
    print("KNN: Accuracy score: ", knn_mean,
          "Confidence interval: [", knn_mean - knn_std, ",", knn_mean + knn_std, "]",
          "F1 score:", np.mean(knn_f1),
          "Interval of F1: [", knn_mean_of_F1 - knn_std_of_F1, ",", knn_mean_of_F1 + knn_std_of_F1, "]",)
    print("Best Parameters: ", knn_best_params)

    svm_mean = np.mean(svm_list)
    svm_std = (1.95 * np.std(svm_list)) / np.sqrt(5)
    svm_mean_of_F1 = np.mean(svm_f1)
    svm_std_of_F1 = (1.95 * np.std(svm_list)) / np.sqrt(5)
    print("SVM: Accuracy score: ", svm_mean,
          "Confidence interval: [", svm_mean - svm_std, ",", svm_mean + svm_std, "]",
          "F1 score:", np.mean(svm_f1),
          "Interval of F1: [", svm_mean_of_F1 - svm_std_of_F1, ",", svm_mean_of_F1 + svm_std_of_F1, "]")
    print("Best Parameters: ", svm_best_params)

    dec_tree_mean = np.mean(dec_tree_list)
    dec_tree_std = (1.95 * np.std(dec_tree_list)) / np.sqrt(5)
    dec_tree_mean_of_F1 = np.mean(dec_tree_f1)
    dec_tree_std_of_F1 = (1.95 * np.std(dec_tree_f1)) / np.sqrt(5)
    print("Decision Tree: Accuracy score: ", dec_tree_mean,
          "Confidence interval: [", dec_tree_mean - dec_tree_std, ",", dec_tree_mean + dec_tree_std, "]",
          "F1 score:", np.mean(dec_tree_f1),
          "Interval of F1: [", dec_tree_mean_of_F1 - dec_tree_std_of_F1, ",", dec_tree_mean_of_F1 + dec_tree_std_of_F1, "]")
    print("Best Parameters: ", dec_tree_best_params)

    # values, counts = np.unique(best, return_counts=True)
    # ind = np.argmax(counts)
    random_forest_mean = np.mean(random_forest_list)
    random_forest_std = (1.95 * np.std(random_forest_list)) / np.sqrt(5)
    random_forest_mean_of_F1 = np.mean(random_forest_f1)
    random_forest_std_of_F1 = (1.95 * np.std(random_forest_f1)) / np.sqrt(5)
    print("Random Forest: Accuracy score: ", random_forest_mean,
          "Confidence interval: [", random_forest_mean - random_forest_std, ",", random_forest_mean + random_forest_std, "]",
          "F1 score:", np.mean(random_forest_f1),
          "Interval of F1: [", random_forest_mean_of_F1 - random_forest_std_of_F1, ",", random_forest_mean_of_F1 + random_forest_std_of_F1, "]",)
    print("Best Parameters: ", best)
