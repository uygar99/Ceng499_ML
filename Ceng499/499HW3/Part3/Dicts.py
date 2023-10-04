import static as static


class Dicts:
    knn_configurations = {
        'n_neighbors': [3, 5],
        'metric': ['euclidean', 'manhattan'],
    }
    svm_configurations = {
        'C': [0.1, 1],
        'kernel': ['rbf', 'linear'],
    }

    decision_tree_configurations = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5]
    }

    random_forest_configurations = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5]
    }

    results = {
        "hyperparameter_scores": {},
        "mean_test_score": [],
        "f1_score": []
    }
    results = {
        'knn': {
            "hyperparameter_scores": {},
            "mean_test_score": [],
            "f1_score": []
        },
        'svm': {
            "hyperparameter_scores": {},
            "mean_test_score": [],
            "f1_score": []
        },
        'dt': {
            "hyperparameter_scores": {},
            "mean_test_score": [],
            "f1_score": []
        },
        'rf': {
            "hyperparameter_scores": {},
            "mean_test_score": [],
            "f1_score": []
        }
    }
