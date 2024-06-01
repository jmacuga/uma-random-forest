import numpy as np
from itertools import product
from .types import Classifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import pandas as pd

def grid_search(
    param_grid: dict,
    model_class: Classifier,
    X_train: np.array,
    X_test: np.array,
    y_train: np.array,
    y_test: np.array,
    num_calls: int,
    path: str = None,
) -> dict:
    """
    Perform grid search on a random forest model.

    Parameters
    ----------
    param_grid : dict
        Dictionary with hyperparameters to search.
    model : RandomForestClassifier
        Random forest model to use.
    X_train : np.array
        Training data.
    X_test : np.array
        Testing data.
    y_train : np.array
        Training labels.
    y_test : np.array
        Testing labels.
    score : callable
        Scoring function.

    Returns
    -------
    dict
        Best hyperparameters found.
    """
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    # Perform grid search
    best_params = None
    best_accuracy = 0
    results = []

    for params in tqdm(param_combinations):
        accuracy_arr, precision_arr, recall_arr, f1_arr = [], [], [], []
        param_dict = dict(zip(param_names, params))
        for i in range(num_calls):
            model = model_class(**param_dict)
            model.fit(X_train, y_train)
            y_pred = np.array([model.predict(x) for x in X_test])

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            accuracy_arr.append(accuracy)
            precision_arr.append(precision)
            recall_arr.append(recall)
            f1_arr.append(f1)

        accuracy = np.mean(accuracy_arr)
        precision = np.mean(precision_arr)
        recall = np.mean(recall_arr)
        f1 = np.mean(f1_arr)

        result = param_dict.copy()
        result["accuracy"] = accuracy
        result["precision"] = precision
        result["recall"] = recall
        result["f1"] = f1
        results.append(result)
        if path is not None:
            save_results(results, path)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = param_dict
    return best_params, best_accuracy, results


def save_results(results: dict, path: str):
    """
    Append results to a CSV file.

    Parameters
    ----------
    results : dict
        Results dictionary.
    path : str
        Path to save the CSV file.
    """
    df = pd.DataFrame(results)
    df.to_csv(path, mode='a', index=False)
    return df
