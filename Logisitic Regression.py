import numpy as np
import pandas as pd
from spambase.Dataset import norm_data
import Evalutaion_Prediction as Tools


def gradient_decent_logistic_regression(GD_data: pd.DataFrame, iteration: int = 100000, learning_rate: float =
.001) -> list:
    """
    This function takes in a data frame and returns a model to use to make predictions
    Args:
        GD_data: This is the entire training dataset with labels, and feature matrix.
        iteration: Number of iterations to perform gradient decent with
        learning_rate: this is the learning rate to use in the gradient decent.

    Returns: a prediction model (list of weights).

    """
    GD_data = GD_data.copy()

    GD_data['Bias'] = 1

    Y = np.matrix(GD_data.pop('Labels'))

    X = GD_data.values

    X_t = X.transpose()

    thetas = [0] * len(GD_data.keys())

    thetas = np.matrix(thetas)

    for i in range(iteration):
        # (1xm) * (m*n) => (1xn)
        z = thetas @ X_t

        # (1xn)
        predictions = 1 / (1 + np.exp(-z))

        # (1xn) -(1xn) => (1xn)
        output_error_signal = Y - predictions

        # (1xm)
        gradient = output_error_signal @ X

        thetas = thetas + learning_rate * gradient

    return thetas.T


def logistic_prediction(data: pd.DataFrame, model: pd.DataFrame):
    results = data @ model

    results = results.values.T.tolist()[0]
    results = map(sigmoid, results)
    results = map(round, results)
    results = list(results)
    return results


def sigmoid(score):
    return 1 / (1 + np.exp(-score))


def k_fold_logistic(data: pd.DataFrame, folds: int = 5) -> (float, float):
    d = data.sample(frac=1)
    segments = np.array_split(d, folds)
    acc_test = []
    err_test = []

    acc_train = []
    err_train = []

    for i in range(folds):
        temp = segments.copy()

        test = temp.pop(i)
        test_labels = list(test['Labels'])
        train = pd.concat(temp)

        model = gradient_decent_logistic_regression(train)
        train_labels = list(train['Labels'])

        test_results = logistic_prediction(test, model)
        train_results = logistic_prediction(train, model)

        acc_test.append(Tools.accuracy(test_results, test_labels))
        err_test.append(Tools.mean_square_error(test_results, test_labels))

        acc_train.append(Tools.accuracy(train_results, train_labels))
        err_train.append(Tools.mean_square_error(train_results, train_labels))

    return Tools.avg(acc_train), Tools.avg(acc_test)


GD_log_Spam: pd.DataFrame = norm_data

GD_log_Spam['bias'] = 1

Training_accuracy, Testing_error = k_fold_logistic(GD_log_Spam)

print("////////////////////")
print("Spam Data Results")
print("Training Accuracy:")
print(Training_accuracy)
print("Testing Accuracy:")
print(Testing_error)
print("---------")
