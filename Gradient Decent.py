"""
Contains the functionality to perform gradient decent on the housing and spam dataset.
"""

from HousingData.Datasets import norm_train, norm_test
from spambase.Dataset import norm_data

from Evalutaion_Prediction import mean_square_error
from Evalutaion_Prediction import accuracy
from Evalutaion_Prediction import avg

from Evalutaion_Prediction import Confusion_Matrix

import pandas as pd
import numpy as np


def gradient_decent_linear_regression(feature_matrix: pd.DataFrame, labels: pd.DataFrame, learning_rate: float = .1,
                                      iterations: int = 10000) -> np.matrix:
    """
    This function will return (mx1) vector of weights that can be applied to a data set to gather a prediction model.
    :param feature_matrix: is the training feature matrix that will be used to develop the prediction model.
    :param labels: is the training labels that will be used to develop the prediction model
    :param learning_rate: This is the learning rate that will be used when calculating the prediction model.
    :param iterations: this is the number of iteration that will be used to find the prediction model.
    :return: a vector of weights that can be used for predictions. The vector shape is (m X 1)
    """
    data = feature_matrix.copy()

    m = len(data)

    Y = labels.values

    X = data.values

    thetas = np.zeros(len(data.keys()))

    for i in range(iterations):
        # (1xn) - ((1xm) * (mxn)) => (1xn) - (1xn) => (1xn)

        # h_w(x) = w*x
        prediction = thetas @ X.T

        # 2*(h_w(x) - y)(x)/m
        gradient = 2 * ((prediction - Y) @ X) / m

        # (1xm) - lambda*(1xm) => (1xm)   : Perfect
        thetas = thetas - learning_rate * gradient
    # (mx1)
    return thetas.T


def predict(data: pd.DataFrame, model: np.matrix):
    """
    This function is used to predict the values of a dataset given the dataset and a prediction model.
    :param data: the data that is going to be predicted
    :param model: the prediction model that is going to be used.
    :return: the predicted labels for the data that was given to the function.
    """
    results = data @ model
    return list(np.round(results.values.T))


def k_fold_GD(data: pd.DataFrame, folds: int = 5) -> (float, float):
    """
    This function performs cross validation on a the learning algorithm liner_regression_gradient_decent
    :param data: the data that is to be used to test the algorithm
    :param folds: The number of test to perform on the algorithm
    :return: the average accuracy of the training and testing results.
    """
    d = data.sample(frac=1)
    segments = np.array_split(d, folds)

    acc_test = []
    acc_train = []
    for i in range(folds):
        temp = segments.copy()

        test = temp.pop(i)
        test_labels = list(test['Labels'])
        test = test.drop(['Labels'], axis=1)

        train = pd.concat(temp)
        train_labels = train['Labels']
        train = train.drop(['Labels'], axis=1)

        model = gradient_decent_linear_regression(train, train_labels)
        train_labels = list(train_labels)
        test_results = predict(test, model)
        train_results = predict(train, model)

        # Confusion_Matrix(test_results, test_labels)

        acc_test.append(accuracy(test_results, test_labels))
        acc_train.append(accuracy(train_results, train_labels))

    return avg(acc_train), avg(acc_test)


"""
-----------------------------------------------------------------------------------------------------------------------
House Run:
-----------------------------------------------------------------------------------------------------------------------
"""
house_train = norm_train
house_train_labels = house_train.pop('Labels')

house_test = norm_test
house_test_labels = list(house_test.pop('Labels'))

housing_model = gradient_decent_linear_regression(house_train, house_train_labels)

Test_predictions = predict(house_test, housing_model)
Train_predictions = predict(house_train, housing_model)

print("Housing Linear Regression")
print("Train Error")
print(mean_square_error(Train_predictions, house_train_labels))

print("Test Error:")
print(mean_square_error(Test_predictions, house_test_labels))

"""
-----------------------------------------------------------------------------------------------------------------------
Spam Run:
-----------------------------------------------------------------------------------------------------------------------
"""
spam = norm_data

train_accuracy, test_accuracy = k_fold_GD(spam)

print("////////////////////")
print("Spam Data Results")
print("Train Accuracy:")
print(train_accuracy)
print("Test Accuracy:")
print(test_accuracy)
