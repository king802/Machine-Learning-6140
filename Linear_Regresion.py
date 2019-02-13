"""
This is a file that contains all functions to perform linear regression on a dataset
"""

import numpy as np
import pandas as pd

from HousingData.Datasets import data_housing_train as house_train
from HousingData.Datasets import data_housing_test as house_test

from spambase.Dataset import data_spam as spam_data

from Evalutaion_Prediction import mean_square_error
from Evalutaion_Prediction import accuracy
from Evalutaion_Prediction import Confusion_Matrix
from Evalutaion_Prediction import avg


def linear_regression(data: pd.DataFrame, labels: pd.DataFrame):
    """
    This is the linear regression function that takes in a dataset and returns the
    :param data: is a pandas DataFrame of the dataset
    :param labels: is the training labels to use to build the prediction model.
    :return: the model that was found by performing linear regression.
    """
    labels = list(labels)

    features = data.values

    features_transposed = features.transpose()

    return np.linalg.inv(features_transposed @ features) @ (features_transposed @ labels)


def predict(data: pd.DataFrame, model: np.matrix):
    """
    This function is used to predict the values of a dataset given the dataset and a prediction model.
    :param data: the data that is going to be predicted
    :param model: the prediction model that is going to be used.
    :return: the predicted labels for the data that was given to the function.
    """
    results = data @ model
    return list(np.round(results.values.T))


def k_fold_linear_regression(data: pd.DataFrame, folds: int = 5) -> (float, float):
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

        model = linear_regression(train, train_labels)
        train_labels = list(train_labels)
        test_results = predict(test, model)
        train_results = predict(train, model)

        Confusion_Matrix(test_results, test_labels)

        acc_test.append(accuracy(test_results, test_labels))
        acc_train.append(accuracy(train_results, train_labels))

    return avg(acc_train), avg(acc_test)


"""
-----------------------------------------------------------------------------------------------------------------------
House Run:
-----------------------------------------------------------------------------------------------------------------------
"""
house_train = house_train
house_train_labels = house_train.pop('Labels')

house_test = house_test
house_test_labels = list(house_test.pop('Labels'))

housing_model = linear_regression(house_train, house_train_labels)

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
spam = spam_data

train_accuracy, test_accuracy = k_fold_linear_regression(spam)

print("////////////////////")
print("Spam Data Results")
print("Train Accuracy:")
print(train_accuracy)
print("Test Accuracy:")
print(test_accuracy)
