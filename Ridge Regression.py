"""
Contains the functionality to perform ridge regression on the spam and housing data.
"""

import HousingData.Datasets as house
import spambase.Dataset as spam

import Evalutaion_Prediction as tools

import pandas as pd
import numpy as np


def ridge_regression(data: pd.DataFrame, lam: int = 0):
    """
    This function returns the model for a ridge regression
    :param data: this is the training data including the labels, features and bias feature
    :param lam: is the hyper-parameter that is used to offset weights. It should be between 0-inf
    :return: a model in the form of a list that is weights that corresponds to the different features for regression
    prediction.
    """
    data = data.copy()
    # Labels of data set
    y = list(data.pop('Labels'))

    # Feature Matrix
    X = data.values

    # Transpose of Feature Matrix
    X_t = np.transpose(X)

    # Identity Matrix
    I = np.identity(len(X_t))

    # ((Z_t * Z) + (lam * I))^(-1) * (Z_t * y)
    return np.linalg.inv((X_t @ X) + (lam * I)) @ (X_t @ y)


"""
-----------------------------------------------------------------------------------------------------------------------
House Run:
-----------------------------------------------------------------------------------------------------------------------
"""
house_test = house.data_housing_test
house_train = house.data_housing_train

model = ridge_regression(house_train)

house_test_labels = list(house_test.pop('Labels'))
house_train_labels = list(house_train.pop('Labels'))

test_predictions = tools.predict_linear_regression(house_test, model)

train_predictions = tools.predict_linear_regression(house_train, model)

print("------------------")
print("Housing Linear Regression")
print("Training Error:")
print(tools.mean_square_error(train_predictions, house_train_labels))
print("Testing Error:")
print(tools.mean_square_error(test_predictions, house_test_labels))

"""
-----------------------------------------------------------------------------------------------------------------------
Spam Run:
-----------------------------------------------------------------------------------------------------------------------
"""
spam = spam.data_spam

training_accuracy, testing_accuracy = tools.k_fold_linear(spam, ridge_regression)

print("////////////////////")
print("Spam Data Results")
print("Train Accuracy:")
print(training_accuracy)
print("Test Accuracy:")
print(testing_accuracy)
