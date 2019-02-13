"""
This Python file contains Prediction and evaluation functions that are used to calculate and evaluate how accurate a
prediction model is.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def predict_dataset(row, model):
    """
    This allows for the final decision tree to be utilized by the data
    :param row: testing Data
    :param model: the tree that is being used
    :return: the predicted value from the tree
    """
    feature, threshold, left, right = model
    if row[feature] >= threshold:
        if type(left) is tuple:
            return predict_dataset(row, left)
        else:
            return left
    else:
        if type(right) is tuple:
            return predict_dataset(row, right)
        else:
            return right


def predict_data(data: pd.DataFrame, model: list):
    """
    Takes in a testing data set and a model and produces the models predictions
    :param data: the testing data (features only)
    :param model: the ML model that is being used to predict
    :return: a list of the predictions for each data collection in the data set
    """
    prediction = []
    for i, row in data.iterrows():
        prediction.append(predict_dataset(row, model))
    return prediction


def predict_linear_regression(data: pd.DataFrame, model: list):
    """
    This helper function is used to fit the model on to the test data for linear regression.
    :param data: Test data (with out the labels) that is to be fitted the model
    :param model: the model that was returned from the training data
    :return: a vector of predicted labels to then be used.
    """
    prediction = []
    for i, row in data.iterrows():
        value = 0
        j = 0
        for col in data.keys():
            value += row[col] * model[j]
            j += 1
        prediction.append((round(value)))
    return prediction


def mean_square_error(predictions: list, labels: list):
    """
    Calculates the mean square error between the predicted labels and the actual labels
    :param predictions: is the list of predicted data labels
    :param labels: is the list of actual data labels
    :return: the mean square error of prediction model
    """
    temp = 0
    for i in range(len(predictions)):
        temp += (labels[i] - predictions[i]) ** 2

    return round(temp / len(predictions), 2)


def accuracy(predictions, labels):
    """
    Calculates the accuracy of 2 list and how they compare to each other
    :param predictions: the list of predicted values
    :param labels: the list of actual values
    :return: the accuracy of the predictions
    """
    predictions = list(predictions)
    labels = list(labels)
    count = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            count += 1

    return count / len(labels)


def avg(lst: list):
    """
    Returns the average of the given list
    :param lst: the list of values that the average is wanted for.
    :return: the average value of the list
    """
    return sum(lst) / len(lst)


def k_fold_tree(data: pd.DataFrame, algorithm, folds: int = 5) -> (float, float):
    """
    This will return the accuracy and error for a given dataset,model with a designated amount of folds
    :param data: is the data to test the algorithm with
    :param algorithm: is the algorithm that is being evaluated
    :param folds: defaults to 5 and is the amount of splits to test the alo out with.
    :return: the average accuracy and error of the model that was made from the algorithm.
    """
    d = data.sample(frac=1)
    segments = np.array_split(d, folds)
    acc_test = []
    acc_train = []
    for i in range(folds):
        temp = segments.copy()
        test = temp.pop(i)

        test_labels = list(test['Labels'])

        train = pd.concat(temp)

        train_labels = list(train['Labels'])

        model = algorithm(train)

        test_predictions = predict_data(test, model)
        train_predictions = predict_data(train, model)

        acc_test.append(accuracy(test_predictions, test_labels))
        acc_train.append(accuracy(train_predictions, train_labels))

    return avg(acc_train), avg(acc_test)


def k_fold_linear(data: pd.DataFrame, algorithm, folds: int = 5) -> (float, float):
    """
    This will return the accuracy and error for a given dataset,model with a designated amount of folds
    :param data: is the data to test the algorithm with
    :param algorithm: is the algorithm that is being evaluated
    :param folds: defaults to 5 and is the amount of splits to test the alo out with.
    :return: the average accuracy and error of the model that was made from the algorithm.
    """
    d = data.sample(frac=1)
    segments = np.array_split(d, folds)
    acc_test = []

    acc_train = []
    for i in range(folds):
        temp = segments.copy()

        test = temp.pop(i)
        train = pd.concat(temp)
        test_labels = list(test['Labels'])
        train_labels = list(train['Labels'])

        model = algorithm(train)
        test_predictions = [round(x, 1) for x in predict_linear_regression(test.drop(['Labels'], axis=1), model)]
        train_predictions = [round(x, 1) for x in predict_linear_regression(train.drop(['Labels'], axis=1), model)]

        Confusion_Matrix(test_predictions, test_labels)

        acc_test.append(accuracy(test_predictions, test_labels))
        acc_train.append(accuracy(train_predictions, train_labels))

    return avg(acc_train), avg(acc_test)


def evaluate(train: pd.DataFrame, test: pd.DataFrame, algorithm):
    """
    This takes in a training and test set as well as an algorithm and then returns the error and accuracy for that
    algorithm using the given data sets.
    :param train: is the training data set for the algorithm
    :param test: is the data set that will test the model returned by the algorithm
    :param algorithm: is the algorithm that is being used to build the model
    :return: the accuracy and mean  square error of the algorithm
    """

    model = algorithm(train)

    test_labels = test['Labels']

    predictions = predict_data(test, model)

    error = mean_square_error(predictions, test_labels)

    acc = accuracy(predictions, test_labels)

    return acc, error


def custom_round(x):
    """
    This function is used to round the predicted labels to the right classes.
    :param x: is the label that is to be rounded.
    :return: a list of labels that are now correctly rounded.
    """
    if x >= 1:
        x = 1
    else:
        x = 0
    return x


def Confusion_Matrix(predicted_labels: list, actual_labels: list):
    """
    This function will take in a set of predicted Labels as well as the actual labels and return the confusion matrix
    that correlates to that data
    Args:
        predicted_labels: is the labels that were predicted by the model
        actual_labels: is the actual labels that the model is being evaluated with.
    """
    labels = set(actual_labels)

    predicted_labels = list(map(custom_round, predicted_labels))

    matrix = pd.DataFrame(index=labels, columns=labels)

    matrix = matrix.fillna(0)

    for i in range(len(actual_labels)):
        matrix[actual_labels[i]][predicted_labels[i]] += 1
    m = matrix.values

    plt.matshow(m, cmap=plt.cm.Blues)

    for i in range(2):
        for j in range(2):
            c = m[j, i]
            plt.text(i, j, str(c), va='center', ha='center')

    plt.show()
