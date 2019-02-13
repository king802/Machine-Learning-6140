import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    predicted_labels = list(map(custom_round, predicted_labels))

    total_positives = sum(predicted_labels)
    true_positive = []
    for i in range(len(actual_labels)):
        if predicted_labels[i] == actual_labels[i] and actual_labels == 1:
            





    plt.matshow(m, cmap=plt.cm.Blues)

    plt.show()
