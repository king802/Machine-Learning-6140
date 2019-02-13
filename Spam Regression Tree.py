"""
This File Contains the Regression Tree that is used to predict the Spam Data set
"""

import math
import Evalutaion_Prediction
from spambase.Dataset import data_spam as spam


def threshold(feature_data):
    """
    Takes in a list of the data for a feature and then returns 10 thresholds to try
    :param feature_data: the list of all the data points for a given feature
    :return: a list of 100 thresholds for the dataset.
    """
    maximum = max(feature_data)
    minimum = min(feature_data)
    mode = maximum - minimum
    thresholds = []
    for i in range(1, 100):
        thresholds.append(minimum + (i * mode) / 100)
    return thresholds


def data_splitter(data, feature, thresh):
    """
    Takes in all of the data (in parent node) as well as criteria and then returns the new subsets of info
    :param data: all data including features and labels
    :param feature: the feature that the threshold belongs too
    :param thresh: break for the data.
    :return: two subsections (greater then or equal too, less then)
    """

    sub1 = data[data[feature] >= thresh]
    sub2 = data[data[feature] < thresh]
    return sub1, sub2


def info_gain(current_entropy, labels_subset1, labels_subset2):
    """
    Used to calculate the info_gained by the split
    :param current_entropy: The current entropy of the system
    :param labels_subset1: data that matches the first part of the threshold
    :param labels_subset2: data that doesnt match the threshold
    :return: the info gained by the split type float
    """
    subset1_size = len(labels_subset1)
    subset2_size = len(labels_subset2)
    tot_size = subset1_size + subset2_size
    if tot_size == 0:
        return 0
    return current_entropy - (
            ((subset1_size / tot_size) * entropy(labels_subset1)) + (
            (subset2_size / tot_size) * entropy(labels_subset2)))


def entropy(labels):
    """
    Takes in the a list of binary labels and returns the entropy of the data
    :param labels: a list of labels
    :return: the entropy of the system type float
    """
    total = len(labels)
    trues = sum(labels)
    falses = total - trues
    if falses == 0 and trues == 0:
        return 1
    if falses == 0:
        return (trues / total) * math.log(1 / (trues / total), 2)
    elif trues == 0:
        return (falses / total) * math.log(1 / (falses / total), 2)
    else:
        return (trues / total) * math.log(1 / (trues / total), 2) + (falses / total) * math.log(1 / (falses / total), 2)


def prediction(data):
    """
    Returns what you should predict given the data based off of the number of labels
    :param data: data with labels and features
    :return: Binary True if predict spam or False if predict not spam
    """
    return sum(data['Labels']) / len(data) >= .5


def Regression_Tree_Builder(data, done_list=set()):
    """
    This will take in data that has continuous labels and then build a tree out of the data.
    :param data: the full data matrix with the labels (header is 'Labels')
    :param done_list is a list of features and thresholds that have already been used and can be skipped.
    :return: a Model that can be used in the tree form (feature,threshold,left,right)
    """
    rules_used = done_list

    list_of_features = list(data.keys())

    initial_predictor = prediction(data)

    initial_entropy = entropy(data['Labels'])

    # Stop Criteria 1
    if len(data) < 50 or initial_entropy < .3:
        return int(initial_predictor)

    temp_information_gained = 0

    temp_rule = 0

    for feature in list_of_features:
        if feature is 'Labels':
            continue

        list_of_thresholds = threshold(data[feature])

        for thresh in list_of_thresholds:

            if (feature, thresh) in rules_used:
                # This is a prune feature
                continue

            bigger_data, smaller_data = data_splitter(data, feature, thresh)
            if len(bigger_data) == 0 or len(smaller_data) == 0:
                # This is a prune feature
                rule = (feature, thresh)
                rules_used.add(rule)
                continue
            gain = info_gain(initial_entropy, bigger_data['Labels'], smaller_data['Labels'])
            if temp_information_gained < gain:
                temp_rule = (feature, thresh, bigger_data, smaller_data)

                temp_information_gained = gain

    # Stop Criteria 2
    if temp_information_gained == 0:
        return int(initial_predictor)
    else:
        rule = (temp_rule[0], temp_rule[1])
        rules_used.add(rule)
        return (
            temp_rule[0], temp_rule[1], Regression_Tree_Builder(temp_rule[2], rules_used), Regression_Tree_Builder(
                temp_rule[3], rules_used))


"""
-----------------------------------------------------------------------------------------------------------------------
Spam Run:
-----------------------------------------------------------------------------------------------------------------------
"""
data_spam = spam

train_accuracy, test_accuracy = Evalutaion_Prediction.k_fold_tree(data_spam, Regression_Tree_Builder)

print("////////////////////")
print("Spam Data Results")
print("Train Accuracy:")
print(train_accuracy)
print("Test Accuracy:")
print(test_accuracy)
