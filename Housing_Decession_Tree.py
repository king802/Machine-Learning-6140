"""
This code is the decision tree for the Boston Housing Data
"""

import Evalutaion_Prediction as tools
from HousingData.Datasets import data_housing_train as house_train
from HousingData.Datasets import data_housing_test as house_test


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


def error_of_group(list_of_labels):
    """
    Takes in a list of labels and returns the error
    :param list_of_labels: labels (not binary)
    :return: error (sum of variation)
    """
    mean = sum(list_of_labels) / len(list_of_labels)
    error_of_data = 0
    for label in list_of_labels:
        error_of_data += (label - mean) ** 2
    return error_of_data


def total_error(labels_sub1, labels_sub2):
    """
    Calculate the total_error of the split
    :param labels_sub1: labels of the first half of split
    :param labels_sub2: labels of the second half of the split
    :return: the total error that would be returned by this split
    """
    return error_of_group(labels_sub1) + error_of_group(labels_sub2)


def Decision_Tree_Builder(data, done_list=set()):
    """
    This will take in data that has continuous labels and then build a tree out of the data.
    :param data: the full data matrix with the labels (header is 'Labels')
    :param done_list: this contains a set of all of the actions that have been tried before (feature, threshold). So
    pruning can be done on the tree via this list.
    :return: a Model that can be used in the tree form (feature,threshold,left,right)
    """
    used = done_list

    initial_predictor = sum(data['Labels']) / len(data['Labels'])

    # Stop Criteria
    if len(data) < 3:
        return initial_predictor

    initial_error = error_of_group(data['Labels'])

    temp_error = initial_error

    temp_rule = initial_predictor

    for feature in data.iteritems():
        if feature[0] is 'Labels':
            continue

        for thresh in threshold(feature[1]):
            if (feature[0], thresh) in used:
                continue

            bigger_data, smaller_data = data_splitter(data, feature[0], thresh)

            if len(bigger_data) == 0 or len(smaller_data) == 0:
                used.add((feature[0], thresh))
                continue

            new_error = total_error(bigger_data['Labels'], smaller_data['Labels'])

            if temp_error > new_error:  # minimization of error

                temp_rule = (feature[0], thresh, bigger_data, smaller_data)
                temp_error = new_error

    if initial_error == temp_error:
        return round(initial_predictor, 1)
    else:
        used.add((temp_rule[0], temp_rule[1]))
        return (
            temp_rule[0], temp_rule[1], Decision_Tree_Builder(temp_rule[2], used),
            Decision_Tree_Builder(temp_rule[3], used))


"""
-----------------------------------------------------------------------------------------------------------------------
House Run:
-----------------------------------------------------------------------------------------------------------------------
"""

model = Decision_Tree_Builder(house_train)

house_test_labels = list(house_test.pop('Labels'))
house_train_labels = list(house_train.pop('Labels'))

test_predictions = tools.predict_data(house_test, model)

train_predictions = tools.predict_data(house_train, model)

print("------------------")
print("Housing Linear Regression")
print("Training Error:")
print(tools.mean_square_error(train_predictions, house_train_labels))
print("Testing Error:")
print(tools.mean_square_error(test_predictions, house_test_labels))
