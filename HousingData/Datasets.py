import pandas as pd

_names_housing = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSAT",
                  "Labels"]

data_housing_train = pd.read_csv('./HousingData/Housing_Training.txt', delim_whitespace=True, header=None,
                                 names=_names_housing, dtype=float)

data_housing_test = pd.read_csv('./HousingData/Housing_Test.txt', delim_whitespace=True, header=None,
                                names=_names_housing, dtype=float)


def Rescaling_Normalization(data1: pd.DataFrame, data2=pd.DataFrame) -> pd.DataFrame:
    """
    Uses Rescaling Method (x-min(X))/(max(X)-min(X)) to normalize the data set
    Args:
        data1: the test or train dataset that is to be normalized
        data2: the test or train dataset that is to be normalized

    Returns: the normalized pandas data

    """

    data = pd.concat([data1, data2])

    labels = data.pop('Labels')

    norm_data = (data - data.min()) / (data.max() - data.min())

    norm_data['Labels'] = labels

    norm_data1 = norm_data.iloc[:len(data1)]

    norm_data2 = norm_data.iloc[len(data1):]

    return norm_data1, norm_data2


norm_train, norm_test = Rescaling_Normalization(data_housing_train.copy(), data_housing_test.copy())

data_housing_train['Bias'] = 1
data_housing_test['Bias'] = 1
norm_train['Bias'] = 1
norm_test['Bias'] = 1
