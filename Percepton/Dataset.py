import pandas as pd

headers = ['x1', 'x2', 'x3', 'x4', 'Labels']

data = pd.read_csv('./Percepton/Perecepton_Data', names=headers, delim_whitespace=True, header=None, dtype=float)

