""" Process data for use in ANN """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 200

# Load pre-split data sets
x_train = pd.read_csv('split_raw/x_train.csv',index_col=0)
x_test  = pd.read_csv('split_raw/x_test.csv',index_col=0)
x_valid = pd.read_csv('split_raw/x_valid.csv',index_col=0)
y_train = pd.read_csv('split_raw/y_train.csv',index_col=0)
y_test  = pd.read_csv('split_raw/y_test.csv',index_col=0)
y_valid = pd.read_csv('split_raw/y_valid.csv',index_col=0)

scal = StandardScaler()

# scale to x_train data, then transform others accordingly (avoid data leakage)
x_train_features = scal.fit_transform(x_train.values)
x_train = pd.DataFrame(x_train_features, index=x_train.index, columns=x_train.columns)

x_test_features = scal.transform(x_test.values)
x_test = pd.DataFrame(x_test_features, index=x_test.index, columns=x_test.columns)

x_valid_features = scal.transform(x_valid.values)
x_valid = pd.DataFrame(x_valid_features, index=x_valid.index, columns=x_valid.columns)


# NOTE: no TX on y-data, so we just copy over as-is
x_train.to_csv('./processed/x_train.csv')
y_train.to_csv('./processed/y_train.csv')

x_test.to_csv('./processed/x_test.csv')
y_test.to_csv('./processed/y_test.csv')

x_valid.to_csv('./processed/x_valid.csv')
y_valid.to_csv('./processed/y_valid.csv')
