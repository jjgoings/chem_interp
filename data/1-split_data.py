""" Process data for use in ANN """
import numpy as np
import pandas as pd

RANDOM_SEED = 200

df = pd.read_csv('raw/proton_transfer_data.csv',index_col=0)
df.index.name = 'data_id'

train=df.sample(frac=0.80,random_state=RANDOM_SEED)
test=df.drop(train.index)
valid=test.sample(frac=0.50,random_state=RANDOM_SEED)
test=test.drop(valid.index)

x_train = train[train.columns[~train.columns.isin(['t1'])]]
x_test = test[test.columns[~test.columns.isin(['t1'])]]
x_valid = valid[valid.columns[~valid.columns.isin(['t1'])]]

y_train = train[train.columns[train.columns.isin(['t1'])]]
y_test = test[test.columns[test.columns.isin(['t1'])]]
y_valid = valid[valid.columns[valid.columns.isin(['t1'])]]

x_train.to_csv('./split_raw/x_train.csv')
y_train.to_csv('./split_raw/y_train.csv')

x_test.to_csv('./split_raw/x_test.csv')
y_test.to_csv('./split_raw/y_test.csv')

x_valid.to_csv('./split_raw/x_valid.csv')
y_valid.to_csv('./split_raw/y_valid.csv')
