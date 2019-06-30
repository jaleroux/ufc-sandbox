import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from keras.models import Sequential
from keras.layers import Dense, Activation

import tensorflow as tf

import matplotlib.pyplot as plt

import logging
logging.getLogger().setLevel(logging.INFO)

working_dir = os.getcwd()

data_dir = '../Data/'
data_dir = 'Data/'

conn = sqlite3.connect(data_dir + 'db.sqlite3')

# read data into pandas df
sql_query = '''SELECT * FROM fights;'''
data_df = pd.read_sql_query(sql_query, conn)
data_df.to_csv('dbsqlite3.csv')

with open('output.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['Column 1', 'Column 2', ...])
    writer.writerows(data)

# missing variable's
data_df.isnull().mean()

# types
data_df.dtypes

# first removal of features
# data_df.filter(regex='judge|rating').columns.to_list
# remove_features = [col for col in data_df.columns if 'judge' in col]
remove_features = data_df.columns[data_df.columns.str.contains('judge|rating')].to_list()

data_df = data_df[data_df.columns.difference(remove_features)]

# type conversions
print(f'feature types are {data_df.dtypes}')
print(f'data types are {data_df.dtypes.unique()}')
feaeture_types = data_df.dtypes.unique()

# continuuous
feature_cont_names = data_df.select_dtypes(include=['float64','int64']).columns.to_list()
# categorical
feature_cat_names = data_df.select_dtypes(include='O').columns.to_list()

# missings
data_df[feature_cont_names].isnull().mean()
data_df[feature_cat_names].isnull().mean()

feature_cont_missings = data_df[feature_cont_names].columns[data_df[feature_cont_names].isnull().any()]

# replace missings
# continuous
def impute_na(df, variable, median):
    df[variable] = df[variable].fillna(median)

for name in feature_cont_missings:
    impute_na(data_df, name, data_df[name].median())

data_df[feature_cont_names].isnull().mean()

# categorical labels
# drop some variable for speed (date, time) etc
features_temporal_names = ['date', 'time']
feature_cat_names = [name for name in feature_cat_names if name not in features_temporal_names]

# number of categories
data_df[feature_cat_names]

for name in feature_cat_names:
    caridnality = data_df[name].unique().shape
    print(f'{name} has a caridnality: {caridnality}')

# drop some tag names
drop_features = ['fighter_1', 'fighter_2']
new_cont = ['attendance']
feature_cat_names = [name for name in feature_cat_names if name not in drop_features if name not in new_cont]
feature_cont_names = feature_cont_names + new_cont

target_name = ['result']
feature_cont_names.remove('result')

# categorical
# count/frequency encoding
def frequency_encoding(df, variable):
    x_frequency_map = df[variable].value_counts().to_dict()
    df[variable] = df[variable].map(x_frequency_map)

for name in feature_cat_names:
    frequency_encoding(data_df, name)

# continuous
data_df['attendance'] = data_df['attendance'].replace('','0').str.replace(',','').astype(dtype='int64')

# selection , final checks and to numpy
data_df[target_name].dtypes
data_df[feature_cat_names + feature_cont_names].dtypes
X_df = data_df[feature_cat_names + feature_cont_names]
Y_df = data_df[target_name]

X = X_df.values
Y = Y_df.values

Y = np.array([2 if y == 1 else ( 1 if y == 0 else 0) for y in Y])
#Y = np.array([1 if y == 1 else 0 for y in Y])

# tensorflow neural net (with estimator API)
K = len(set(Y.flat))
# make feature dict
features = feature_cat_names + feature_cont_names
feature_dict = dict(zip(features, X.T))

# 1 numpy input
train_input_1 = tf.estimator.inputs.numpy_input_fn(feature_dict, Y, batch_size = 100, num_epochs = 1, shuffle = True)

# define feature columns for input
tf_feature_columns = []
for key in feature_dict.keys():
    tf_feature_columns.append(tf.feature_column.numeric_column(key=key))

# instantiate estimator
tensorflow_dnn_model = tf.estimator.DNNClassifier(
    feature_columns = tf_feature_columns,
    hidden_units = [5,5],
    n_classes = K)

# train model
# 1) using the numpy input parameter
tf.logging.set_verbosity(tf.logging.INFO)
tensorflow_dnn_model.train(
    input_fn = train_input_1
)

# evaluate model
# 1
tf.logging.set_verbosity(tf.logging.INFO)
results = tensorflow_dnn_model.evaluate(
    input_fn = train_input_1,
    steps=1
)

print(f'Accuracy of the ole tensorflow api {results}')