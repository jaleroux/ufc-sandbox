import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from keras.models import Sequential
from keras.layers import Dense, Activation

import matplotlib.pyplot as plt

working_dir = os.getcwd()

data_dir = '../Data/'

conn = sqlite3.connect(data_dir + 'db.sqlite3')

# cursor = conn.cursor()
# cursor.execute('''SELECT * FROM fights;''')
# data = cursor.fetchall()
# cursor.close()

# read data into pandas df
sql_query = '''SELECT * FROM fights;'''
data_df = pd.read_sql_query(sql_query, conn)

# summary stats

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

Y = np.array([1 if y == 1 else 0 for y in Y])

def y2indicator(y):
    N = len(y)
    k = len(set(y.flat))
    y = y.astype(np.int32)
    ind = np.zeros((N, k))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

Y = y2indicator(Y)

# multinomial logistic regression
# logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
# logreg = LogisticRegression(C=1e3, solver='lbfgs')
# logreg.fit(X, Y)
# pred = logreg.predict(X)
#
# print("accuracy: ", accuracy_score(Y, pred))

# keras neural net
N, D = X.shape
K = len(set(Y.flat))

model = Sequential()

# ANN: layers [23] -> [5] -> [5] -> [K]
model.add(Dense(units = 5, input_dim=D))
model.add(Activation('relu'))
model.add(Dense(units=5))
model.add(Activation('relu'))
model.add(Dense(units=K))
model.add(Activation('softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

r = model.fit(X, Y, validation_split=0.33, epochs=15, batch_size=32)
print(f'Returned: {r}')
print(f'{r.history.keys()}')

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['acc'], label = 'acc')
plt.plot(r.history['val_acc'], label = 'val_acc')
plt.legend()
plt.show()

# tensorflow neural net (with estimator API)

# tensorflow neural net (lower level)

########################################
