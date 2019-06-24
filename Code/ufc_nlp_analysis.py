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
logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
logreg = LogisticRegression(C=1e3, solver='lbfgs')
logreg.fit(X, Y)
pred = logreg.predict(X)

print("accuracy: ", accuracy_score(Y, pred))

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

# make feature dict
features = feature_cat_names + feature_cont_names
feature_dict = dict(zip(features, X.T))

# 1 numpy input
train_input_1 = tf.estimator.inputs.numpy_input_fn(feature_dict, Y, batch_size = 100, num_epochs = 1, shuffle = True)

# 2 regular input
def input_fn_2(feature_dict, labels, batch_size):

    dataset = tf.data.Dataset.from_tensor_slices((dict(feature_dict), labels))

    return dataset.shuffle(100).repeat().batch(batch_size) # returns (features, labels)


# define feature columns for input
tf_feature_columns = []
for key in feature_dict.keys():
    tf_feature_columns.append(tf.feature_column.numeric_column(key=key))

# instantiate estimator
tensorflow_dnn_model = tf.estimator.DNNClassifier(
    feature_columns = tf_feature_columns,
    hidden_units = [5,5],
    n_classes = 2)

# train model
# 1) using the numpy input parameter
tf.logging.set_verbosity(tf.logging.INFO)
tensorflow_dnn_model.train(
    input_fn = train_input_1
)

# 2)
tf.logging.set_verbosity(tf.logging.INFO)
tensorflow_dnn_model.train(
    input_fn = lambda: input_fn_2(feature_dict, Y, batch_size=100),
    steps = np.floor(N/100))

# evaluate model
# 1
tf.logging.set_verbosity(tf.logging.INFO)
results = tensorflow_dnn_model.evaluate(
    input_fn = train_input_1,
    steps=1
)

# 2
tf.logging.set_verbosity(tf.logging.INFO)
results = tensorflow_dnn_model.evaluate(
    input_fn = lambda: input_fn_2(feature_dict, Y, batch_size=N),
    steps=1
)

print(f'Accuracy of the ole tensorflow api {results}')

# tensorflow neural net (lower level)

# helper functions
def predict(p_y):
    return np.argmax(p_y, axis=1)
def error_rate(p_y, t):
    prediction = predict(p_y)
    return np.mean(prediction != t)

# define / rename variables
X_train = X
Y_train = Y

learning_rate = 0.00004
max_iter = 10
batchsize = 100
#n_batches= np.floor(N/batchsize)
n_batches = N // batchsize

M1 = 5
M2 = 5

W1_init = np.random.rand(D, M1) / np.sqrt(D)
b1_init = np.zeros(M1)
W2_init = np.random.rand(M1, M2) / np.sqrt(M1)
b2_init = np.zeros(M2)
W3_init = np.random.rand(M2, K) / np.sqrt(M2)
b3_init = np.zeros(K)

X = tf.placeholder(tf.float32, shape=(None, D), name ='X')
T = tf.placeholder(tf.float32, shape=(None, K), name ='T')
W1 = tf.Variable(W1_init.astype(np.float32))
b1 = tf.Variable(b1_init.astype(np.float32))
W2 = tf.Variable(W2_init.astype(np.float32))
b2 = tf.Variable(b2_init.astype(np.float32))
W3 = tf.Variable(W3_init.astype(np.float32))
b3 = tf.Variable(b3_init.astype(np.float32))

# assert shapes of tensors
assert X.get_shape().as_list() == [None, D]
assert T.get_shape().as_list() == [None, K]
assert W1.get_shape().as_list() == [D, M1]
assert b1.get_shape().as_list() == [K]
assert W2.get_shape().as_list() == [M1, M2]
assert b2.get_shape().as_list() == [M2]
assert W3.get_shape().as_list() == [M2, K]
assert b3.get_shape().as_list() == [K]

# model (feed forward algo)
Z1 = tf.nn.relu( tf.matmul(X, W1) + b1 )
Z2 = tf.nn.relu( tf.matmul(Z1, W2) + b2 )
Yish = tf.matmul(Z2, W3) + b3

# cost
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Yish, labels=T))

# train
train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99, momentum=0.9).minimize(cost)

# predict
predict_op = tf.argmax(Yish, 1)

# train loop
costs = []
init = tf.global_variables_initializer()

# alternative way of starting
# sess = tf.Session()
# sess.run(init)


with tf.Session() as session:
    session.run(init)

    # train loop
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = X_train[j*batchsize:(j*batchsize+batchsize),]
            Ybatch = Y_train[j*batchsize:(j*batchsize+batchsize),]

            session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})

            if j % 5 == 0:
                train_cost = session.run(cost, feed_dict={X: Xbatch, T: Ybatch})
                prediction = session.run(predict_op, feed_dict={X: Xbatch, T: Ybatch})
                error = error_rate(Ybatch, prediction)

                # Evaluate the accuracy of the model
                correct_prediction = tf.equal(prediction, tf.argmax(Ybatch, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

                print(f"Cost / err at iteration {i}, {j}: {train_cost} / {error} / {accuracy}")
                costs.append(train_cost)

# sess.close()s

plt.plot(costs)
plt.show()

# run on the GCP


# try cross validation
# asserting shapes


########################################
