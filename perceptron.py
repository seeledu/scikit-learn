import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
import gzip
import pickle
from keras.preprocessing import sequence
import numpy as np
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from tensorflow.contrib import learn
import os
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import os
import warnings
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Convolution2D, Input, Activation, MaxPooling2D, Reshape, Dropout, Dense, \
    Flatten, Merge
from keras.layers import LSTM
import gzip
import pickle
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import tensorflow as tf
from keras import regularizers


filename = 'log.txt'
max_features = 2298
maxlen = 50  # 限定最大词数
len_wv = 50 # 词向量维数
nb_classes = 31 # 分类类别
batch_size = 100 # 每一批次样本量
mnist = fetch_mldata("MNIST original",data_home="H:/research/data")
tf.flags.DEFINE_float("dev_sample_percentage", .05, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

def load_data(max_features,dataset):
    '''
    读取数据，暂不设验证集
    :param dataset:
    :return:
    '''
    TRAIN_SET = max_features
    f = gzip.open(dataset, 'rb')
    data = pickle.load(f)
    f.close()
    data_x, data_y = data
    train_x = data_x[:TRAIN_SET]
    train_y = data_y[:TRAIN_SET]
    test_x = data_x[TRAIN_SET:]
    test_y = data_y[TRAIN_SET:]
    return train_x, train_y, test_x, test_y

print('Loading data')
x_train, y_train, x_test, y_test = load_data(max_features, dataset='H:/research/data/smp.pretrain.pkl.gz')
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
for i in range(50):
    print(x_train[i])
# Memory 足够时用
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

y_test = np_utils.to_categorical(y_test, nb_classes)  # 必须使用固定格式表示标签
y_train = np_utils.to_categorical(y_train, nb_classes)  # 必须使用固定格式表示标签 一共 42分类

x_train = x_train.reshape((-1, maxlen*len_wv))
x_test = x_test.reshape((-1, maxlen*len_wv))
# x=np.vstack((x_test,x_train))
# y=np.vstack((y_test,y_train))


# Split train/test set
# TODO: This is very crude, should use cross-validation
# dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y))) #dev_sample_percentage为0.1，表示train中划分出0.1的做验证
# x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
# y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

# alg = RandomForestClassifier()
# alg = RandomForestClassifier(min_samples_leaf=1, n_estimators=500, random_state=50)
# alg.fit(x_train,y_train)
# print("2",alg.score(x_test,y_test))
# for i in range(50):
#     for j in range(50):
#         print(x_dev[i])
# ===========================
# rescale the data, use the traditional train/test split
# X, y = mnist.data / 255., mnist.target
# X_train, X_test = X[:60000], X[60000:]
# y_train, y_test = y[:60000], y[60000:]
# print(y_test)

# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
mlp = MLPClassifier(hidden_layer_sizes=(50,260), max_iter=1000, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.001)

mlp.fit(x_train, y_train)
print("Training set score: %f" % mlp.score(x_train, y_train))
print("Test set score: %f" % mlp.score(x_test, y_test))

# fig, axes = plt.subplots(4, 4)
# # use global min / max to ensure all weights are shown on the same scale
# vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
# for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
#     ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
#                vmax=.5 * vmax)
#     ax.set_xticks(())
#     ax.set_yticks(())
#
# plt.show()