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
batch_size = 2000# 每一批次样本量
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


# def compute_accuracy( v_xs, v_ys):
#     global prediction,sess, x, y_, keep_prob
#     y_pre = sess.run(prediction, feed_dict={x: v_xs, keep_prob: 1})
#     correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     result = sess.run(accuracy, feed_dict={x: v_xs, y_: v_ys, keep_prob: 1})
#     return result
#
print('Loading data')
x_train, y_train, x_test, y_test = load_data(max_features, dataset='H:/research/data/smp.pretrain.pkl.gz')
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
# for i in range(50):
#     print(x_train[i])

# Memory 足够时用
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

y_test = np_utils.to_categorical(y_test, nb_classes)  # 必须使用固定格式表示标签
y_train = np_utils.to_categorical(y_train, nb_classes)  # 必须使用固定格式表示标签 一共 32分类

x_train = x_train.reshape((-1, maxlen*len_wv))
x_test = x_test.reshape((-1, maxlen*len_wv))
print(len(x_train[1]))
# sklearn 下的简易多层感知机
mlp = MLPClassifier(hidden_layer_sizes=(50,200), max_iter=1000, alpha=0.01,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.001)

mlp.fit(x_train, y_train)
print("Training set score: %f" , mlp.score(x_train, y_train))
print("Test set score: %f" , mlp.score(x_test, y_test))
#
# #调高扩展性，使用tensorflow进行网络建设

# learning_rate = 0.001
# training_epochs = 150
# batch_size = 200
# display_step = 1
# # Network Parameters
# n_hidden_1 = 256 # 1st layer number of features
# n_hidden_2 = 256 # 2nd layer number of features
# n_input = 2500 # MNIST data input (img shape: 28*28)
# n_classes = 32 # MNIST total classes (0-9 digits)
# # tf Graph input
# x = tf.placeholder("float", [None, n_input])
# y = tf.placeholder("float", [None, n_classes])
# # Store layers weight & bias
# weights = {
#     'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
#     'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
# }
# biases = {
#     'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }
# # Create model
# def multilayer_perceptron(x, weights, biases):
#     # Hidden layer with RELU activation
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     layer_1 = tf.nn.relu(layer_1)
#     # Hidden layer with RELU activation
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     layer_2 = tf.nn.relu(layer_2)
#     # Output layer with linear activation
#     out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
#     return out_layer
# # Construct model
# pred = multilayer_perceptron(x, weights, biases)
# # Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# # Initializing the variables
# init = tf.initialize_all_variables()
# # Launch the graph
# with tf.Session() as sess:
#     sess.run(init)
#     # Training cycle
#     for epoch in range(1):
#         avg_cost = 0.
#         start_index = 0
#         end_index = batch_size
#         for i in range(300):
#             if end_index > start_index:
#                 batch_xs = x_train[start_index:end_index]
#                 batch_ys = y_train[start_index:end_index]
#             else:
#                 batch_xs = np.concatenate((x_train[start_index:len(x_train)], x_train[0:end_index]), axis=0)
#                 batch_ys = np.concatenate((y_train[start_index:len(x_train)], y_train[0:end_index]), axis=0)
#             start_index += batch_size
#             end_index += batch_size
#             start_index %= len(x_train)
#             end_index %= len(x_train)
#             # Run optimization op (backprop) and cost op (to get loss value)
#             _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,y: batch_ys})
#
#         # Display logs per epoch step
#         if epoch % display_step == 0:
#             print ("Epoch:", '%04d' % (epoch+1))
#     print ("Optimization Finished!")
#     # Test model
#     correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#     # Calculate accuracy
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#     print ("Accuracy:", accuracy.eval({x: x_test, y: y_test}))
#
#
# in_units = 2500  # 输入节点数
# h1_units = 500  # 隐含层节点数
# W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))  # 初始化隐含层权重W1，服从默认均值为0，标准差为0.1的截断正态分布
# b1 = tf.Variable(tf.zeros([h1_units]))  # 隐含层偏置b1全部初始化为0
# W2 = tf.Variable(tf.zeros([h1_units, 31]))
# b2 = tf.Variable(tf.zeros([31]))
# x = tf.placeholder(tf.float32, [None, in_units])
# keep_prob = tf.placeholder(tf.float32)  # Dropout失活率
#
# # 定义模型结构
# hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
# hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
# y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)
#
# # 训练部分
# y_ = tf.placeholder(tf.float32, [None, 31])
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
#
# # 定义一个InteractiveSession会话并初始化全部变量
# sess = tf.InteractiveSession()
# tf.global_variables_initializer().run()
# correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# start_index = 0
# end_index = batch_size
# for i in range(3001):
#     if end_index > start_index:
#         batch_xs = x_train[start_index:end_index]
#         batch_ys = y_train[start_index:end_index]
#     else:
#         batch_xs = np.concatenate((x_train[start_index:len(x_train)], x_train[0:end_index]), axis=0)
#         batch_ys = np.concatenate((y_train[start_index:len(x_train)], y_train[0:end_index]), axis=0)
#     start_index += batch_size
#     end_index += batch_size
#     start_index %= len(x_train)
#     end_index %= len(x_train)
#     train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})
#     if i % 200 == 0:
#         # 训练过程每200步在测试集上验证一下准确率，动态显示训练过程
#         print(i, 'training_arruracy:', accuracy.eval({x:x_train, y_: y_train,
#                                                       keep_prob: 1.0}))
# print('final_accuracy:', accuracy.eval({x:x_test, y_: y_test, keep_prob: 1.0}))



#=========================================
# global prediction, x, y_, keep_prob
#
#
# in_units = 2500
# h1_units = 300
# w1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
# b1 = tf.Variable(tf.zeros([h1_units]))
# w2 = tf.Variable(tf.zeros([h1_units,31]))
# b2 = tf.Variable(tf.zeros([31]))
#
# x = tf.placeholder(tf.float32, [None, in_units])
# y_ = tf.placeholder(tf.float32, [None, 31])
# keep_prob = tf.placeholder(tf.float32)
#
# hidden1 = tf.nn.relu(tf.matmul(x, w1)+b1)
# hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
# y = tf.nn.softmax(tf.matmul(hidden1_drop, w2) + b2)
# prediction = y
#
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)), reduction_indices=[1]))
# print(cross_entropy)
# # cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
# train_step = tf.train.AdadeltaOptimizer(0.01).minimize(cross_entropy)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     start_index = 0
#     end_index = batch_size
#     for i in range(1000):
#         if end_index > start_index:
#             batch_xs = x_train[start_index:end_index]
#             batch_ys = y_train[start_index:end_index]
#         else:
#             batch_xs = np.concatenate((x_train[start_index:len(x_train)], x_train[0:end_index]), axis=0)
#             batch_ys = np.concatenate((y_train[start_index:len(x_train)], y_train[0:end_index]), axis=0)
#         sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.90})
#         start_index += batch_size
#         end_index += batch_size
#         start_index %= len(x_train)
#         end_index %= len(x_train)
#         if i % 25 == 0:
#             print("第",i,"batch",compute_accuracy(x_train, y_train))
#         if i % 100 == 0:
#             print(compute_accuracy(x_test, y_test))


#     sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
#

#
