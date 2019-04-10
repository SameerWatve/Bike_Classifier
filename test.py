# test.py

import pickle
import os
import cv2
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import array
from tensorflow.keras.models import Sequential
from numpy import newaxis
from tensorflow.keras import optimizers
import struct
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras import regularizers

img_size = 200
X_test = []
Y_test = []
path = "/Users/sameerwatve/Desktop/Nascent/TensorFlow_Tut_2_Classification_Walk-through-master/test_images/mountain_bikes/"
image = os.listdir(path)
for img in image:
    img1 = cv2.imread(path + img)
    if img1 is None:
        continue
    img1 = cv2.resize(img1, (img_size, img_size))
    X_test.append(array(img1 / 255.0))
    Y_test.append([1, 0])
path = "/Users/sameerwatve/Desktop/Nascent/TensorFlow_Tut_2_Classification_Walk-through-master/test_images/road_bikes/"
image = os.listdir(path)
for img in image:
    img1 = cv2.imread(path + img)
    if img1 is None:
        continue
    img1 = cv2.resize(img1, (img_size, img_size))
    X_test.append(array(img1 / 255.0))
    Y_test.append([0, 1])

X_test = array(X_test)
Y_test = array(Y_test)

print(X_test.shape, Y_test.shape)

learning_rate = 1e-3
epochs = 5
batch_size = 1
n_classes = 2  # num of output classes
drop_out = 0.2  # Drop-Out probability (Hyperparameter)
reg = 0.000001  # Regularization constant (Hyperparameter)
n_channels = 3  # RGB Image
len_test = X_test.shape[0]

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, img_size * img_size])
x = tf.reshape(x, [-1, img_size, img_size, n_channels])
y = tf.placeholder(tf.float32, shape=[None, 2])
# keep_prob = tf.placeholder(tf.float32)

W_conv1 = weight_variable([5, 5, n_channels, 16])
b_conv1 = bias_variable([16])
W_conv2 = weight_variable([5, 5, 16, 16])
b_conv2 = bias_variable([16])
W_fc1 = weight_variable([50 * 50 * 16, 64])
b_fc1 = bias_variable([64])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2d(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2d(h_conv2)

flattened = tf.reshape(h_pool2, [-1, 50 * 50 * 16])

h_fc1 = tf.nn.relu(tf.matmul(flattened, W_fc1) + b_fc1)
# h_fc1_drop = tf.nn.dropout(h_fc1, 1-drop_out)

W_fc2 = weight_variable([64, n_classes])
b_fc2 = bias_variable([n_classes])

y_ = tf.matmul(h_fc1, W_fc2) + b_fc2
prediction = tf.nn.softmax(y_)


print("Evaluate the model")
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, "/tmp/model.ckpt")

acc_test = 0
for i in range(len_test):
    batch_x = X_test[i * batch_size: (i + 1) * batch_size]
    batch_y = Y_test[i * batch_size: (i + 1) * batch_size]
    acc_test += sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}) / len_test

print("Test accuracy:", acc_test)