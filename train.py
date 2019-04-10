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

print("loading train data")
X_train = []
Y_train = []
path = "/Users/sameerwatve/Desktop/Nascent/TensorFlow_Tut_2_Classification_Walk-through-master/training_images/mountain_bikes/"
image = os.listdir(path)
for img in image:
    img1 = cv2.imread(path + "/" + img)
    if img1 is None:
        continue
    img1 = cv2.resize(img1,(img_size,img_size))
    X_train.append(np.array(img1 / 255.0))
    Y_train.append([1, 0])
path = "/Users/sameerwatve/Desktop/Nascent/TensorFlow_Tut_2_Classification_Walk-through-master/training_images/road_bikes/"
image = os.listdir(path)
for img in image:
    img1 = cv2.imread(path + img)
    if img1 is None:
        continue
    img1 = cv2.resize(img1, (img_size, img_size))
    X_train.append(array(img1 / 255.0))
    Y_train.append([0, 1])

X_train = array(X_train)
Y_train = array(Y_train)
print("loading test data")


print(X_train.shape, Y_train.shape)



X_train, X_CV, Y_train, Y_CV = train_test_split(X_train, Y_train, test_size=0.30, random_state=42)

learning_rate = 1e-3
epochs = 5
batch_size = 1
n_classes = 2  # num of output classes
drop_out = 0.2  # Drop-Out probability (Hyperparameter)
reg = 0.000001  # Regularization constant (Hyperparameter)
n_channels = 3  # RGB Image
len_train = X_train.shape[0]
len_CV = X_CV.shape[0]


def print_params():
    print('Learning Rate: ', learning_rate)
    print('Regularization:', reg)


def log_parameters():
    arr = [len_train, len_CV, learning_rate, epochs, batch_size, 1 - drop_out, reg]
    np.savetxt('Results/params.txt', arr)
    print("Parameters logged")


def save(arr):
    print("Dumping to a file")
    with open('data.p', 'wb') as f:
        pickle.dump(arr, f)
    print("Saved")


print("Building model")


# Using Keras API
# model = Sequential()
# model.add(Conv2D(16, (5, 5), input_shape=(60, 60, 3)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(16, (5, 5), input_shape=(30, 30, 3)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Flatten())
#
# model.add(Dense(3600))
# model.add(Dropout(0.2))
# model.add(Activation('relu'))
# model.add(Dense(128))
# model.add(Dropout(0.2))
# model.add(Activation('relu'))
# model.add(Dense(2))
# model.add(Activation('softmax'))
#
# adam = optimizers.Adam(lr=0.00724, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
# history= model.fit(X_train, Y_train, batch_size=128, epochs=5, validation_split=0.2)
# model.evaluate(X_CV, Y_CV)


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

total_batches_train = int(len_train / batch_size)
total_batches_CV = int(len_CV / batch_size)

saver = tf.train.Saver()
sess = tf.Session()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_)) + \
                reg * (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + \
                       tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

print("Evaluate the model")
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

train_acc_arr = []
test_acc_arr = []
cost_arr = []
epoch_arr = []
print_params()

for epoch in range(epochs):

    # Average Cost
    avg_cost = 0
    for i in range(total_batches_train):
        batch_x = X_train[i * batch_size: (i + 1) * batch_size]
        batch_y = Y_train[i * batch_size: (i + 1) * batch_size]

        _, c = sess.run([train_step, cross_entropy],
                        feed_dict={x: batch_x, y: batch_y})
        avg_cost += c / total_batches_train

    # Training Accuracy
    acc_train = 0
    for i in range(total_batches_train):
        batch_x = X_train[i * batch_size: (i + 1) * batch_size]
        batch_y = Y_train[i * batch_size: (i + 1) * batch_size]
        acc_train += sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}) / total_batches_train

        # Validation Accuracy
    acc_val = 0
    for i in range(total_batches_CV):
        batch_x = X_CV[i * batch_size: (i + 1) * batch_size]
        batch_y = Y_CV[i * batch_size: (i + 1) * batch_size]
        acc_val += sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}) / total_batches_CV

    # acc_val = 0
    # for i in range(total_batches_test):
    #     batch_x = X_test[i * batch_size: (i + 1) * batch_size]
    #     batch_y = Y_test[i * batch_size: (i + 1) * batch_size]
    #     acc_val += sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}) / total_batches_test

    epoch_arr.append(epoch + 1)
    cost_arr.append(avg_cost)
    train_acc_arr.append(acc_train)
    test_acc_arr.append(acc_val)

    results = [epoch_arr, train_acc_arr, test_acc_arr, cost_arr]
    results = np.array(results)
    with open('Results/results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("Epoch:", epoch, "cost:", avg_cost, "Train accuracy:", acc_train, "Val accuracy:", acc_val)

log_parameters()
save_path = saver.save(sess, "/tmp/model.ckpt")
print("Model saved in path: %s" % save_path)
#
#
# p1 = plt.figure(1)
# plt.plot(epoch_arr, train_acc_arr, label='Training Accuracy', marker='o')
# plt.plot(epoch_arr, test_acc_arr, label='Testing Accuracy', marker='o')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Epoch vs Accuracy')
# plt.legend(loc=4)
# p1.savefig('Results/Accuracy.png')
#
# p2 = plt.figure(2)
# plt.plot(epoch_arr, cost_arr, marker='o')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Epoch vs Loss')
# plt.legend(loc=0)
# p2.savefig('Results/Loss.png')
#
# plt.show()
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper right')
# plt.grid()
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper right')
# plt.grid()
# plt.show()

pass



















































































































