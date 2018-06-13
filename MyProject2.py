import numpy as np
import tensorflow as tf
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2

# Load Data
Image_path='./npy_dataset/X.npy'
x = np.load(Image_path)
label_path='./npy_dataset/Y.npy'
y = np.load(label_path)

print("X Dataset:",x.shape)
print("Y Dataset:",y.shape)

# print(X_train)  # (1649, 64, 64)
# print(X_test.shape)   # (413, 64, 64)
# print(Y_train.shape)  # (1649, 10)
# print(Y_test.shape)   # (413, 10)


# Split Dataset
from sklearn.model_selection import train_test_split
test_size = 0.15
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=test_size, random_state=42)

img_size = 64
channel_size = 1
print("Training Size:", X_train.shape)
print(X_train.shape[0],"samples - ", X_train.shape[1],"x",X_train.shape[2],"grayscale image")
print("Test Size:",X_test.shape)
print(X_test.shape[0],"samples - ", X_test.shape[1],"x",X_test.shape[2],"grayscale image")

# Array For Training
x_result_array = np.zeros(shape=(1024, 0))

width = 32
height = 32

for i in range(0, len(X_train)):
    img = X_train[i]
    img1 = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    t = img1.reshape(1, 1024).T
    x_result_array = np.concatenate([x_result_array, t], axis=1)

x_result_array = x_result_array.T
print(x_result_array.shape)
print(Y_train.shape)

# Array For Testing
x_test_array = np.zeros(shape=(1024,0))

width_test = 32
height_test = 32

for i in range (0,len(X_test)):
    img = X_test[i]
    img1 = cv2.resize(img, (width,height), interpolation=cv2.INTER_CUBIC)
    t = img1.reshape(1,1024).T
    x_test_array = np.concatenate([x_test_array,t], axis=1)

x_test_array = x_test_array.T
print(x_test_array.shape)
print(Y_test.shape)


# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, 64, 64])
X_img = tf.reshape(X, [-1, 64, 64, 1])   # img 64x64x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape=(?, 64, 64, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
#    Conv     -> (?, 64, 64, 32)
#    Pool     -> (?, 32, 32, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
'''
Tensor("Conv2D:0", shape=(?, 64, 64, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 64, 64, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 32, 32, 32), dtype=float32)
Tensor("dropout/mul:0", shape=(?, 32, 32, 32), dtype=float32)
'''

# L2 ImgIn shape=(?, 32, 32, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 32, 32, 64)
#    Pool      ->(?, 16, 16, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
'''
Tensor("Conv2D_1:0", shape=(?, 32, 32, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 32, 32, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 16, 16, 64), dtype=float32)
Tensor("dropout_1/mul:0", shape=(?, 16, 16, 64), dtype=float32)
'''

# L3 ImgIn shape=(?, 16, 16, 64)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
#    Conv      ->(?, 16, 16, 128)
#    Pool      ->(?, 8, 8, 128)
#    Reshape   ->(?, 8 * 8 * 128) # Flatten them for FC
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3_flat = tf.reshape(L3, [-1, 128 * 8 * 8])
'''
Tensor("Conv2D_2:0", shape=(?, 16, 16, 128), dtype=float32)
Tensor("Relu_2:0", shape=(?, 16, 16, 128), dtype=float32)
Tensor("MaxPool_2:0", shape=(?, 8, 8, 128), dtype=float32)
Tensor("dropout_2/mul:0", shape=(?, 8, 8, 128), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 8192), dtype=float32)
'''

# L4 FC 4x4x128 inputs -> 625 outputs
W4 = tf.get_variable("W4", shape=[128 * 8 * 8, 625],  # fully-connected layer을 하나 더 늘림, dropout도 사용
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
'''
Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
'''

# L5 Final FC 625 inputs -> 10 outputs
W5 = tf.get_variable("W5", shape=[625, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4, W5) + b5

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Prediction
y_pred = tf.nn.softmax(logits)
pred = tf.argmax(y_pred, axis=1)

# train my model
print('Learning started. It takes sometime.')

print(x_result_array[0])

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for step in range(2000):
#         feed_dict={X: X_train, Y: Y_train, keep_prob: 0.7}
#         c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
#         sess.run(optimizer, feed_dict={X: X_train, Y: Y_train})
#         if step % 100 == 0:
#             loss, acc = sess.run([cost, y_pred], feed_dict={
#                                  X: X_train, Y: Y_train})
#             print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
#                 step, loss, acc))

    # # Let's see if we can predict
    # pred = sess.run(prediction, feed_dict={X: x_data})
    # # y_data: (N,1) = flatten => (N, ) matches pred.shape
    # for p, y in zip(pred, y_data.flatten()):
    #     print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

# for epoch in range(training_epochs):
#     avg_cost = 0
#     total_batch = int(2062 / batch_size)
#
#     for i in range(total_batch):
#         batch_xs = x.train.next_batch(batch_size)
#         batch_ys = y.train.next_batch(batch_size)
#         feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
#         c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
#         avg_cost += c / total_batch
#
#     print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
#
# print('Learning Finished!')

# # Test model and check accuracy
#
# # if you have a OOM error, please refer to lab-11-X-mnist_deep_cnn_low_memory.py
#
# correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print('Accuracy:', sess.run(accuracy, feed_dict={
#       X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
#
# # Get one and predict
# r = random.randint(0, mnist.test.num_examples - 1)
# print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
# print("Prediction: ", sess.run(
#     tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))
#
# # plt.imshow(mnist.test.images[r:r + 1].
# #           reshape(28, 28), cmap='Greys', interpolation='nearest')
# # plt.show()
#
