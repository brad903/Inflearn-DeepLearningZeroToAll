import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import random

# 데이터 로드
Image_path='./npy_dataset/X.npy'
X = np.load(Image_path)
label_path='./npy_dataset/Y.npy'
Y = np.load(label_path)

print("X Dataset:",X.shape)
print("Y Dataset:",Y.shape)

# 테스트 데이터, 트레이닝 데이터 분리
from sklearn.model_selection import train_test_split
test_size = 0.15
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=test_size, random_state=42)

img_size = 64
channel_size = 1
print("Training Size:", X_train.shape)
print(X_train.shape[0],"samples - ", X_train.shape[1],"x",X_train.shape[2],"grayscale image")

print("Test Size:",X_test.shape)
print(X_test.shape[0],"samples - ", X_test.shape[1],"x",X_test.shape[2],"grayscale image")

# 데이터 가공
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

# 데이터 가공
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


import tensorflow as tf

# Placeholder
X = tf.placeholder(tf.float32,name="X-Input")
Y = tf.placeholder(tf.float32,name="Y-Output")

# Hyperparameter
learning_rate = 0.1
epochs = 20000
x_inp = x_result_array.shape[1]  # 뉴런 입력 값
n_n = 2000  # wide 결정할 변수
y_out = 10  # 뉴런 최종 출력값 → 10 Classes
keep_prob = tf.placeholder(tf.float32) # dropout값

# 뉴런 네트워크 모델
with tf.name_scope("layer1"):
    w1 = tf.get_variable("w1", shape=[x_inp, n_n],
                     initializer=tf.contrib.layers.xavier_initializer())  # xavier를 이용하여 초기화!
    b1 = tf.Variable(tf.random_normal([n_n]))
    A1 = tf.matmul(X, w1) + b1
    H1 = tf.nn.relu(A1)
    H1 = tf.nn.dropout(H1, keep_prob=keep_prob)

    w1_hist = tf.summary.histogram("weights1", w1)
    b1_hist = tf.summary.histogram("biases1", b1)
    layer1_hist = tf.summary.histogram("layer1", A1)

with tf.name_scope("layer2"):
    w2 = tf.get_variable("w2", shape=[n_n, n_n],
                     initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([n_n]))
    A2 = tf.matmul(H1, w2) + b2
    H2 = tf.nn.relu(A2)
    H2 = tf.nn.dropout(H2, keep_prob=keep_prob)

    w2_hist = tf.summary.histogram("weights2", w2)
    b2_hist = tf.summary.histogram("biases2", b2)
    layer2_hist = tf.summary.histogram("layer2", A2)

with tf.name_scope("layer3"):
    w3 = tf.get_variable("w3", shape=[n_n, n_n],
                         initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([n_n]))
    A3 = tf.matmul(H2, w3) + b3
    H3 = tf.nn.relu(A3)
    H3 = tf.nn.dropout(H3, keep_prob=keep_prob)

    w3_hist = tf.summary.histogram("weights3", w3)
    b3_hist = tf.summary.histogram("biases3", b3)
    layer3_hist = tf.summary.histogram("layer3", A3)

with tf.name_scope("layer4"):
    w4 = tf.get_variable("w4", shape=[n_n, n_n],
                         initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([n_n]))
    A4 = tf.matmul(H3, w4) + b4
    H4 = tf.nn.relu(A3)
    H4 = tf.nn.dropout(H4, keep_prob=keep_prob)

    w4_hist = tf.summary.histogram("weights4", w4)
    b4_hist = tf.summary.histogram("biases4", b4)
    layer4_hist = tf.summary.histogram("layer4", A4)

with tf.name_scope("layer5"):
    w5 = tf.get_variable("w5", shape=[n_n, y_out],
                         initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([y_out]))
    hypothesis = tf.matmul(H4, w5) + b5

    w5_hist = tf.summary.histogram("weights5", w5)
    b5_hist = tf.summary.histogram("biases5", b5)
    layer5_hist = tf.summary.histogram("layer5", hypothesis)

#Output Layer
logit = tf.add(tf.matmul(H4,w5),b5)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,labels=Y)

#Cost
with tf.name_scope("cost"):
    cost = (tf.reduce_mean(cross_entropy))
    cost_summ = tf.summary.scalar("cost", cost)

#Optimizer
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#Prediction
y_pred = tf.nn.softmax(logit)


with tf.Session() as sess:
    # tensorboard --logdir=./logs/sign_language
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/sign_language_01")
    writer.add_graph(sess.graph)  # Show the graph

    init = tf.global_variables_initializer()
    sess.run(init)

    p = []
    for epoch in range(epochs):
        feed_dict = {X: x_result_array, Y: Y_train, keep_prob: 0.7}
        c, _, summary = sess.run([cost, optimizer, merged_summary], feed_dict=feed_dict)
        writer.add_summary(summary, global_step=epoch)

        if epoch%10 ==0:
            print('Epoch:', '%05d' % (epoch), 'cost =', '{:.9f}'.format(c))
            p.append(c)

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_summ = tf.summary.scalar("accuracy", accuracy)
    print('Accuracy:', sess.run(accuracy, feed_dict={X:x_test_array, Y:Y_test, keep_prob: 1}))

    # Get one and predict
    r = random.randint(0, x_test_array.shape[0] - 1)
    print("Label: ", sess.run(tf.argmax(Y_test[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(logit, 1), feed_dict={X: x_test_array[r:r + 1], keep_prob: 1}))

    plt.plot(p[1:])
    plt.title("Cost Decrease")
    plt.show()

    # plt.imshow(X_test[r:r + 1].
    #           reshape(64, 64), cmap='Greys', interpolation='nearest')
    # plt.show()