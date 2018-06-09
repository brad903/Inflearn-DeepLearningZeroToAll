# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
tf.set_random_seed(777)  # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data  # MNIST 데이터는 유명해서 따로 라이브러리가 있음
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 데이터를 읽어올 때 one_hot 통해 읽어올 수 있음

nb_classes = 10  # 0~9 까지 숫자를 의미
layer_num = 40

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# 수정 부분
with tf.name_scope("layer1"):
    W1 = tf.Variable(tf.random_normal([784, layer_num]), name='weight1')
    b1 = tf.Variable(tf.random_normal([layer_num]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    w1_hist = tf.summary.histogram("weights1", W1)
    b1_hist = tf.summary.histogram("biases1", b1)
    layer1_hist = tf.summary.histogram("layer1", layer1)

with tf.name_scope("layer2"):
    W2 = tf.Variable(tf.random_normal([layer_num, layer_num]), name='weight2')
    b2 = tf.Variable(tf.random_normal([layer_num]), name='bias2')
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    w2_hist = tf.summary.histogram("weights2", W2)
    b2_hist = tf.summary.histogram("biases2", b2)
    layer2_hist = tf.summary.histogram("layer2", layer2)

with tf.name_scope("layer3"):
    W3 = tf.Variable(tf.random_normal([layer_num, layer_num]), name='weight3')
    b3 = tf.Variable(tf.random_normal([layer_num]), name='bias3')
    layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

    w3_hist = tf.summary.histogram("weights3", W3)
    b3_hist = tf.summary.histogram("biases3", b3)
    layer3_hist = tf.summary.histogram("layer3", layer3)

with tf.name_scope("layer4"):
    W4 = tf.Variable(tf.random_normal([layer_num, nb_classes]), name='weight4')
    b4 = tf.Variable(tf.random_normal([nb_classes]), name='bias4')
    hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

    w4_hist = tf.summary.histogram("weights4", W4)
    b4_hist = tf.summary.histogram("biases4", b4)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

# Hypothesis (using softmax)
#hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

#cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
with tf.name_scope("cost"):
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    cost_summ = tf.summary.scalar("cost", cost)

with tf.name_scope("optimizer"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=10.0).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)

# parameters
training_epochs = 15  # 1 epoch는 전체 training data를 1번 읽음을 의미
batch_size = 100  # batch_size는 전체 training data 중 한번에 읽어올 양을 의미

with tf.Session() as sess:
    # tensorboard --logdir=./logs/xor_logs
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/mnist_01")
    writer.add_graph(sess.graph)  # Show the graph

    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)  # batch_size 크기만큼 몇 번 해야하는지를 의미

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={
                            X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

            summary, _ = sess.run([merged_summary, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            writer.add_summary(summary, global_step=i)

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)  # 테스트 데이터 중 임의로 하나를 가져옴
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))  # 실제값
    print("Prediction: ", sess.run(  # 예측값
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    plt.imshow(
        mnist.test.images[r:r + 1].reshape(28, 28),
        cmap='Greys',
        interpolation='nearest')
    plt.show()
