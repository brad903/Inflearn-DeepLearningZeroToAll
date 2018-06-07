# Lab 3 Minimizing Cost
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
tf.set_random_seed(777)  # for reproducibility

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)  # W는 값을 나중에 주기위해 placeholder 이용

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Launch the graph in a session.
sess = tf.Session()

# Variables for plotting cost function
# 그래프 생성을 위한 값을 넣어주기 위한 list 생성
W_history = []
cost_history = []

for i in range(-30, 50):
    curr_W = i * 0.1  # -3 ~ 5 까지 이동하면서 값 주어줌
    curr_cost = sess.run(cost, feed_dict={W: curr_W})
    W_history.append(curr_W)
    cost_history.append(curr_cost)

# Show the cost function
plt.plot(W_history, cost_history)  # W_history : x축, cost_history : y축
plt.show()