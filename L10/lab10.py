##


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from tqdm import tqdm_notebook as tqdm

import tensorflow as tf

##
# xor task
xor_data = np.array([[1, 0],
                    [0, 1],
                    [1, 1],
                    [0, 0]])
xor_label = np.array([[1], [1], [0], [0]])

##

tf.reset_default_graph()

with tf.Graph().as_default() as g:
    """ Define dataset and iterator """
    with tf.name_scope("data"):
        x = tf.placeholder(tf.float32, [None, 2])
        y = tf.placeholder(tf.float32, [None, 1])

    """ Build the model """
    with tf.name_scope("model"):
        W1 = tf.Variable(tf.random_normal([2,2],-1,1))
        W2 = tf.Variable(tf.random_normal([2, 1], -1, 1))
        b1 = tf.Variable(tf.random_uniform([2]))
        b2 = tf.Variable(tf.random_uniform([1]))

        z1 = tf.nn.tanh((x @ W1) + b1)
        z2 = tf.nn.sigmoid((z1 @ W2) + b2)
        p = z2



    """ Define the loss """
    with tf.name_scope("loss"):
        loss = -tf.reduce_mean(y * tf.log(p) + (1 - y) * tf.log(1 - p))

    """ Define the optimizer """
    with tf.name_scope("optimizer"):
        optim = tf.train.GradientDescentOptimizer(1e-2).minimize(loss)

    """ Other tensors or operations you need """
    with tf.name_scope("accuracy"):
        eq = tf.equal(tf.round(p), y)
        acc = tf.reduce_mean(tf.cast(eq, tf.float32))

with tf.Session(graph=g) as sess:
    """ Initialize the variables """
    """ Run the target tensors and operations """

    sess.run(tf.global_variables_initializer())

    loss_history = []
    acc_history = []
    for i in (range(100)):
        l, a, _ = sess.run([loss, acc, optim], feed_dict={x: xor_data, y: xor_label})
        if i % 10 == 0:
            loss_history.append(l)
            acc_history.append(a)
    #
    ns = ['W1', 'b1', 'W2', 'b2']
    ws = sess.run([W1, b1, W2, b2], feed_dict={x: xor_data, y: xor_label})
    np.set_printoptions(precision=1)
    for n, w in zip(ns, ws):
        print(n, ':')
        print(w)

##
fig, ax = plt.subplots(dpi=100, figsize=(8, 4))
ax.plot(np.arange(len(loss_history)), loss_history, '-', label='loss')
ax.plot(np.arange(len(acc_history)), acc_history, '-', label='acc')
ax.legend()
plt.show()

##

