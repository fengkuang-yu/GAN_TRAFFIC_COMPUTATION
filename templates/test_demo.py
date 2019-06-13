#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:lenovo
# datetime:2019/6/11 17:22
# software: PyCharm

import numpy as np
import tensorflow as tf


sess = tf.Session()

x_input = tf.placeholder(tf.float32, name='x_input')
y_input = tf.placeholder(tf.float32, name='y_input')
w = tf.Variable([2.0], name='weight')
b = tf.Variable(1.0, name='biases')
y = tf.add(tf.multiply(x_input, w), b)
loss_op = tf.reduce_sum(tf.pow(y_input - y, 2)) / (2 * 32)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss_op)

'''tensorboard'''
gradients_node = tf.gradients(loss_op, w)[0]
print(gradients_node)

init = tf.global_variables_initializer()
sess.run(init)

'''构造数据集'''
x_pure = np.random.randint(-10, 100, 32)
x_train = x_pure + np.random.randn(32) / 10  # 为x加噪声
y_train = 3 * x_pure + 2 + np.random.randn(32) / 10  # 为y加噪声

for i in range(20):
    _, gradients, loss = sess.run([train_op, gradients_node, loss_op],
                                  feed_dict={x_input: x_train[i], y_input: y_train[i]})
    print("epoch: {} \t loss: {} \t gradients: {}".format(i, loss, gradients))

sess.close()