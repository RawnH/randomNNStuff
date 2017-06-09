#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 15:15:33 2017

@author: rawn
"""

import tensorflow as tf

W = tf.Variable(100.0, tf.float32)
b = tf.Variable(-100.0, tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = W*x + b
loss = tf.reduce_sum(tf.square(linear_model - y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train, feed_dict = {x : [1,2,3,4], y : [0, -1, -2, -3]})


print(sess.run([W,b]))
