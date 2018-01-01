# coding: utf-8
import tensorflow as tf

x_train = [1,2,3]
y_train = [1,2,3]
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = W * x_train + b
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)
s = tf.Session()
s.run(tf.global_variables_initializer())
for step in range(2001):
    s.run(train)
    if step % 100 == 0:
        print(step, s.run(cost), s.run(W), s.run(b))
        
