# coding: utf-8
import tensorflow as tf

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = W * X + b
cost = tf.reduce_min(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)
s = tf.Session()
s.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, W_val, b_val, _ = s.run([cost, W, b, train], feed_dict={X: [1,2,3], Y: [1,2,3]})
    if step % 100 == 0:
        print(step, cost_val, W_val, b_val)
        
