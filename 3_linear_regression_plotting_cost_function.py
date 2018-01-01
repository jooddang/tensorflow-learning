import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)
hypothesis = X * W
cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

w_vals = []
cost_vals = []

# By changing W value in a range, 
# we wanna see how W and cost function value changes in each step.
for i in range(-30, 50):
    W_feeding = i * 0.1
    current_W, current_cost = sess.run([W, cost], feed_dict={W: W_feeding})
    w_vals.append(current_W)
    cost_vals.append(current_cost)

plt.plot(w_vals, cost_vals)
plt.show()
