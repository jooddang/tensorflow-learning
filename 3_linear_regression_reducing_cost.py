import tensorflow as tf
import matplotlib.pyplot as plt


def linear_regression_plotting_cost_function():
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


def linear_regression_gradient_descent():
    x_data = [1, 2, 3]
    y_data = [1, 2, 3]

    # test with different initial W values
    # W = tf.Variable(tf.random_normal([1]), name="weight")
    # W = tf.Variable(5.0)
    W = tf.Variable(-5.0)

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    learning_rate = 0.1
    hypothesis = W * X
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # manually find gradient and update W
    gradient = tf.reduce_mean((W * X - Y) * X)
    descent = W - learning_rate * gradient
    update = W.assign(descent)

    # Use optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    # train = optimizer.minimize(cost)
    # it works the same way as `optimizer.minimize` does
    gvs = optimizer.compute_gradients(cost)
    apply_gradients = optimizer.apply_gradients(gvs)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # compare manual work 
    print('==========session 1')
    for step in range(21):
        sess.run(update, feed_dict={X: x_data, Y: y_data})
        print(step, sess.run(W), sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    
    sess2 = tf.Session()
    sess2.run(tf.global_variables_initializer())

    # to optimizer
    print('==========session 2')
    for step in range(21):
        print(step, sess2.run([gradient, W, gvs], feed_dict={X: x_data, Y: y_data}))
        sess2.run(apply_gradients, feed_dict={X: x_data, Y: y_data})


# linear_regression_gradient_descent()


def manual_compute_gradients():
    X = [1, 2, 3]
    Y = [1, 2, 3]

    W = tf.Variable(5.)
    hypothesis = W * X
    gradient = tf.reduce_mean((W * X - Y) * X) * 2
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    gvs = optimizer.compute_gradients(cost)
    apply_gradients = optimizer.apply_gradients(gvs)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        print(step, sess.run([gradient, W, gvs]))
        sess.run(apply_gradients)


manual_compute_gradients()
