import numpy as np
import tensorflow as tf


def primitive_multi_var():
    x1_data = [58.0, 52.0, 53.0, 53.0, 56.0]
    x2_data = [100.0, 64.0, 51.0, 93.0, 79.0]
    x3_data = [51.0, 58.0, 63.0, 74.0, 85.0]
    y_data = [187.0, 184.0, 166.0, 168.0, 169.0]

    x1 = tf.placeholder(tf.float32)
    x2 = tf.placeholder(tf.float32)
    x3 = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    w1 = tf.Variable(tf.random_normal([1]), name='weight1')
    w2 = tf.Variable(tf.random_normal([1]), name='weight2')
    w3 = tf.Variable(tf.random_normal([1]), name='weight3')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(4001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                       feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
        if step % 100 == 0:
            print(step, "cost: ", cost_val, "prediction: ", hy_val)
            # 4000 cost:  61.5923
            # prediction:  [ 191.70774841  169.1756897   171.99211121  165.69598389  173.98240662]


def matrix_multi_var():
    x_data = [[58.0, 100.0, 51.0],
              [52.0, 64.0, 58.0],
              [53.0, 51.0, 63.0],
              [53.0, 93.0, 74.0],
              [56.0, 79.0, 85.0]]
    y_data = [[187.0], [184.0], [166.0], [168.0], [169.0]]

    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([3, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    hypothesis = tf.matmul(X, W) + b

    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(4001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                       feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step, "cost: ", cost_val, "prediction: ", hy_val)
            # 4000 cost:  60.7822
            # prediction:  [[ 191.85115051]
            #               [ 169.44267273]
            #               [ 172.31044006]
            #               [ 165.4127655 ]
            #               [ 173.68466187]]


# matrix_multi_var()


def multi_var_from_file():
    xy = np.loadtxt('4_multi_var_from_file.csv', delimiter=',', dtype=np.float32)
    x_data = xy[:, 0:3]
    y_data = xy[..., [-1]]

    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([3, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    hypothesis = tf.matmul(X, W) + b

    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # train
    for step in range(4001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                       feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step, "cost: ", cost_val, "prediction: ", hy_val)
            # 4000 cost:  60.7822
            # prediction:  [[ 191.85115051]
            #               [ 169.44267273]
            #               [ 172.31044006]
            #               [ 165.4127655 ]
            #               [ 173.68466187]]

    # test
    print("my score will be: ", sess.run(hypothesis, feed_dict={X: [[100, 100, 100]]}))
    print("Other scores will be: ", sess.run(hypothesis,
                                             feed_dict={X: [[60, 70, 80], [55, 94, 29]]}))


# multi_var_from_file()


def multi_var_from_file_batch():
    # When you use string_input_producer(), do not insert #comments in csv files.
    # It will spit inefficient data error.
    filename_queue = tf.train.string_input_producer(
        ['/Users/eunkwang/Documents/tensorflow-learning/4_multi_var_from_file.csv',
         '/Users/eunkwang/Documents/tensorflow-learning/4_multi_var_from_file_2.csv'],
        shuffle=False)

    reader = tf.TextLineReader()
    _, value = reader.read(filename_queue)

    # in case of empty columns, and also specify type of the result
    record_defaults = [[0.], [0.], [0.], [0.]]
    xy = tf.decode_csv(value, record_defaults=record_defaults)
    train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=2)

    X = tf.placeholder(tf.float32, shape=[None, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([3, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')
    hypothesis = tf.matmul(X, W) + b

    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # start populating filename queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # train
    for step in range(4001):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                       feed_dict={X: x_batch, Y: y_batch})
        if step % 100 == 0:
            print(step, "cost: ", cost_val, "prediction: ", hy_val)
            # 4000 cost:  60.7822
            # prediction:  [[ 191.85115051]
            #               [ 169.44267273]
            #               [ 172.31044006]
            #               [ 165.4127655 ]
            #               [ 173.68466187]]

    coord.request_stop()
    coord.join(threads)

    # test
    print("my score will be: ", sess.run(hypothesis, feed_dict={X: [[100, 100, 100]]}))
    print("Other scores will be: ", sess.run(hypothesis,
                                             feed_dict={X: [[60, 70, 80], [55, 94, 29]]}))


multi_var_from_file_batch()
