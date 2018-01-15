import tensorflow as tf


def logistic_regression_lab_1():
    x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
    y_data = [[0], [0], [0], [1], [1], [1]]

    X = tf.placeholder(tf.float32, [None, 2])
    Y = tf.placeholder(tf.float32, [None, 1])

    W = tf.Variable(tf.random_normal([2, 1]), dtype=tf.float32)
    b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

    # hypothesis = tf.div(1., 1. + tf.exp(tf.matmul(W, X)))
    hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(10001):
            sess.run(train, feed_dict={X: x_data, Y: y_data})

        h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
        print('hypothesis: ', h, 'predict: ', p, 'accuracy: ', a)


def diabetes():
    _, input_data = tf.TextLineReader().read(tf.train.string_input_producer(['5_diabetes.csv']))
    xy = tf.decode_csv(input_data, [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]])
    x_batch, y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=50)

    X = tf.placeholder(tf.float32, shape=[None, 8])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([8, 1]))
    b = tf.Variable(tf.random_normal([1]))

    hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for step in range(10001):
            x_data, y_data = sess.run([x_batch, y_batch])
            sess.run(train, feed_dict={X: x_data, Y: y_data})

            if step % 500 == 0:
                print('cost: ', sess.run(cost, feed_dict={X: x_data, Y: y_data}))

        x_data, y_data = sess.run([x_batch, y_batch])
        h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
        print('hypothesis: ', h, 'predict: ', p, 'accuracy: ', a)

        coord.request_stop()
        coord.join(threads)


diabetes()
