import tensorflow as tf


n_input = 1000
n_classes = 1
learning_rate = 0.001

weights = {
    'fc1': tf.Variable(tf.random_normal([n_input, 100])),
    'fc2': tf.Variable(tf.random_normal([100, 10])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([10, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([100])),
    'bc2': tf.Variable(tf.random_normal([10])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}




def generate_batches(inputs):
    return [[0]*l + [1] + [0] * (inputs-l-1) for l in range(inputs)], [[i] for i in range(inputs)]



batch_x, batch_y = generate_batches(n_input)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


def nn(x, weights, biases):
    x = tf.reshape(x, [-1, weights['fc1'].get_shape().as_list()[0]])

    fc1 = tf.add(tf.matmul(x, weights['fc1']), biases['bc1'])
    fc1 = tf.nn.relu(fc1)

    fc2 = tf.add(tf.matmul(fc1, weights['fc2']), biases['bc2'])
    fc2 = tf.nn.relu(fc2)

    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out




# Construct model
pred = nn(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_sum(tf.square(pred-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = pred - y
accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 0
    # Keep training until reach max iterations
    while step * 4 < 20000:
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % 100 == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
            print(loss)
            print("Iter " + str(step*3) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    print("Testing Accuracy:",
        sess.run(accuracy, feed_dict={x: batch_x,
                                      y: batch_y}))

    while True:
        strinput = input("Position: ")
        listInput = [0]*int(strinput) + [1] + [0]*(n_input - int(strinput) - 1)
        nn_prediction = nn(tf.cast(listInput, tf.float32), weights, biases).eval()[0][0]
        print("I guess that the 1 is on position %f"%nn_prediction)
