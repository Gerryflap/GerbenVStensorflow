import tensorflow as tf


n_input = 7
n_classes = 2
learning_rate = 0.001

weights = {
    'fc1': tf.Variable(tf.random_normal([n_input, 5])),
    'fc2': tf.Variable(tf.random_normal([5, 5])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([5, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([5])),
    'bc2': tf.Variable(tf.random_normal([5])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def get_bin(n, l):
    li = [int(x) for x in bin(n)[2:]]
    li = [0]*(l-len(li)) + li
    return li


def generate_input_batch(n):
    return [get_bin(y, n) for y in range(2**n)]


def output(batch_in):
    out = []
    for inp in batch_in:
        outp = inp[0] & inp[1] ^ inp[2] | (inp[3] & inp[4] & inp[5]) & ((inp[6]+1)%2)
        out.append([outp, (outp+1) % 2])
    return out

batch_x = generate_input_batch(n_input)
batch_y = output(batch_x)

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
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 0
    # Keep training until reach max iterations
    while step * 4 < 300000:
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % 100 == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
            print("Iter " + str(step*3) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    print("Testing Accuracy:",
        sess.run(accuracy, feed_dict={x: batch_x,
                                      y: batch_y}))
