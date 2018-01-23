# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Recurrent Neural Network.
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/).
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf).
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

def rnn_network(x_in, weight_set, bias_set):
    """To classify images using a recurrent neural network, we consider every image
    row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
    handle 28 sequences of 28 steps for every sample.
    Args:
        x_in: input images.
        weight_set: collection of input/ output weight.
        bias_set: collection input/ output bias.
    Returns:
        Output tensor with the computed logits.
    """

    # Prepare data shape to match `rnn` function requirements.
    # Current data input shape: (BATCH_SIZE, num_step, n_input).
    # Required shape: 'num_step' tensors list of shape (BATCH_SIZE, n_input).
    x_in = tf.reshape(x_in, [-1, NUM_INPUT])

    # Hidden layer ==> (128 batch, 28 steps, 128 hidden).
    x_hidden = tf.matmul(x_in, weight_set['in']) + bias_set['in']
    x_hidden = tf.reshape(x_hidden, [-1, NUM_STEP, NUM_HIDDEN_UNITS])

    # Define a lstm cell with tensorflow.
    lstm_cell = rnn.BasicLSTMCell(NUM_HIDDEN_UNITS, forget_bias=1.0, state_is_tuple=True)

    # init zero state, lstm cell is consist of twp parts: (c_state, h_state).
    init_state = lstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)

    # Get lstm cell output.
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x_hidden, \
        initial_state=init_state, time_major=False, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output.
    return tf.matmul(states[1], weight_set['out']) + bias_set['out']

if __name__ == '__main__':
    # Import MNIST data.
    MNIST = input_data.read_data_sets('/tmp/mnist_data/', one_hot=True)

    # Training Parameters.
    LEARNING_RATE = 0.001
    TRAINING_STEPS = 10000
    BATCH_SIZE = 128
    DISPLAY_STEP = 200

    # Network Parameters.
    NUM_INPUT = 28 # MNIST data input (img shape: 28*28)
    NUM_STEP = 28 # step number
    NUM_HIDDEN_UNITS = 128 # hidden layer num of features
    NUM_CLASSES = 10 # MNIST total classes (0-9 digits)

    # tf Graph input.
    input_images = tf.placeholder('float', shape=[None, NUM_STEP, NUM_INPUT])
    input_classes = tf.placeholder('float', shape=[None, NUM_CLASSES])

    # Define weights.
    weights = {
        # (28, 128)
        'in': tf.Variable(tf.random_normal([NUM_INPUT, NUM_HIDDEN_UNITS])),
        # (128, 10)
        'out': tf.Variable(tf.random_normal([NUM_HIDDEN_UNITS, NUM_CLASSES]))
    }
    biases = {
        # (128, )
        'in': tf.Variable(tf.constant(0.1, shape=[NUM_HIDDEN_UNITS, ])),
        # (10, )
        'out': tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES, ]))
    }

    logits = rnn_network(input_images, weights, biases)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer.
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=input_classes))
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    # Evaluate model (with test logits, for dropout to be disabled).
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(input_classes, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value).
    init = tf.global_variables_initializer()

    # Start training.
    with tf.Session() as sess:

        # Run the initializer.
        sess.run(init)

        for step in range(1, TRAINING_STEPS+1):
            batch_x, batch_y = MNIST.train.next_batch(BATCH_SIZE)
            # Reshape data to get 28 seq of 28 elements.
            batch_x = batch_x.reshape((BATCH_SIZE, NUM_STEP, NUM_INPUT))
            # Run optimization op (backprop).
            sess.run(train_op, feed_dict={input_images: batch_x, input_classes: batch_y})
            if step % DISPLAY_STEP == 0 or step == 1:
                # Calculate batch loss and accuracy.
                loss, acc = sess.run([cost, accuracy], \
                    feed_dict={input_images: batch_x, input_classes: batch_y})
                print('Step ' + str(step) + ', Minibatch Loss= ' + \
                      '{:.4f}'.format(loss) + ', Training Accuracy= ' + \
                      '{:.3f}'.format(acc))

        print('Training Finished!')

        # Calculate accuracy for 128 mnist test images.
        test_len = 128
        test_data = MNIST.test.images[:test_len].reshape((-1, NUM_STEP, NUM_INPUT))
        test_label = MNIST.test.labels[:test_len]
        print('Testing Accuracy:', \
            sess.run(accuracy, feed_dict={input_images: test_data, input_classes: test_label}))
