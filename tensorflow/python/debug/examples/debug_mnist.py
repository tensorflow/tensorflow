# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Demo of the tfdbg curses CLI: Locating the source of bad numerical values.

The neural network in this demo is larged based on the tutorial at:
  tensorflow/examples/tutorials/mnist/mnist_with_summaries.py

But modifications are made so that problematic numerical values (infs and nans)
appear in nodes of the graph during training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("max_steps", 10, "Number of steps to run trainer.")
flags.DEFINE_integer("train_batch_size", 100,
                     "Batch size used during training.")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_string("data_dir", "/tmp/mnist_data", "Directory for storing data")
flags.DEFINE_boolean("debug", False,
                     "Use debugger to track down bad values during training")

IMAGE_SIZE = 28
HIDDEN_SIZE = 500
NUM_LABELS = 10
RAND_SEED = 42


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  def feed_dict(train):
    if train:
      xs, ys = mnist.train.next_batch(FLAGS.train_batch_size, fake_data=False)
    else:
      xs, ys = mnist.test.images, mnist.test.labels

    return {x: xs, y_: ys}

  sess = tf.InteractiveSession()

  # Create the MNIST neural network graph.

  # Input placeholders.
  with tf.name_scope("input"):
    x = tf.placeholder(
        tf.float32, [None, IMAGE_SIZE * IMAGE_SIZE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, NUM_LABELS], name="y-input")

  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1, seed=RAND_SEED)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer."""
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope("weights"):
        weights = weight_variable([input_dim, output_dim])
      with tf.name_scope("biases"):
        biases = bias_variable([output_dim])
      with tf.name_scope("Wx_plus_b"):
        preactivate = tf.matmul(input_tensor, weights) + biases

      activations = act(preactivate)
      return activations

  hidden = nn_layer(x, IMAGE_SIZE**2, HIDDEN_SIZE, "hidden")
  y = nn_layer(hidden, HIDDEN_SIZE, NUM_LABELS, "softmax", act=tf.nn.softmax)

  with tf.name_scope("cross_entropy"):
    # The following line is the culprit of the bad numerical values that appear
    # during training of this graph. Log of zero gives inf, which is first seen
    # in the intermediate tensor "cross_entropy/Log:0" during the 4th run()
    # call. A multiplication of the inf values with zeros leads to nans,
    # which is first in "cross_entropy/mul:0".
    #
    # You can use clipping to fix this issue, e.g.,
    #   diff = y_ * tf.log(tf.clip_by_value(y, 1e-8, 1.0))

    diff = y_ * tf.log(y)
    with tf.name_scope("total"):
      cross_entropy = -tf.reduce_mean(diff)

  with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope("accuracy"):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sess.run(tf.global_variables_initializer())

  if FLAGS.debug:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

  # Add this point, sess is a debug wrapper around the actual Session if
  # FLAGS.debug is true. In that case, calling run() will launch the CLI.
  for i in range(FLAGS.max_steps):
    acc = sess.run(accuracy, feed_dict=feed_dict(False))
    print("Accuracy at step %d: %s" % (i, acc))

    sess.run(train_step, feed_dict=feed_dict(True))


if __name__ == "__main__":
  tf.app.run()
