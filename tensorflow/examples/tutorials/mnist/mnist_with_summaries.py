# Copyright 2015 Google Inc. All Rights Reserved.
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

"""A very simple MNIST classifier, modified to display data in TensorBoard.

See extensive documentation for the original model at
http://tensorflow.org/tutorials/mnist/beginners/index.md

See documentation on the TensorBoard specific pieces at
http://tensorflow.org/how_tos/summaries_and_tensorboard/index.md

If you modify this file, please update the excerpt in
how_tos/summaries_and_tensorboard/index.md.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/mnist_logs', 'Summaries directory')


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True,
                                    fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784], name='x-input')
  W = tf.Variable(tf.zeros([784, 10]), name='weights')
  b = tf.Variable(tf.zeros([10]), name='bias')

  # Use a name scope to organize nodes in the graph visualizer
  with tf.name_scope('Wx_b'):
    y = tf.nn.softmax(tf.matmul(x, W) + b)

  # Add summary ops to collect data
  tf.histogram_summary('weights', W)
  tf.histogram_summary('biases', b)
  tf.histogram_summary('y', y)

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
  # More name scopes will clean up the graph representation
  with tf.name_scope('xent'):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    tf.scalar_summary('cross entropy', cross_entropy)
  with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(
        FLAGS.learning_rate).minimize(cross_entropy)

  with tf.name_scope('test'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged = tf.merge_all_summaries()
  writer = tf.train.SummaryWriter(FLAGS.summaries_dir,
                                  sess.graph.as_graph_def(add_shapes=True))
  tf.initialize_all_variables().run()

  # Train the model, and feed in test data and record summaries every 10 steps

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summary data and the accuracy
      if FLAGS.fake_data:
        batch_xs, batch_ys = mnist.train.next_batch(
            100, fake_data=FLAGS.fake_data)
        feed = {x: batch_xs, y_: batch_ys}
      else:
        feed = {x: mnist.test.images, y_: mnist.test.labels}
      result = sess.run([merged, accuracy], feed_dict=feed)
      summary_str = result[0]
      acc = result[1]
      writer.add_summary(summary_str, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:
      batch_xs, batch_ys = mnist.train.next_batch(
          100, fake_data=FLAGS.fake_data)
      feed = {x: batch_xs, y_: batch_ys}
      sess.run(train_step, feed_dict=feed)

if __name__ == '__main__':
  tf.app.run()
