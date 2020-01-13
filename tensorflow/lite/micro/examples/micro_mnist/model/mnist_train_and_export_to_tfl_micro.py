# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""A simple MNIST classifier which displays summaries in TensorBoard and
exports. the header and source files neccesary to be included in a TenforFlow
lite micro project.

This code also exports a small set of 25 test samples to a header file which is
used to test the model on embedded systems.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np

import tensorflow as tf
import fetch_mnist
import layers
import progress_bar as pb
import flatbuffer_2_tfl_micro as save_tflm

def write_test_data_header(file_name, x, y):
  """
  Function to write a c header file containing a set of test input and output
  data for this classification model
  :param file_name: name of the header file to create
  :param x: input data numpy array
  :param y: output data numpy array, first dimension much match above
  :return: Nothing
  """

  with open(file_name, "w") as header:
    header.write("// MNIST test data\n\n")

    header.write("int mnistSampleCount = %d;\n\n" % x.shape[0])

    header.write("float mnistInput[%d][784] = {\n" % x.shape[0])
    for i in range(x.shape[0]):
      if i != 0:
        header.write(",\n")
      header.write("{ ")
      row = x[i].reshape(1, 784).astype(np.int)
      np.savetxt(header, row, delimiter=', ', newline='', fmt='%d')
      header.write(" }")
    header.write("};\n\n")

    header.write("int mnistOutput[%d] = { " % y.shape[0])
    np.savetxt(header,
               y.reshape(1, y.shape[0]),
               delimiter=', ',
               newline='',
               fmt='%d')
    header.write(" };\n")

def train():
  """
  Function to build, train and export a simple MNIST classifation model
  :return: Nothing
  """

  print("Getting training data")
  mnist = fetch_mnist.MNISTDataset()
  print("Done")

  # ----------------------------------------------------------------------------
  # Build the model. Three fully connected layers, cross entropy loss fn.
  # ----------------------------------------------------------------------------
  sess = tf.compat.v1.InteractiveSession()

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28], name='x-input')
    y_ = tf.compat.v1.placeholder(tf.int64, [None], name='y-input')
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.compat.v1.summary.image('input', image_shaped_input, 10)

  # Create simple three layer fully connected network.
  x_flat = tf.reshape(x, [-1, 784], name='x_flat')
  hidden1 = layers.nn_layer(x_flat, 784, 25, 'layer1', act=tf.nn.relu)
  hidden2 = layers.nn_layer(hidden1, 25, 25, 'layer1', act=tf.nn.relu)
  y = layers.nn_layer(hidden2, 25, 10, 'layer2', act=tf.identity)
  classifcation = tf.compat.v1.argmax(y,
                                      name='classification',
                                      axis=1,
                                      output_type=tf.dtypes.int32)

  print("classification tensor shape is [%s]" % classifcation.shape)

  # Add training operations
  [train_step, accuracy] = layers.cross_entropy_training(y, y_, FLAGS.learning_rate)

  # Merge all the summaries and write them out to
  # /tmp/tensorflow/mnist/logs/mnist_with_summaries (for TensorBoard)
  merged = tf.compat.v1.summary.merge_all()
  train_writer = tf.compat.v1.summary.FileWriter(FLAGS.log_dir + '/train',
                                                 sess.graph)
  test_writer = tf.compat.v1.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.compat.v1.global_variables_initializer().run()

  # ----------------------------------------------------------------------------
  # Train the model, and write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries
  # ----------------------------------------------------------------------------
  def feed_dict(train_d):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train_d:
      xs, ys = mnist.get_batch(100)
    else:
      xs, ys = mnist.x_test, mnist.y_test
    return {x: xs, y_: ys}

  acc = 0
  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy],
                              feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      pb.update_progress_bar(i / float(FLAGS.max_steps),
                             pre_msg=' Training MNIST Classifier',
                             post_msg='Accuracy is %s' % acc, size=40)
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  pb.update_progress_bar(1.0,
                         pre_msg=' Training MNIST Classifier',
                         post_msg='Accuracy is %s' % acc, size=40,
                         c_return='\n')
  train_writer.close()
  test_writer.close()

  # ----------------------------------------------------------------------------
  # Export Inference model TFlite flatbuffer as a cpp and hpp file pair for
  # use by TFL micro
  # ----------------------------------------------------------------------------

  # The following line is disabled because the output needs post
  # processing to conform to Google standards, this prevents the corrected
  # version being overwritten by mistake
  # write_test_data_header("mnist_test_data.h",
  #                        mnist.x_test[0:25],
  #                        mnist.y_test[0:25])
  print("Saving TFLite flatbuffer for use with TFLite Micro.")
  converter = tf.lite.TFLiteConverter.from_session(sess, [x], [classifcation])

  save_tflm.write_tf_lite_micro_model(
      converter.convert(),
      base_file_name="mnist_model",
      data_variable_name="mnist_dense_model_tflite",
      header_comment="Example MNIST classification model,"
                     "for use with TFlite Micro")
  print("Complete")


def main(_):
  if tf.io.gfile.exists(FLAGS.log_dir):
    tf.io.gfile.rmtree(FLAGS.log_dir)
  tf.io.gfile.makedirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--max_steps', type=int, default=10000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/mnist_with_summaries'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
