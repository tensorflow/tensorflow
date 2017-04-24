#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""This is an example of using convolutional networks over characters for
   DBpedia dataset to predict class from description of an entity.

This model is similar to one described in this paper:
   "Character-level Convolutional Networks for Text Classification"
   http://arxiv.org/abs/1509.01626

and is somewhat alternative to the Lua code from here:
   https://github.com/zhangxiangxiao/Crepe
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf

learn = tf.contrib.learn

FLAGS = None

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, N_FILTERS]
POOLING_WINDOW = 4
POOLING_STRIDE = 2


def char_cnn_model(features, target):
  """Character level convolutional neural network model to predict classes."""
  target = tf.one_hot(target, 15, 1, 0)
  byte_list = tf.reshape(
      tf.one_hot(features, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1])
  with tf.variable_scope('CNN_Layer1'):
    # Apply Convolution filtering on input sequence.
    conv1 = tf.contrib.layers.convolution2d(
        byte_list, N_FILTERS, FILTER_SHAPE1, padding='VALID')
    # Add a ReLU for non linearity.
    conv1 = tf.nn.relu(conv1)
    # Max pooling across output of Convolution+Relu.
    pool1 = tf.nn.max_pool(
        conv1,
        ksize=[1, POOLING_WINDOW, 1, 1],
        strides=[1, POOLING_STRIDE, 1, 1],
        padding='SAME')
    # Transpose matrix so that n_filters from convolution becomes width.
    pool1 = tf.transpose(pool1, [0, 1, 3, 2])
  with tf.variable_scope('CNN_Layer2'):
    # Second level of convolution filtering.
    conv2 = tf.contrib.layers.convolution2d(
        pool1, N_FILTERS, FILTER_SHAPE2, padding='VALID')
    # Max across each filter to get useful features for classification.
    pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])

  # Apply regular WX + B and classification.
  logits = tf.contrib.layers.fully_connected(pool2, 15, activation_fn=None)
  loss = tf.losses.softmax_cross_entropy(target, logits)

  train_op = tf.contrib.layers.optimize_loss(
      loss,
      tf.contrib.framework.get_global_step(),
      optimizer='Adam',
      learning_rate=0.01)

  return ({
      'class': tf.argmax(logits, 1),
      'prob': tf.nn.softmax(logits)
  }, loss, train_op)


def main(unused_argv):
  # Prepare training and testing data
  dbpedia = learn.datasets.load_dataset(
      'dbpedia', test_with_fake_data=FLAGS.test_with_fake_data, size='large')
  x_train = pandas.DataFrame(dbpedia.train.data)[1]
  y_train = pandas.Series(dbpedia.train.target)
  x_test = pandas.DataFrame(dbpedia.test.data)[1]
  y_test = pandas.Series(dbpedia.test.target)

  # Process vocabulary
  char_processor = learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(char_processor.fit_transform(x_train)))
  x_test = np.array(list(char_processor.transform(x_test)))

  # Build model
  classifier = learn.Estimator(model_fn=char_cnn_model)

  # Train and predict
  classifier.fit(x_train, y_train, steps=100)
  y_predicted = [
      p['class'] for p in classifier.predict(
          x_test, as_iterable=True)
  ]
  score = metrics.accuracy_score(y_test, y_predicted)
  print('Accuracy: {0:f}'.format(score))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--test_with_fake_data',
      default=False,
      help='Test the example code with fake data.',
      action='store_true')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
