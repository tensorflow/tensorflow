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

"""RNN estimator tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators._sklearn import accuracy_score
from tensorflow.contrib.learn.python.learn.estimators._sklearn import mean_squared_error


def rnn_input_fn(x):
  return tf.split(1, 5, x)


data = np.array(list([[2, 1, 2, 2, 3],
                      [2, 2, 3, 4, 5],
                      [3, 3, 1, 2, 1],
                      [2, 4, 5, 4, 1]]),
                dtype=np.float32)
# labels for classification
labels = np.array(list([1, 0, 1, 0]), dtype=np.float32)
# targets for regression
targets = np.array(list([10, 16, 10, 16]), dtype=np.float32)
test_data = np.array(list([[1, 3, 3, 2, 1],
                           [2, 3, 4, 5, 6]]),
                     dtype=np.float32)


class RNNTest(tf.test.TestCase):
  """RNN estimator tests."""

  def setUp(self):
    random.seed(42)
    tf.set_random_seed(42)

  def testRNN(self):
    # Classification
    classifier = tf.contrib.learn.TensorFlowRNNClassifier(rnn_size=2,
                                                          cell_type="lstm",
                                                          n_classes=2,
                                                          steps=150,
                                                          input_op_fn=rnn_input_fn)
    classifier.fit(data, labels)
    # pylint: disable=pointless-statement
    classifier.weights_
    classifier.bias_
    # pylint: enable=pointless-statement
    predictions = classifier.predict(data[:2])
    self.assertAllClose(predictions, labels[:2])

    classifier = tf.contrib.learn.TensorFlowRNNClassifier(rnn_size=2,
                                                          cell_type="rnn",
                                                          n_classes=2,
                                                          input_op_fn=rnn_input_fn,
                                                          steps=100,
                                                          num_layers=2)
    classifier.fit(data, labels)
    classifier = tf.contrib.learn.TensorFlowRNNClassifier(
        rnn_size=2, cell_type="invalid_cell_type", n_classes=2,
        input_op_fn=rnn_input_fn, num_layers=2)
    with self.assertRaises(ValueError):
      classifier.fit(data, labels)

    # Regression
    regressor = tf.contrib.learn.TensorFlowRNNRegressor(rnn_size=2,
                                                        cell_type="gru",
                                                        steps=100,
                                                        input_op_fn=rnn_input_fn)
    regressor.fit(data, targets)
    # pylint: disable=pointless-statement
    regressor.weights_
    regressor.bias_
    # pylint: enable=pointless-statement
    predictions = regressor.predict(test_data)

    # rnn with attention
    classifier = tf.contrib.learn.TensorFlowRNNClassifier(rnn_size=2,
                                                          cell_type="lstm",
                                                          n_classes=2,
                                                          input_op_fn=rnn_input_fn,
                                                          bidirectional=False,
                                                          attn_length=2,
                                                          steps=100,
                                                          attn_size=2,
                                                          attn_vec_size=2)
    classifier.fit(data, labels)
    predictions = classifier.predict(data[:2])
    self.assertAllClose(predictions, labels[:2])

  def testBidirectionalRNN(self):
    # Classification
    classifier = tf.contrib.learn.TensorFlowRNNClassifier(rnn_size=2,
                                                          cell_type="lstm",
                                                          n_classes=2,
                                                          input_op_fn=rnn_input_fn,
                                                          steps=100,
                                                          bidirectional=True)
    classifier.fit(data, labels)
    predictions = classifier.predict(data[:2])
    self.assertAllClose(predictions, labels[:2])

    # bidirectional rnn with attention
    classifier = tf.contrib.learn.TensorFlowRNNClassifier(rnn_size=2,
                                                          cell_type="lstm",
                                                          n_classes=2,
                                                          input_op_fn=rnn_input_fn,
                                                          bidirectional=True,
                                                          attn_length=2,
                                                          attn_size=2,
                                                          steps=100,
                                                          attn_vec_size=2)
    classifier.fit(data, labels)
    predictions = classifier.predict(data[:2])
    self.assertAllClose(predictions, labels[:2])

if __name__ == "__main__":
  tf.test.main()
