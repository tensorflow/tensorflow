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
"""Tests for Keras metrics functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras._impl import keras
from tensorflow.python.platform import test


class KerasMetricsTest(test.TestCase):

  def test_metrics(self):
    with self.test_session():
      y_a = keras.backend.variable(np.random.random((6, 7)))
      y_b = keras.backend.variable(np.random.random((6, 7)))
      for metric in [keras.metrics.binary_accuracy,
                     keras.metrics.categorical_accuracy]:
        output = metric(y_a, y_b)
        self.assertEqual(keras.backend.eval(output).shape, (6,))

  def test_sparse_categorical_accuracy(self):
    with self.test_session():
      metric = keras.metrics.sparse_categorical_accuracy
      y_a = keras.backend.variable(np.random.randint(0, 7, (6,)))
      y_b = keras.backend.variable(np.random.random((6, 7)))
      self.assertEqual(keras.backend.eval(metric(y_a, y_b)).shape, (6,))

  def test_sparse_top_k_categorical_accuracy(self):
    with self.test_session():
      y_pred = keras.backend.variable(np.array([[0.3, 0.2, 0.1],
                                                [0.1, 0.2, 0.7]]))
      y_true = keras.backend.variable(np.array([[1], [0]]))
      result = keras.backend.eval(
          keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=3))
      self.assertEqual(result, 1)
      result = keras.backend.eval(
          keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=2))
      self.assertEqual(result, 0.5)
      result = keras.backend.eval(
          keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=1))
      self.assertEqual(result, 0.)

  def test_top_k_categorical_accuracy(self):
    with self.test_session():
      y_pred = keras.backend.variable(np.array([[0.3, 0.2, 0.1],
                                                [0.1, 0.2, 0.7]]))
      y_true = keras.backend.variable(np.array([[0, 1, 0], [1, 0, 0]]))
      result = keras.backend.eval(
          keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3))
      self.assertEqual(result, 1)
      result = keras.backend.eval(
          keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2))
      self.assertEqual(result, 0.5)
      result = keras.backend.eval(
          keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=1))
      self.assertEqual(result, 0.)


if __name__ == '__main__':
  test.main()
