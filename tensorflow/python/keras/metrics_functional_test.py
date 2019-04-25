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

from tensorflow.python.eager import context
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import metrics
from tensorflow.python.platform import test


class KerasFunctionalMetricsTest(test.TestCase):

  def test_metrics(self):
    with self.cached_session():
      y_a = K.variable(np.random.random((6, 7)))
      y_b = K.variable(np.random.random((6, 7)))
      for metric in [metrics.binary_accuracy, metrics.categorical_accuracy]:
        output = metric(y_a, y_b)
        self.assertEqual(K.eval(output).shape, (6,))

  def test_sparse_categorical_accuracy_int(self):
    with self.cached_session():
      metric = metrics.sparse_categorical_accuracy
      y_true = K.variable(np.random.randint(0, 7, (6,)))
      y_pred = K.variable(np.random.random((6, 7)))
      self.assertEqual(K.eval(metric(y_true, y_pred)).shape, (6,))

      # Test correctness if the shape of y_true is (num_samples,)
      y_true = K.variable([1., 0., 0., 0.])
      y_pred = K.variable([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3], [0.9, 0.1]])
      print(K.eval(metric(y_true, y_pred)))
      self.assertAllEqual(K.eval(metric(y_true, y_pred)), [0., 1., 1., 1.])

      # Test correctness if the shape of y_true is (num_samples, 1)
      y_true = K.variable([[1.], [0.], [0.], [0.]])
      y_pred = K.variable([[0.8, 0.2], [0.6, 0.4], [0.7, 0.3], [0.9, 0.1]])
      print(K.eval(metric(y_true, y_pred)))
      self.assertAllEqual(K.eval(metric(y_true, y_pred)), [0., 1., 1., 1.])

  def test_sparse_categorical_accuracy_float(self):
    with self.cached_session():
      metric = metrics.sparse_categorical_accuracy
      y_true = K.variable(np.random.random((6,)))
      y_pred = K.variable(np.random.random((6, 7)))
      self.assertEqual(K.eval(metric(y_true, y_pred)).shape, (6,))

  def test_sparse_categorical_accuracy_eager(self):
    """Tests that ints passed in via Eager return results. See b/113504761."""
    with context.eager_mode():
      metric = metrics.sparse_categorical_accuracy
      y_true = np.arange(6).reshape([6, 1])
      y_pred = np.arange(36).reshape([6, 6])
      self.assertAllEqual(metric(y_true, y_pred), [0., 0., 0., 0., 0., 1.])

  def test_sparse_categorical_accuracy_float_eager(self):
    """Tests that floats passed in via Eager return results. See b/113504761."""
    with context.eager_mode():
      metric = metrics.sparse_categorical_accuracy
      y_true = np.arange(6, dtype=np.float32).reshape([6, 1])
      y_pred = np.arange(36).reshape([6, 6])
      self.assertAllEqual(metric(y_true, y_pred), [0., 0., 0., 0., 0., 1.])

  def test_sparse_top_k_categorical_accuracy(self):
    with self.cached_session():
      # Test correctness if the shape of y_true is (num_samples, 1)
      y_pred = K.variable(np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.7]]))
      y_true = K.variable(np.array([[1], [0]]))
      result = K.eval(
          metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=3))
      self.assertEqual(result, 1)
      result = K.eval(
          metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=2))
      self.assertEqual(result, 0.5)
      result = K.eval(
          metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=1))
      self.assertEqual(result, 0.)

      # Test correctness if the shape of y_true is (num_samples,)
      y_pred = K.variable(np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.7]]))
      y_true = K.variable(np.array([1, 0]))
      result = K.eval(
          metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=3))
      self.assertEqual(result, 1)
      result = K.eval(
          metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=2))
      self.assertEqual(result, 0.5)
      result = K.eval(
          metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=1))
      self.assertEqual(result, 0.)

  def test_top_k_categorical_accuracy(self):
    with self.cached_session():
      y_pred = K.variable(np.array([[0.3, 0.2, 0.1], [0.1, 0.2, 0.7]]))
      y_true = K.variable(np.array([[0, 1, 0], [1, 0, 0]]))
      result = K.eval(metrics.top_k_categorical_accuracy(y_true, y_pred, k=3))
      self.assertEqual(result, 1)
      result = K.eval(metrics.top_k_categorical_accuracy(y_true, y_pred, k=2))
      self.assertEqual(result, 0.5)
      result = K.eval(metrics.top_k_categorical_accuracy(y_true, y_pred, k=1))
      self.assertEqual(result, 0.)


if __name__ == '__main__':
  test.main()
