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
"""Tests for Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import bijectors
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.distributions.bijector_test_util import assert_scalar_congruency
from tensorflow.python.platform import test


class CholeskyOuterProductBijectorTest(test.TestCase):
  """Tests the correctness of the Y = X @ X.T transformation."""

  def testBijectorMatrix(self):
    with self.test_session():
      bijector = bijectors.CholeskyOuterProduct(
          event_ndims=2, validate_args=True)
      self.assertEqual("cholesky_outer_product", bijector.name)
      x = [[[1., 0], [2, 1]], [[np.sqrt(2.), 0], [np.sqrt(8.), 1]]]
      y = np.matmul(x, np.transpose(x, axes=(0, 2, 1)))
      # Fairly easy to compute differentials since we have 2x2.
      dx_dy = [[[2. * 1, 0, 0],
                [2, 1, 0],
                [0, 2 * 2, 2 * 1]],
               [[2 * np.sqrt(2.), 0, 0],
                [np.sqrt(8.), np.sqrt(2.), 0],
                [0, 2 * np.sqrt(8.), 2 * 1]]]
      ildj = -np.sum(
          np.log(np.asarray(dx_dy).diagonal(
              offset=0, axis1=1, axis2=2)),
          axis=1)
      self.assertAllEqual((2, 2, 2), bijector.forward(x).get_shape())
      self.assertAllEqual((2, 2, 2), bijector.inverse(y).get_shape())
      self.assertAllClose(y, bijector.forward(x).eval())
      self.assertAllClose(x, bijector.inverse(y).eval())
      self.assertAllClose(
          ildj, bijector.inverse_log_det_jacobian(y).eval(), atol=0., rtol=1e-7)
      self.assertAllClose(
          -bijector.inverse_log_det_jacobian(y).eval(),
          bijector.forward_log_det_jacobian(x).eval(),
          atol=0.,
          rtol=1e-7)

  def testBijectorScalar(self):
    with self.test_session():
      bijector = bijectors.CholeskyOuterProduct(
          event_ndims=0, validate_args=True)
      self.assertEqual("cholesky_outer_product", bijector.name)
      x = [[[1., 5],
            [2, 1]],
           [[np.sqrt(2.), 3],
            [np.sqrt(8.), 1]]]
      y = np.square(x)
      ildj = -np.log(2.) - np.log(x)
      self.assertAllClose(y, bijector.forward(x).eval())
      self.assertAllClose(x, bijector.inverse(y).eval())
      self.assertAllClose(
          ildj, bijector.inverse_log_det_jacobian(y).eval(), atol=0., rtol=1e-7)
      self.assertAllClose(
          -bijector.inverse_log_det_jacobian(y).eval(),
          bijector.forward_log_det_jacobian(x).eval(),
          atol=0.,
          rtol=1e-7)

  def testScalarCongruency(self):
    with self.test_session():
      bijector = bijectors.CholeskyOuterProduct(
          event_ndims=0, validate_args=True)
      assert_scalar_congruency(bijector, lower_x=1e-3, upper_x=1.5, rtol=0.05)

  def testNoBatchStatic(self):
    x = np.array([[1., 0], [2, 1]])  # np.linalg.cholesky(y)
    y = np.array([[1., 2], [2, 5]])  # np.matmul(x, x.T)
    with self.test_session() as sess:
      y_actual = bijectors.CholeskyOuterProduct(event_ndims=2).forward(x=x)
      x_actual = bijectors.CholeskyOuterProduct(event_ndims=2).inverse(y=y)
    [y_actual_, x_actual_] = sess.run([y_actual, x_actual])
    self.assertAllEqual([2, 2], y_actual.get_shape())
    self.assertAllEqual([2, 2], x_actual.get_shape())
    self.assertAllClose(y, y_actual_)
    self.assertAllClose(x, x_actual_)

  def testNoBatchDeferred(self):
    x = np.array([[1., 0], [2, 1]])  # np.linalg.cholesky(y)
    y = np.array([[1., 2], [2, 5]])  # np.matmul(x, x.T)
    with self.test_session() as sess:
      x_pl = array_ops.placeholder(dtypes.float32)
      y_pl = array_ops.placeholder(dtypes.float32)
      y_actual = bijectors.CholeskyOuterProduct(event_ndims=2).forward(x=x_pl)
      x_actual = bijectors.CholeskyOuterProduct(event_ndims=2).inverse(y=y_pl)
    [y_actual_, x_actual_] = sess.run([y_actual, x_actual],
                                      feed_dict={x_pl: x, y_pl: y})
    self.assertEqual(None, y_actual.get_shape())
    self.assertEqual(None, x_actual.get_shape())
    self.assertAllClose(y, y_actual_)
    self.assertAllClose(x, x_actual_)

  def testBatchStatic(self):
    x = np.array([[[1., 0],
                   [2, 1]],
                  [[3., 0],
                   [1, 2]]])  # np.linalg.cholesky(y)
    y = np.array([[[1., 2],
                   [2, 5]],
                  [[9., 3],
                   [3, 5]]])  # np.matmul(x, x.T)
    with self.test_session() as sess:
      y_actual = bijectors.CholeskyOuterProduct(event_ndims=2).forward(x=x)
      x_actual = bijectors.CholeskyOuterProduct(event_ndims=2).inverse(y=y)
    [y_actual_, x_actual_] = sess.run([y_actual, x_actual])
    self.assertEqual([2, 2, 2], y_actual.get_shape())
    self.assertEqual([2, 2, 2], x_actual.get_shape())
    self.assertAllClose(y, y_actual_)
    self.assertAllClose(x, x_actual_)

  def testBatchDeferred(self):
    x = np.array([[[1., 0],
                   [2, 1]],
                  [[3., 0],
                   [1, 2]]])  # np.linalg.cholesky(y)
    y = np.array([[[1., 2],
                   [2, 5]],
                  [[9., 3],
                   [3, 5]]])  # np.matmul(x, x.T)
    with self.test_session() as sess:
      x_pl = array_ops.placeholder(dtypes.float32)
      y_pl = array_ops.placeholder(dtypes.float32)
      y_actual = bijectors.CholeskyOuterProduct(event_ndims=2).forward(x=x_pl)
      x_actual = bijectors.CholeskyOuterProduct(event_ndims=2).inverse(y=y_pl)
    [y_actual_, x_actual_] = sess.run([y_actual, x_actual],
                                      feed_dict={x_pl: x, y_pl: y})
    self.assertEqual(None, y_actual.get_shape())
    self.assertEqual(None, x_actual.get_shape())
    self.assertAllClose(y, y_actual_)
    self.assertAllClose(x, x_actual_)


if __name__ == "__main__":
  test.main()
