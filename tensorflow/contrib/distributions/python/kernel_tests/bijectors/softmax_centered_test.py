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

from tensorflow.contrib.distributions.python.ops.bijectors import bijector_test_util
from tensorflow.contrib.distributions.python.ops.bijectors import softmax_centered as softmax_centered_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import test


rng = np.random.RandomState(42)


class SoftmaxCenteredBijectorTest(test.TestCase):
  """Tests correctness of the Y = g(X) = exp(X) / sum(exp(X)) transformation."""

  def testBijectorScalar(self):
    with self.test_session():
      softmax = softmax_centered_lib.SoftmaxCentered()  # scalar by default
      self.assertEqual("softmax_centered", softmax.name)
      x = np.log([[2., 3, 4],
                  [4., 8, 12]])
      y = [[[2. / 3, 1. / 3],
            [3. / 4, 1. / 4],
            [4. / 5, 1. / 5]],
           [[4. / 5, 1. / 5],
            [8. / 9, 1. / 9],
            [12. / 13, 1. / 13]]]
      self.assertAllClose(y, softmax.forward(x).eval())
      self.assertAllClose(x, softmax.inverse(y).eval())
      self.assertAllClose(
          -np.sum(np.log(y), axis=2),
          softmax.inverse_log_det_jacobian(y).eval(),
          atol=0.,
          rtol=1e-7)
      self.assertAllClose(
          -softmax.inverse_log_det_jacobian(y).eval(),
          softmax.forward_log_det_jacobian(x).eval(),
          atol=0.,
          rtol=1e-7)

  def testBijectorVector(self):
    with self.test_session():
      softmax = softmax_centered_lib.SoftmaxCentered(event_ndims=1)
      self.assertEqual("softmax_centered", softmax.name)
      x = np.log([[2., 3, 4], [4., 8, 12]])
      y = [[0.2, 0.3, 0.4, 0.1], [0.16, 0.32, 0.48, 0.04]]
      self.assertAllClose(y, softmax.forward(x).eval())
      self.assertAllClose(x, softmax.inverse(y).eval())
      self.assertAllClose(
          -np.sum(np.log(y), axis=1),
          softmax.inverse_log_det_jacobian(y).eval(),
          atol=0.,
          rtol=1e-7)
      self.assertAllClose(
          -softmax.inverse_log_det_jacobian(y).eval(),
          softmax.forward_log_det_jacobian(x).eval(),
          atol=0.,
          rtol=1e-7)

  def testShapeGetters(self):
    with self.test_session():
      for x, y, b in ((tensor_shape.TensorShape([]),
                       tensor_shape.TensorShape([2]),
                       softmax_centered_lib.SoftmaxCentered(
                           event_ndims=0, validate_args=True)),
                      (tensor_shape.TensorShape([4]),
                       tensor_shape.TensorShape([5]),
                       softmax_centered_lib.SoftmaxCentered(
                           event_ndims=1, validate_args=True))):
        self.assertAllEqual(y, b.forward_event_shape(x))
        self.assertAllEqual(y.as_list(),
                            b.forward_event_shape_tensor(x.as_list()).eval())
        self.assertAllEqual(x, b.inverse_event_shape(y))
        self.assertAllEqual(x.as_list(),
                            b.inverse_event_shape_tensor(y.as_list()).eval())

  def testBijectiveAndFinite(self):
    with self.test_session():
      softmax = softmax_centered_lib.SoftmaxCentered(event_ndims=1)
      x = np.linspace(-50, 50, num=10).reshape(5, 2).astype(np.float32)
      # Make y values on the simplex with a wide range.
      y_0 = np.ones(5).astype(np.float32)
      y_1 = (1e-5 * rng.rand(5)).astype(np.float32)
      y_2 = (1e1 * rng.rand(5)).astype(np.float32)
      y = np.array([y_0, y_1, y_2])
      y /= y.sum(axis=0)
      y = y.T  # y.shape = [5, 3]
      bijector_test_util.assert_bijective_and_finite(softmax, x, y)


if __name__ == "__main__":
  test.main()
