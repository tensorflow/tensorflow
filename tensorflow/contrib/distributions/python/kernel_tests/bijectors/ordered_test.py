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

from tensorflow.contrib.distributions.python.ops.bijectors.ordered import Ordered
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.distributions.bijector_test_util import assert_bijective_and_finite
from tensorflow.python.platform import test


rng = np.random.RandomState(42)


class OrderedBijectorTest(test.TestCase):
  """Tests correctness of the ordered transformation."""

  def testBijectorVector(self):
    with self.test_session():
      ordered = Ordered()
      self.assertEqual("ordered", ordered.name)
      x = np.asarray([[2., 3, 4], [4., 8, 13]])
      y = [[2., 0, 0], [4., np.log(4.), np.log(5.)]]
      self.assertAllClose(y, ordered.forward(x).eval())
      self.assertAllClose(x, ordered.inverse(y).eval())
      self.assertAllClose(
          np.sum(np.asarray(y)[..., 1:], axis=-1),
          ordered.inverse_log_det_jacobian(y, event_ndims=1).eval(),
          atol=0.,
          rtol=1e-7)
      self.assertAllClose(
          -ordered.inverse_log_det_jacobian(y, event_ndims=1).eval(),
          ordered.forward_log_det_jacobian(x, event_ndims=1).eval(),
          atol=0.,
          rtol=1e-7)

  def testBijectorUnknownShape(self):
    with self.test_session():
      ordered = Ordered()
      self.assertEqual("ordered", ordered.name)
      x = array_ops.placeholder(shape=[2, None], dtype=dtypes.float32)
      real_x = np.asarray([[2., 3, 4], [4., 8, 13]])
      y = array_ops.placeholder(shape=[2, None], dtype=dtypes.float32)
      real_y = [[2., 0, 0], [4., np.log(4.), np.log(5.)]]
      self.assertAllClose(real_y, ordered.forward(x).eval(
          feed_dict={x: real_x}))
      self.assertAllClose(real_x, ordered.inverse(y).eval(
          feed_dict={y: real_y}))
      self.assertAllClose(
          np.sum(np.asarray(real_y)[..., 1:], axis=-1),
          ordered.inverse_log_det_jacobian(y, event_ndims=1).eval(
              feed_dict={y: real_y}),
          atol=0.,
          rtol=1e-7)
      self.assertAllClose(
          -ordered.inverse_log_det_jacobian(y, event_ndims=1).eval(
              feed_dict={y: real_y}),
          ordered.forward_log_det_jacobian(x, event_ndims=1).eval(
              feed_dict={x: real_x}),
          atol=0.,
          rtol=1e-7)

  def testShapeGetters(self):
    with self.test_session():
      x = tensor_shape.TensorShape([4])
      y = tensor_shape.TensorShape([4])
      bijector = Ordered(validate_args=True)
      self.assertAllEqual(y, bijector.forward_event_shape(x))
      self.assertAllEqual(y.as_list(),
                          bijector.forward_event_shape_tensor(
                              x.as_list()).eval())
      self.assertAllEqual(x, bijector.inverse_event_shape(y))
      self.assertAllEqual(x.as_list(),
                          bijector.inverse_event_shape_tensor(
                              y.as_list()).eval())

  def testBijectiveAndFinite(self):
    with self.test_session():
      ordered = Ordered()
      x = np.sort(rng.randn(3, 10), axis=-1).astype(np.float32)
      y = (rng.randn(3, 10)).astype(np.float32)
      assert_bijective_and_finite(ordered, x, y, event_ndims=1)


if __name__ == "__main__":
  test.main()
