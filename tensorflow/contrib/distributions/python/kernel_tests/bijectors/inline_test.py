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

from tensorflow.contrib.distributions.python.ops.bijectors.exp import Exp
from tensorflow.contrib.distributions.python.ops.bijectors.inline import Inline
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class InlineBijectorTest(test.TestCase):
  """Tests correctness of the inline constructed bijector."""

  def testBijector(self):
    with self.test_session():
      exp = Exp(event_ndims=1)
      inline = Inline(
          forward_fn=math_ops.exp,
          inverse_fn=math_ops.log,
          inverse_log_det_jacobian_fn=(
              lambda y: -math_ops.reduce_sum(  # pylint: disable=g-long-lambda
                  math_ops.log(y), reduction_indices=-1)),
          forward_log_det_jacobian_fn=(
              lambda x: math_ops.reduce_sum(x, reduction_indices=-1)),
          name="exp")

      self.assertEqual(exp.name, inline.name)
      x = [[[1., 2.], [3., 4.], [5., 6.]]]
      y = np.exp(x)
      self.assertAllClose(y, inline.forward(x).eval())
      self.assertAllClose(x, inline.inverse(y).eval())
      self.assertAllClose(
          -np.sum(np.log(y), axis=-1),
          inline.inverse_log_det_jacobian(y).eval())
      self.assertAllClose(-inline.inverse_log_det_jacobian(y).eval(),
                          inline.forward_log_det_jacobian(x).eval())

  def testShapeGetters(self):
    with self.test_session():
      bijector = Inline(
          forward_event_shape_tensor_fn=lambda x: array_ops.concat((x, [1]), 0),
          forward_event_shape_fn=lambda x: x.as_list() + [1],
          inverse_event_shape_tensor_fn=lambda x: x[:-1],
          inverse_event_shape_fn=lambda x: x[:-1],
          name="shape_only")
      x = tensor_shape.TensorShape([1, 2, 3])
      y = tensor_shape.TensorShape([1, 2, 3, 1])
      self.assertAllEqual(y, bijector.forward_event_shape(x))
      self.assertAllEqual(
          y.as_list(),
          bijector.forward_event_shape_tensor(x.as_list()).eval())
      self.assertAllEqual(x, bijector.inverse_event_shape(y))
      self.assertAllEqual(
          x.as_list(),
          bijector.inverse_event_shape_tensor(y.as_list()).eval())


if __name__ == "__main__":
  test.main()
