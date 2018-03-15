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
"""Chain Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops.bijectors.chain import Chain
from tensorflow.contrib.distributions.python.ops.bijectors.exp import Exp
from tensorflow.contrib.distributions.python.ops.bijectors.softmax_centered import SoftmaxCentered
from tensorflow.contrib.distributions.python.ops.bijectors.softplus import Softplus
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.distributions.bijector_test_util import assert_scalar_congruency
from tensorflow.python.platform import test


class ChainBijectorTest(test.TestCase):
  """Tests the correctness of the Y = Chain(bij1, bij2, bij3) transformation."""

  def testBijector(self):
    with self.test_session():
      chain = Chain((Exp(event_ndims=1), Softplus(event_ndims=1)))
      self.assertEqual("chain_of_exp_of_softplus", chain.name)
      x = np.asarray([[[1., 2.],
                       [2., 3.]]])
      self.assertAllClose(1. + np.exp(x), chain.forward(x).eval())
      self.assertAllClose(np.log(x - 1.), chain.inverse(x).eval())
      self.assertAllClose(
          -np.sum(np.log(x - 1.), axis=2),
          chain.inverse_log_det_jacobian(x).eval())
      self.assertAllClose(
          np.sum(x, axis=2), chain.forward_log_det_jacobian(x).eval())

  def testBijectorIdentity(self):
    with self.test_session():
      chain = Chain()
      self.assertEqual("identity", chain.name)
      x = np.asarray([[[1., 2.],
                       [2., 3.]]])
      self.assertAllClose(x, chain.forward(x).eval())
      self.assertAllClose(x, chain.inverse(x).eval())
      self.assertAllClose(0., chain.inverse_log_det_jacobian(x).eval())
      self.assertAllClose(0., chain.forward_log_det_jacobian(x).eval())

  def testScalarCongruency(self):
    with self.test_session():
      bijector = Chain((Exp(), Softplus()))
      assert_scalar_congruency(
          bijector, lower_x=1e-3, upper_x=1.5, rtol=0.05)

  def testShapeGetters(self):
    with self.test_session():
      bijector = Chain([
          SoftmaxCentered(
              event_ndims=1, validate_args=True),
          SoftmaxCentered(
              event_ndims=0, validate_args=True)
      ])
      x = tensor_shape.TensorShape([])
      y = tensor_shape.TensorShape([2 + 1])
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
