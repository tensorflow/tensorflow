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

from tensorflow.contrib.distributions.python.ops.bijectors.affine import Affine
from tensorflow.contrib.distributions.python.ops.bijectors.chain import Chain
from tensorflow.contrib.distributions.python.ops.bijectors.exp import Exp
from tensorflow.contrib.distributions.python.ops.bijectors.softmax_centered import SoftmaxCentered
from tensorflow.contrib.distributions.python.ops.bijectors.softplus import Softplus
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.distributions import bijector
from tensorflow.python.ops.distributions.bijector_test_util import assert_scalar_congruency
from tensorflow.python.platform import test


class ShapeChanging(bijector.Bijector):
  """Only used for op_ndims manipulation."""

  def __init__(self, forward_min_event_ndims=0, inverse_min_event_ndims=3):
    super(ShapeChanging, self).__init__(
        forward_min_event_ndims=forward_min_event_ndims,
        inverse_min_event_ndims=inverse_min_event_ndims,
        validate_args=False, name="shape_changer")


class ChainBijectorTest(test.TestCase):
  """Tests the correctness of the Y = Chain(bij1, bij2, bij3) transformation."""

  def testBijector(self):
    with self.test_session():
      chain = Chain((Exp(), Softplus()))
      self.assertEqual("chain_of_exp_of_softplus", chain.name)
      x = np.asarray([[[1., 2.],
                       [2., 3.]]])
      self.assertAllClose(1. + np.exp(x), chain.forward(x).eval())
      self.assertAllClose(np.log(x - 1.), chain.inverse(x).eval())
      self.assertAllClose(
          -np.sum(np.log(x - 1.), axis=2),
          chain.inverse_log_det_jacobian(x, event_ndims=1).eval())
      self.assertAllClose(
          np.sum(x, axis=2),
          chain.forward_log_det_jacobian(x, event_ndims=1).eval())

  def testBijectorIdentity(self):
    with self.test_session():
      chain = Chain()
      self.assertEqual("identity", chain.name)
      x = np.asarray([[[1., 2.],
                       [2., 3.]]])
      self.assertAllClose(x, chain.forward(x).eval())
      self.assertAllClose(x, chain.inverse(x).eval())
      self.assertAllClose(
          0., chain.inverse_log_det_jacobian(x, event_ndims=1).eval())
      self.assertAllClose(
          0., chain.forward_log_det_jacobian(x, event_ndims=1).eval())

  def testScalarCongruency(self):
    with self.test_session():
      chain = Chain((Exp(), Softplus()))
      assert_scalar_congruency(
          chain, lower_x=1e-3, upper_x=1.5, rtol=0.05)

  def testShapeGetters(self):
    with self.test_session():
      chain = Chain([
          SoftmaxCentered(validate_args=True),
          SoftmaxCentered(validate_args=True),
      ])
      x = tensor_shape.TensorShape([1])
      y = tensor_shape.TensorShape([2 + 1])
      self.assertAllEqual(y, chain.forward_event_shape(x))
      self.assertAllEqual(
          y.as_list(),
          chain.forward_event_shape_tensor(x.as_list()).eval())
      self.assertAllEqual(x, chain.inverse_event_shape(y))
      self.assertAllEqual(
          x.as_list(),
          chain.inverse_event_shape_tensor(y.as_list()).eval())

  def testMinEventNdimsChain(self):
    chain = Chain([Exp(), Exp(), Exp()])
    self.assertEqual(0, chain.forward_min_event_ndims)
    self.assertEqual(0, chain.inverse_min_event_ndims)

    chain = Chain([Affine(), Affine(), Affine()])
    self.assertEqual(1, chain.forward_min_event_ndims)
    self.assertEqual(1, chain.inverse_min_event_ndims)

    chain = Chain([Exp(), Affine()])
    self.assertEqual(1, chain.forward_min_event_ndims)
    self.assertEqual(1, chain.inverse_min_event_ndims)

    chain = Chain([Affine(), Exp()])
    self.assertEqual(1, chain.forward_min_event_ndims)
    self.assertEqual(1, chain.inverse_min_event_ndims)

    chain = Chain([Affine(), Exp(), Softplus(), Affine()])
    self.assertEqual(1, chain.forward_min_event_ndims)
    self.assertEqual(1, chain.inverse_min_event_ndims)

  def testMinEventNdimsShapeChangingAddDims(self):
    chain = Chain([ShapeChanging()])
    self.assertEqual(0, chain.forward_min_event_ndims)
    self.assertEqual(3, chain.inverse_min_event_ndims)

    chain = Chain([ShapeChanging(), Affine()])
    self.assertEqual(1, chain.forward_min_event_ndims)
    self.assertEqual(4, chain.inverse_min_event_ndims)

    chain = Chain([Affine(), ShapeChanging()])
    self.assertEqual(0, chain.forward_min_event_ndims)
    self.assertEqual(3, chain.inverse_min_event_ndims)

    chain = Chain([ShapeChanging(), ShapeChanging()])
    self.assertEqual(0, chain.forward_min_event_ndims)
    self.assertEqual(6, chain.inverse_min_event_ndims)

  def testMinEventNdimsShapeChangingRemoveDims(self):
    chain = Chain([ShapeChanging(3, 0)])
    self.assertEqual(3, chain.forward_min_event_ndims)
    self.assertEqual(0, chain.inverse_min_event_ndims)

    chain = Chain([ShapeChanging(3, 0), Affine()])
    self.assertEqual(3, chain.forward_min_event_ndims)
    self.assertEqual(0, chain.inverse_min_event_ndims)

    chain = Chain([Affine(), ShapeChanging(3, 0)])
    self.assertEqual(4, chain.forward_min_event_ndims)
    self.assertEqual(1, chain.inverse_min_event_ndims)

    chain = Chain([ShapeChanging(3, 0), ShapeChanging(3, 0)])
    self.assertEqual(6, chain.forward_min_event_ndims)
    self.assertEqual(0, chain.inverse_min_event_ndims)

  def testMinEventNdimsShapeChangingAddRemoveDims(self):
    chain = Chain([
        ShapeChanging(2, 1),
        ShapeChanging(3, 0),
        ShapeChanging(1, 2)])
    self.assertEqual(4, chain.forward_min_event_ndims)
    self.assertEqual(1, chain.inverse_min_event_ndims)

  def testChainExpAffine(self):
    scale_diag = np.array([1., 2., 3.], dtype=np.float32)
    chain = Chain([Exp(), Affine(scale_diag=scale_diag)])
    x = [0., np.log(2., dtype=np.float32), np.log(3., dtype=np.float32)]
    y = [1., 4., 27.]
    self.assertAllClose(y, self.evaluate(chain.forward(x)))
    self.assertAllClose(x, self.evaluate(chain.inverse(y)))
    self.assertAllClose(
        np.log(6, dtype=np.float32) + np.sum(scale_diag * x),
        self.evaluate(chain.forward_log_det_jacobian(x, event_ndims=1)))

    self.assertAllClose(
        -np.log(6, dtype=np.float32) - np.sum(scale_diag * x),
        self.evaluate(chain.inverse_log_det_jacobian(y, event_ndims=1)))

  def testChainAffineExp(self):
    scale_diag = np.array([1., 2., 3.], dtype=np.float32)
    chain = Chain([Affine(scale_diag=scale_diag), Exp()])
    x = [0., np.log(2., dtype=np.float32), np.log(3., dtype=np.float32)]
    y = [1., 4., 9.]
    self.assertAllClose(y, self.evaluate(chain.forward(x)))
    self.assertAllClose(x, self.evaluate(chain.inverse(y)))
    self.assertAllClose(
        np.log(6, dtype=np.float32) + np.sum(x),
        self.evaluate(chain.forward_log_det_jacobian(x, event_ndims=1)))

    self.assertAllClose(
        -np.log(6, dtype=np.float32) - np.sum(x),
        self.evaluate(chain.inverse_log_det_jacobian(y, event_ndims=1)))

  def testChainIldjWithPlaceholder(self):
    chain = Chain((Exp(), Exp()))
    samples = array_ops.placeholder(
        dtype=np.float32, shape=[None, 10], name="samples")
    ildj = chain.inverse_log_det_jacobian(samples, event_ndims=0)
    self.assertTrue(ildj is not None)
    with self.test_session():
      ildj.eval({samples: np.zeros([2, 10], np.float32)})


if __name__ == "__main__":
  test.main()
