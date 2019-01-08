# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.contrib.distributions.python.ops.bijectors.softsign import Softsign
from tensorflow.python.framework import test_util
from tensorflow.python.ops.distributions.bijector_test_util import assert_bijective_and_finite
from tensorflow.python.ops.distributions.bijector_test_util import assert_scalar_congruency
from tensorflow.python.platform import test


class SoftsignBijectorTest(test.TestCase):
  """Tests the correctness of the Y = g(X) = X / (1 + |X|) transformation."""

  def _softsign(self, x):
    return x / (1. + np.abs(x))

  def _softsign_ildj_before_reduction(self, y):
    """Inverse log det jacobian, before being reduced."""
    return -2. * np.log1p(-np.abs(y))

  def setUp(self):
    self._rng = np.random.RandomState(42)

  @test_util.run_in_graph_and_eager_modes
  def testBijectorBounds(self):
    bijector = Softsign(validate_args=True)
    with self.assertRaisesOpError("greater than -1"):
      self.evaluate(bijector.inverse(-3.))
    with self.assertRaisesOpError("greater than -1"):
      self.evaluate(bijector.inverse_log_det_jacobian(-3., event_ndims=0))

    with self.assertRaisesOpError("less than 1"):
      self.evaluate(bijector.inverse(3.))
    with self.assertRaisesOpError("less than 1"):
      self.evaluate(bijector.inverse_log_det_jacobian(3., event_ndims=0))

  @test_util.run_in_graph_and_eager_modes
  def testBijectorForwardInverse(self):
    bijector = Softsign(validate_args=True)
    self.assertEqual("softsign", bijector.name)
    x = 2. * self._rng.randn(2, 10)
    y = self._softsign(x)

    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))

  @test_util.run_in_graph_and_eager_modes
  def testBijectorLogDetJacobianEventDimsZero(self):
    bijector = Softsign(validate_args=True)
    y = self._rng.rand(2, 10)
    # No reduction needed if event_dims = 0.
    ildj = self._softsign_ildj_before_reduction(y)

    self.assertAllClose(ildj, self.evaluate(
        bijector.inverse_log_det_jacobian(y, event_ndims=0)))

  @test_util.run_in_graph_and_eager_modes
  def testBijectorForwardInverseEventDimsOne(self):
    bijector = Softsign(validate_args=True)
    self.assertEqual("softsign", bijector.name)
    x = 2. * self._rng.randn(2, 10)
    y = self._softsign(x)
    self.assertAllClose(y, self.evaluate(bijector.forward(x)))
    self.assertAllClose(x, self.evaluate(bijector.inverse(y)))

  @test_util.run_in_graph_and_eager_modes
  def testBijectorLogDetJacobianEventDimsOne(self):
    bijector = Softsign(validate_args=True)
    y = self._rng.rand(2, 10)
    ildj_before = self._softsign_ildj_before_reduction(y)
    ildj = np.sum(ildj_before, axis=1)
    self.assertAllClose(
        ildj, self.evaluate(
            bijector.inverse_log_det_jacobian(y, event_ndims=1)))

  def testScalarCongruency(self):
    with self.cached_session():
      bijector = Softsign(validate_args=True)
      assert_scalar_congruency(bijector, lower_x=-20., upper_x=20.)

  def testBijectiveAndFinite(self):
    with self.cached_session():
      bijector = Softsign(validate_args=True)
      x = np.linspace(-20., 20., 100).astype(np.float32)
      y = np.linspace(-0.99, 0.99, 100).astype(np.float32)
      assert_bijective_and_finite(
          bijector, x, y, event_ndims=0, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
  test.main()
