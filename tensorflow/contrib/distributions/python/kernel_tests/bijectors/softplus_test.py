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
from tensorflow.contrib.distributions.python.ops.bijectors import softplus as softplus_lib
from tensorflow.python.platform import test

rng = np.random.RandomState(42)


class SoftplusBijectorTest(test.TestCase):
  """Tests the correctness of the Y = g(X) = Log[1 + exp(X)] transformation."""

  def _softplus(self, x):
    return np.log(1 + np.exp(x))

  def _softplus_inverse(self, y):
    return np.log(np.exp(y) - 1)

  def _softplus_ildj_before_reduction(self, y):
    """Inverse log det jacobian, before being reduced."""
    return -np.log(1 - np.exp(-y))

  def testBijectorForwardInverseEventDimsZero(self):
    with self.test_session():
      bijector = softplus_lib.Softplus(event_ndims=0)
      self.assertEqual("softplus", bijector.name)
      x = 2 * rng.randn(2, 10)
      y = self._softplus(x)

      self.assertAllClose(y, bijector.forward(x).eval())
      self.assertAllClose(x, bijector.inverse(y).eval())

  def testBijectorLogDetJacobianEventDimsZero(self):
    with self.test_session():
      bijector = softplus_lib.Softplus(event_ndims=0)
      y = 2 * rng.rand(2, 10)
      # No reduction needed if event_dims = 0.
      ildj = self._softplus_ildj_before_reduction(y)

      self.assertAllClose(ildj, bijector.inverse_log_det_jacobian(y).eval())

  def testBijectorForwardInverseEventDimsOne(self):
    with self.test_session():
      bijector = softplus_lib.Softplus(event_ndims=1)
      self.assertEqual("softplus", bijector.name)
      x = 2 * rng.randn(2, 10)
      y = self._softplus(x)

      self.assertAllClose(y, bijector.forward(x).eval())
      self.assertAllClose(x, bijector.inverse(y).eval())

  def testBijectorLogDetJacobianEventDimsOne(self):
    with self.test_session():
      bijector = softplus_lib.Softplus(event_ndims=1)
      y = 2 * rng.rand(2, 10)
      ildj_before = self._softplus_ildj_before_reduction(y)
      ildj = np.sum(ildj_before, axis=1)

      self.assertAllClose(ildj, bijector.inverse_log_det_jacobian(y).eval())

  def testScalarCongruency(self):
    with self.test_session():
      bijector = softplus_lib.Softplus(event_ndims=0)
      bijector_test_util.assert_scalar_congruency(
          bijector, lower_x=-2., upper_x=2.)

  def testBijectiveAndFinite32bit(self):
    with self.test_session():
      bijector = softplus_lib.Softplus(event_ndims=0)
      x = np.linspace(-20., 20., 100).astype(np.float32)
      y = np.logspace(-10, 10, 100).astype(np.float32)
      bijector_test_util.assert_bijective_and_finite(
          bijector, x, y, rtol=1e-2, atol=1e-2)

  def testBijectiveAndFinite16bit(self):
    with self.test_session():
      bijector = softplus_lib.Softplus(event_ndims=0)
      # softplus(-20) is zero, so we can't use such a large range as in 32bit.
      x = np.linspace(-10., 20., 100).astype(np.float16)
      # Note that float16 is only in the open set (0, inf) for a smaller
      # logspace range.  The actual range was (-7, 4), so use something smaller
      # for the test.
      y = np.logspace(-6, 3, 100).astype(np.float16)
      bijector_test_util.assert_bijective_and_finite(
          bijector, x, y, rtol=1e-1, atol=1e-3)


if __name__ == "__main__":
  test.main()
