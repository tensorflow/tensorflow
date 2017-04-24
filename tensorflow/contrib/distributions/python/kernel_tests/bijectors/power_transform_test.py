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

from tensorflow.contrib.distributions.python.ops.bijectors.bijector_test_util import assert_bijective_and_finite
from tensorflow.contrib.distributions.python.ops.bijectors.bijector_test_util import assert_scalar_congruency
from tensorflow.contrib.distributions.python.ops.bijectors.power_transform import PowerTransform
from tensorflow.python.platform import test


class PowerTransformBijectorTest(test.TestCase):
  """Tests correctness of the power transformation."""

  def testBijector(self):
    with self.test_session():
      c = 0.2
      bijector = PowerTransform(
          power=c, event_ndims=1, validate_args=True)
      self.assertEqual("power_transform", bijector.name)
      x = np.array([[[-1.], [2.], [-5. + 1e-4]]])
      y = (1. + x * c)**(1. / c)
      self.assertAllClose(y, bijector.forward(x).eval())
      self.assertAllClose(x, bijector.inverse(y).eval())
      self.assertAllClose(
          (c - 1.) * np.sum(np.log(y), axis=-1),
          bijector.inverse_log_det_jacobian(y).eval())
      self.assertAllClose(
          -bijector.inverse_log_det_jacobian(y).eval(),
          bijector.forward_log_det_jacobian(x).eval(),
          rtol=1e-4,
          atol=0.)

  def testScalarCongruency(self):
    with self.test_session():
      bijector = PowerTransform(
          power=0.2, validate_args=True)
      assert_scalar_congruency(
          bijector, lower_x=-2., upper_x=1.5, rtol=0.05)

  def testBijectiveAndFinite(self):
    with self.test_session():
      bijector = PowerTransform(
          power=0.2, event_ndims=0, validate_args=True)
      x = np.linspace(-4.999, 10, num=10).astype(np.float32)
      y = np.logspace(0.001, 10, num=10).astype(np.float32)
      assert_bijective_and_finite(bijector, x, y, rtol=1e-3)


if __name__ == "__main__":
  test.main()
