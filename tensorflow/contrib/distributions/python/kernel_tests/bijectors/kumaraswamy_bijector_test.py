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
"""Tests for Kumaraswamy Bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops.bijectors.kumaraswamy import Kumaraswamy
from tensorflow.python.ops.distributions.bijector_test_util import assert_bijective_and_finite
from tensorflow.python.ops.distributions.bijector_test_util import assert_scalar_congruency
from tensorflow.python.platform import test


class KumaraswamyBijectorTest(test.TestCase):
  """Tests correctness of the Kumaraswamy bijector."""

  def testBijector(self):
    with self.test_session():
      a = 2.
      b = 0.3
      bijector = Kumaraswamy(
          concentration1=a, concentration0=b,
          event_ndims=0, validate_args=True)
      self.assertEqual("kumaraswamy", bijector.name)
      x = np.array([[[0.1], [0.2], [0.3], [0.4], [0.5]]], dtype=np.float32)
      # Kumaraswamy cdf. This is the same as inverse(x).
      y = 1. - (1. - x ** a) ** b
      self.assertAllClose(y, bijector.inverse(x).eval())
      self.assertAllClose(x, bijector.forward(y).eval())
      kumaraswamy_log_pdf = (np.log(a) + np.log(b) + (a - 1) * np.log(x) +
                             (b - 1) * np.log1p(-x ** a))

      self.assertAllClose(
          # We should lose a dimension from calculating the determinant of the
          # jacobian.
          kumaraswamy_log_pdf,
          bijector.inverse_log_det_jacobian(x).eval())
      self.assertAllClose(
          -bijector.inverse_log_det_jacobian(x).eval(),
          bijector.forward_log_det_jacobian(y).eval(),
          rtol=1e-4,
          atol=0.)

  def testScalarCongruency(self):
    with self.test_session():
      assert_scalar_congruency(
          Kumaraswamy(concentration1=0.5, concentration0=1.1),
          lower_x=0., upper_x=1., n=int(10e3), rtol=0.02)

  def testBijectiveAndFinite(self):
    with self.test_session():
      concentration1 = 1.2
      concentration0 = 2.
      bijector = Kumaraswamy(
          concentration1=concentration1,
          concentration0=concentration0, validate_args=True)
      # Omitting the endpoints 0 and 1, since idlj will be infinity at these
      # endpoints.
      y = np.linspace(.01, 0.99, num=10).astype(np.float32)
      x = 1 - (1 - y ** concentration1) ** concentration0
      assert_bijective_and_finite(bijector, x, y, rtol=1e-3)


if __name__ == "__main__":
  test.main()
