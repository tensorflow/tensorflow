# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
from scipy import stats

from tensorflow.contrib.distributions.python.ops.bijectors.gumbel import Gumbel
from tensorflow.python.ops.distributions.bijector_test_util import assert_bijective_and_finite
from tensorflow.python.ops.distributions.bijector_test_util import assert_scalar_congruency
from tensorflow.python.platform import test


class GumbelBijectorTest(test.TestCase):
  """Tests correctness of the Gumbel bijector."""

  def testBijector(self):
    with self.test_session():
      loc = 0.3
      scale = 5.
      bijector = Gumbel(loc=loc, scale=scale, event_ndims=1, validate_args=True)
      self.assertEqual("gumbel", bijector.name)
      x = np.array([[[-3.], [0.], [0.5], [4.2], [12.]]], dtype=np.float32)
      # Gumbel distribution
      gumbel_dist = stats.gumbel_r(loc=loc, scale=scale)
      y = gumbel_dist.cdf(x).astype(np.float32)
      self.assertAllClose(y, bijector.forward(x).eval())
      self.assertAllClose(x, bijector.inverse(y).eval())
      self.assertAllClose(
          # We should lose a dimension from calculating the determinant of the
          # jacobian.
          np.squeeze(gumbel_dist.logpdf(x), axis=2),
          bijector.forward_log_det_jacobian(x).eval())
      self.assertAllClose(
          -bijector.inverse_log_det_jacobian(y).eval(),
          bijector.forward_log_det_jacobian(x).eval(),
          rtol=1e-4,
          atol=0.)

  def testScalarCongruency(self):
    with self.test_session():
      assert_scalar_congruency(
          Gumbel(loc=0.3, scale=20.), lower_x=1., upper_x=100., rtol=0.02)

  def testBijectiveAndFinite(self):
    with self.test_session():
      bijector = Gumbel(loc=0., scale=3.0, event_ndims=0, validate_args=True)
      x = np.linspace(-10., 10., num=10).astype(np.float32)
      y = np.linspace(0.01, 0.99, num=10).astype(np.float32)
      assert_bijective_and_finite(bijector, x, y, rtol=1e-3)


if __name__ == "__main__":
  test.main()
