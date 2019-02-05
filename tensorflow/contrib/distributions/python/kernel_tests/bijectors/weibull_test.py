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

from tensorflow.contrib.distributions.python.ops.bijectors.weibull import Weibull
from tensorflow.python.ops.distributions.bijector_test_util import assert_bijective_and_finite
from tensorflow.python.ops.distributions.bijector_test_util import assert_scalar_congruency
from tensorflow.python.platform import test


class WeibullBijectorTest(test.TestCase):
  """Tests correctness of the weibull bijector."""

  def testBijector(self):
    with self.cached_session():
      scale = 5.
      concentration = 0.3
      bijector = Weibull(
          scale=scale, concentration=concentration,
          validate_args=True)
      self.assertEqual("weibull", bijector.name)
      x = np.array([[[0.], [1.], [14.], [20.], [100.]]], dtype=np.float32)
      # Weibull distribution
      weibull_dist = stats.frechet_r(c=concentration, scale=scale)
      y = weibull_dist.cdf(x).astype(np.float32)
      self.assertAllClose(y, bijector.forward(x).eval())
      self.assertAllClose(x, bijector.inverse(y).eval())
      self.assertAllClose(
          weibull_dist.logpdf(x),
          bijector.forward_log_det_jacobian(x, event_ndims=0).eval())
      self.assertAllClose(
          -bijector.inverse_log_det_jacobian(y, event_ndims=0).eval(),
          bijector.forward_log_det_jacobian(x, event_ndims=0).eval(),
          rtol=1e-4,
          atol=0.)

  def testScalarCongruency(self):
    with self.cached_session():
      assert_scalar_congruency(
          Weibull(scale=20., concentration=0.3),
          lower_x=1., upper_x=100., rtol=0.02)

  def testBijectiveAndFinite(self):
    with self.cached_session():
      bijector = Weibull(
          scale=20., concentration=2., validate_args=True)
      x = np.linspace(1., 8., num=10).astype(np.float32)
      y = np.linspace(
          -np.expm1(-1 / 400.),
          -np.expm1(-16), num=10).astype(np.float32)
      assert_bijective_and_finite(bijector, x, y, event_ndims=0, rtol=1e-3)


if __name__ == "__main__":
  test.main()
