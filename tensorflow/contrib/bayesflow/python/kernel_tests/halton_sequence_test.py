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
"""Tests for halton_sequence.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.bayesflow.python.ops import halton_sequence as halton
from tensorflow.contrib.bayesflow.python.ops import monte_carlo_impl as monte_carlo_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.platform import test


mc = monte_carlo_lib


class HaltonSequenceTest(test.TestCase):

  def test_known_values_small_bases(self):
    with self.test_session():
      # The first five elements of the Halton sequence with base 2 and 3
      expected = np.array(((1. / 2, 1. / 3),
                           (1. / 4, 2. / 3),
                           (3. / 4, 1. / 9),
                           (1. / 8, 4. / 9),
                           (5. / 8, 7. / 9)), dtype=np.float32)
      sample = halton.sample(2, num_samples=5)
      self.assertAllClose(expected, sample.eval(), rtol=1e-6)

  def test_sample_indices(self):
    with self.test_session():
      dim = 5
      indices = math_ops.range(10, dtype=dtypes.int32)
      sample_direct = halton.sample(dim, num_samples=10)
      sample_from_indices = halton.sample(dim, sample_indices=indices)
      self.assertAllClose(sample_direct.eval(), sample_from_indices.eval(),
                          rtol=1e-6)

  def test_dtypes_works_correctly(self):
    with self.test_session():
      dim = 3
      sample_float32 = halton.sample(dim, num_samples=10, dtype=dtypes.float32)
      sample_float64 = halton.sample(dim, num_samples=10, dtype=dtypes.float64)
      self.assertEqual(sample_float32.eval().dtype, np.float32)
      self.assertEqual(sample_float64.eval().dtype, np.float64)

  def test_normal_integral_mean_and_var_correctly_estimated(self):
    n = int(1000)
    # This test is almost identical to the similarly named test in
    # monte_carlo_test.py. The only difference is that we use the Halton
    # samples instead of the random samples to evaluate the expectations.
    # MC with pseudo random numbers converges at the rate of 1/ Sqrt(N)
    # (N=number of samples). For QMC in low dimensions, the expected convergence
    # rate is ~ 1/N. Hence we should only need 1e3 samples as compared to the
    # 1e6 samples used in the pseudo-random monte carlo.
    with self.test_session():
      mu_p = array_ops.constant([-1.0, 1.0], dtype=dtypes.float64)
      mu_q = array_ops.constant([0.0, 0.0], dtype=dtypes.float64)
      sigma_p = array_ops.constant([0.5, 0.5], dtype=dtypes.float64)
      sigma_q = array_ops.constant([1.0, 1.0], dtype=dtypes.float64)
      p = normal_lib.Normal(loc=mu_p, scale=sigma_p)
      q = normal_lib.Normal(loc=mu_q, scale=sigma_q)

      cdf_sample = halton.sample(2, num_samples=n, dtype=dtypes.float64)
      q_sample = q.quantile(cdf_sample)

      # Compute E_p[X].
      e_x = mc.expectation_importance_sampler(
          f=lambda x: x, log_p=p.log_prob, sampling_dist_q=q, z=q_sample,
          seed=42)

      # Compute E_p[X^2].
      e_x2 = mc.expectation_importance_sampler(
          f=math_ops.square, log_p=p.log_prob, sampling_dist_q=q, z=q_sample,
          seed=42)

      stddev = math_ops.sqrt(e_x2 - math_ops.square(e_x))
      # Keep the tolerance levels the same as in monte_carlo_test.py.
      self.assertEqual(p.batch_shape, e_x.get_shape())
      self.assertAllClose(p.mean().eval(), e_x.eval(), rtol=0.01)
      self.assertAllClose(p.stddev().eval(), stddev.eval(), rtol=0.02)

  def test_docstring_example(self):
    # Produce the first 1000 members of the Halton sequence in 3 dimensions.
    num_samples = 1000
    dim = 3
    with self.test_session():
      sample = halton.sample(dim, num_samples=num_samples)

      # Evaluate the integral of x_1 * x_2^2 * x_3^3  over the three dimensional
      # hypercube.
      powers = math_ops.range(1.0, limit=dim + 1)
      integral = math_ops.reduce_mean(
          math_ops.reduce_prod(sample ** powers, axis=-1))
      true_value = 1.0 / math_ops.reduce_prod(powers + 1.0)

      # Produces a relative absolute error of 1.7%.
      self.assertAllClose(integral.eval(), true_value.eval(), rtol=0.02)

    # Now skip the first 1000 samples and recompute the integral with the next
    # thousand samples. The sample_indices argument can be used to do this.

      sample_indices = math_ops.range(start=1000, limit=1000 + num_samples,
                                      dtype=dtypes.int32)
      sample_leaped = halton.sample(dim, sample_indices=sample_indices)

      integral_leaped = math_ops.reduce_mean(
          math_ops.reduce_prod(sample_leaped ** powers, axis=-1))
      self.assertAllClose(integral_leaped.eval(), true_value.eval(), rtol=0.001)


if __name__ == '__main__':
  test.main()
