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
"""Tests for random-number generation ops in the XLA JIT compiler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import special_math
from tensorflow.python.platform import googletest


class RandomOpsTest(xla_test.XLATestCase):
  """Test cases for random-number generating operators."""

  def _random_types(self):
    return set(self.numeric_types) - set(self.complex_types)

  def _testRngIsNotConstant(self, rng, dtype):
    # Tests that 'rng' does not always return the same value.
    with self.test_session() as sess:
      with self.test_scope():
        x = rng(dtype)

      # The random-number generator, if working correctly, should produce the
      # same output multiple times with low probability.
      y = sess.run(x)
      z = sess.run(x)
      w = sess.run(x)

      # We use exact equality here. If the random-number generator is producing
      # deterministic output, all three outputs will be bitwise identical.
      self.assertTrue((not np.array_equal(y, z)) or
                      (not np.array_equal(z, w)) or (not np.array_equal(y, w)))

  def testRandomUniformIsNotConstant(self):

    def rng(dtype):
      return random_ops.random_uniform(shape=[2], dtype=dtype, maxval=1000000)

    for dtype in self._random_types():
      self._testRngIsNotConstant(rng, dtype)

  def testRandomNormalIsNotConstant(self):

    def rng(dtype):
      return random_ops.random_normal(shape=[2], dtype=dtype)

    # TODO(b/34339814): implement inverse erf support for non-F32 types.
    dtype = dtypes.float32
    self._testRngIsNotConstant(rng, dtype)

  def testRandomUniformIsInRange(self):
    for dtype in self._random_types():
      with self.test_session() as sess:
        with self.test_scope():
          x = random_ops.random_uniform(
              shape=[1000], dtype=dtype, minval=-2, maxval=33)
        y = sess.run(x)
        self.assertTrue((y >= -2).sum() == 1000)
        self.assertTrue((y < 33).sum() == 1000)

  def testTruncatedNormalIsNotConstant(self):

    def rng(dtype):
      return random_ops.truncated_normal(shape=[2], dtype=dtype)

    # TODO(b/34339814): implement inverse erf support for non-F32 types.
    self._testRngIsNotConstant(rng, dtypes.float32)

  def testTruncatedNormalIsInRange(self):
    count = 10000000
    # TODO(b/34339814): implement inverse erf support for non-F32 types.
    for dtype in [dtypes.float32]:
      with self.test_session() as sess:
        with self.test_scope():
          x = random_ops.truncated_normal(shape=[count], dtype=dtype, seed=42)
        y = sess.run(x)

        def normal_cdf(x):
          return .5 * math.erfc(-x / math.sqrt(2))

        def normal_pdf(x):
          return math.exp(-(x**2) / 2.) / math.sqrt(2 * math.pi)

        def probit(x, sess=sess):
          return sess.run(special_math.ndtri(x))

        a = -2.
        b = 2.
        mu = 0.
        sigma = 1.

        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        z = normal_cdf(beta) - normal_cdf(alpha)

        self.assertTrue((y >= a).sum() == count)
        self.assertTrue((y <= b).sum() == count)

        # For more information on these calculations, see:
        # Burkardt, John. "The Truncated Normal Distribution".
        # Department of Scientific Computing website. Florida State University.
        expected_mean = mu + (normal_pdf(alpha) - normal_pdf(beta)) / z * sigma
        actual_mean = np.mean(y)
        self.assertAllClose(actual_mean, expected_mean, atol=2e-4)

        expected_median = mu + probit(
            (normal_cdf(alpha) + normal_cdf(beta)) / 2.) * sigma
        actual_median = np.median(y)
        self.assertAllClose(actual_median, expected_median, atol=8e-4)

        expected_variance = sigma**2 * (1 + (
            (alpha * normal_pdf(alpha) - beta * normal_pdf(beta)) / z) - (
                (normal_pdf(alpha) - normal_pdf(beta)) / z)**2)
        actual_variance = np.var(y)
        self.assertAllClose(actual_variance, expected_variance, rtol=3e-4)

  def testShuffle1d(self):
    with self.test_session() as sess:
      with self.test_scope():
        x = math_ops.range(20)
        shuffle = random_ops.random_shuffle(x)
      result = sess.run(shuffle)
      expected = range(20)
      # Compare sets to avoid randomness behavior changes but make sure still
      # have all the values.
      self.assertAllEqual(set(result), set(expected))

  def testShuffle2d(self):
    with self.test_session() as sess:
      with self.test_scope():
        x = array_ops.diag(math_ops.range(20))
        shuffle = random_ops.random_shuffle(x)
      result = sess.run(shuffle)
      expected = np.diag(range(20)).flatten()
      # Compare sets to avoid randomness behavior changes but make sure still
      # have all the values.
      self.assertAllEqual(len(result.flatten()), len(expected))
      self.assertAllEqual(set(result.flatten()), set(expected))


if __name__ == '__main__':
  googletest.main()
