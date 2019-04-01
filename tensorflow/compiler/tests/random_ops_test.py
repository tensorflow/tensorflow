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
    return set(self.numeric_types) - set(
        self.complex_types) - {np.uint64, np.int64, np.uint8, np.int8}

  def _testRngIsNotConstant(self, rng, dtype):
    # Tests that 'rng' does not always return the same value.
    with self.cached_session() as sess:
      with self.test_scope():
        x = rng(dtype)

      # The random-number generator, if working correctly, should produce the
      # same output multiple times with low probability.
      y = self.evaluate(x)
      z = self.evaluate(x)
      w = self.evaluate(x)

      # We use exact equality here. If the random-number generator is producing
      # deterministic output, all three outputs will be bitwise identical.
      self.assertTrue((not np.array_equal(y, z)) or
                      (not np.array_equal(z, w)) or (not np.array_equal(y, w)))

  def testRandomUniformIsNotConstant(self):

    def rng(dtype):
      dtype = dtypes.as_dtype(dtype)
      return random_ops.random_uniform(shape=[2], dtype=dtype, maxval=dtype.max)

    for dtype in self._random_types():
      self._testRngIsNotConstant(rng, dtype)

  def testRandomNormalIsNotConstant(self):

    def rng(dtype):
      return random_ops.random_normal(shape=[2], dtype=dtype)

    for dtype in self._random_types() & self.float_types:
      self._testRngIsNotConstant(rng, dtype)

  def testRandomUniformIsInRange(self):
    for dtype in self._random_types():
      # TODO (b/112272078): enable bfloat16 for CPU and GPU when the bug is
      # fixed.
      if (self.device in ["XLA_GPU", "XLA_CPU"
                         ]) and (dtype in [dtypes.bfloat16, dtypes.half]):
        continue
      with self.cached_session() as sess:
        with self.test_scope():
          x = random_ops.random_uniform(
              shape=[1000], dtype=dtype, minval=-2, maxval=33)
        y = self.evaluate(x)
        self.assertTrue((y >= -2).sum() == 1000)
        self.assertTrue((y < 33).sum() == 1000)

  def testTruncatedNormalIsNotConstant(self):

    def rng(dtype):
      return random_ops.truncated_normal(shape=[2], dtype=dtype)

    for dtype in self._random_types() & self.float_types:
      self._testRngIsNotConstant(rng, dtype)

  def testTruncatedNormalIsInRange(self):
    count = 10000000
    # TODO(b/34339814): make this test work with 16 bit float types.
    for dtype in self._random_types() & {dtypes.float32, dtypes.float64}:
      with self.cached_session() as sess:
        with self.test_scope():
          x = random_ops.truncated_normal(shape=[count], dtype=dtype)
        y = self.evaluate(x)

        def normal_cdf(x):
          return .5 * math.erfc(-x / math.sqrt(2))

        def normal_pdf(x):
          return math.exp(-(x**2) / 2.) / math.sqrt(2 * math.pi)

        def probit(x, sess=sess):
          return self.evaluate(special_math.ndtri(x))

        a = -2.
        b = 2.
        mu = 0.
        sigma = 1.

        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        z = normal_cdf(beta) - normal_cdf(alpha)

        self.assertEqual((y >= a).sum(), count)
        self.assertEqual((y <= b).sum(), count)

        # For more information on these calculations, see:
        # Burkardt, John. "The Truncated Normal Distribution".
        # Department of Scientific Computing website. Florida State University.
        expected_mean = mu + (normal_pdf(alpha) - normal_pdf(beta)) / z * sigma
        actual_mean = np.mean(y)
        self.assertAllClose(actual_mean, expected_mean, atol=2e-3)

        expected_median = mu + probit(
            (normal_cdf(alpha) + normal_cdf(beta)) / 2.) * sigma
        actual_median = np.median(y)
        self.assertAllClose(actual_median, expected_median, atol=1e-2)

        expected_variance = sigma**2 * (1 + (
            (alpha * normal_pdf(alpha) - beta * normal_pdf(beta)) / z) - (
                (normal_pdf(alpha) - normal_pdf(beta)) / z)**2)
        actual_variance = np.var(y)
        self.assertAllClose(actual_variance, expected_variance, rtol=2*1e-3)

  def testShuffle1d(self):
    with self.cached_session() as sess:
      with self.test_scope():
        x = math_ops.range(1 << 16)
        shuffle = random_ops.random_shuffle(x)
      result = self.evaluate(shuffle)
      expected = range(1 << 16)
      # Compare sets to avoid randomness behavior changes but make sure still
      # have all the values.
      self.assertAllEqual(set(result), set(expected))

  def testShuffle2d(self):
    with self.cached_session() as sess:
      with self.test_scope():
        x = array_ops.diag(math_ops.range(20))
        shuffle = random_ops.random_shuffle(x)
      result = self.evaluate(shuffle)
      expected = np.diag(range(20)).flatten()
      # Compare sets to avoid randomness behavior changes but make sure still
      # have all the values.
      self.assertAllEqual(len(result.flatten()), len(expected))
      self.assertAllEqual(set(result.flatten()), set(expected))


if __name__ == '__main__':
  googletest.main()
