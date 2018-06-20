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
"""Tests for stateless random-number generation ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from tensorflow.compiler.tests.xla_test import XLATestCase
from tensorflow.contrib import stateless
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class StatelessRandomOpsTest(XLATestCase):
  """Test cases for stateless random-number generator operators."""

  def _random_types(self):
    return [dtypes.float32]

  def testDeterminism(self):
    # Stateless values should be equal iff the seeds are equal (roughly)
    with self.test_session(), self.test_scope():
      seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
      seeds = [(x, y) for x in range(5) for y in range(5)] * 3
      for stateless_op in [
          stateless.stateless_random_uniform, stateless.stateless_random_normal
      ]:
        for shape in (), (3,), (2, 5):
          for dtype in self._random_types():
            pure = stateless_op(shape, seed=seed_t, dtype=dtype)
            values = [(seed, pure.eval(feed_dict={
                seed_t: seed
            })) for seed in seeds]
            for s0, v0 in values:
              for s1, v1 in values:
                self.assertEqual(s0 == s1, np.all(v0 == v1))

  def testRandomUniformIsInRange(self):
    with self.test_session() as sess, self.test_scope():
      for dtype in self._random_types():
        seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
        x = stateless.stateless_random_uniform(
            shape=[1000], seed=seed_t, dtype=dtype)
        y = sess.run(x, {seed_t: [0x12345678, 0xabcdef12]})
        self.assertTrue(np.all(y >= 0))
        self.assertTrue(np.all(y < 1))

  def _chi_squared(self, x, bins):
    """Pearson's Chi-squared test."""
    x = np.ravel(x)
    n = len(x)
    histogram, _ = np.histogram(x, bins=bins, range=(0, 1))
    expected = n / float(bins)
    return np.sum(np.square(histogram - expected) / expected)

  def testDistributionOfStatelessRandomUniform(self):
    """Use Pearson's Chi-squared test to test for uniformity."""
    with self.test_session() as sess, self.test_scope():
      for dtype in self._random_types():
        seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
        n = 1000
        x = stateless.stateless_random_uniform(
            shape=[n], seed=seed_t, dtype=dtype)
        y = sess.run(x, {seed_t: [565656, 121212]})
        # Tests that the values are distributed amongst 10 bins with equal
        # probability. 16.92 is the Chi^2 value for 9 degrees of freedom with
        # p=0.05. This test is probabilistic and would be flaky if the random
        # seed were not fixed.
        self.assertTrue(self._chi_squared(y, 10) < 16.92)

  def testRandomNormalIsFinite(self):
    with self.test_session() as sess, self.test_scope():
      for dtype in self._random_types():
        seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
        x = stateless.stateless_random_uniform(
            shape=[10000], seed=seed_t, dtype=dtype)
        y = sess.run(x, {seed_t: [0x12345678, 0xabcdef12]})
        self.assertTrue(np.all(np.isfinite(y)))

  def _normal_cdf(self, x):
    """Cumulative distribution function for a standard normal distribution."""
    return 0.5 + 0.5 * np.vectorize(math.erf)(x / math.sqrt(2))

  def _anderson_darling(self, x):
    """Anderson-Darling test for a standard normal distribution."""
    x = np.sort(np.ravel(x))
    n = len(x)
    i = np.linspace(1, n, n)
    z = np.sum((2 * i - 1) * np.log(self._normal_cdf(x)) +
               (2 * (n - i) + 1) * np.log(1 - self._normal_cdf(x)))
    return -n - z / n

  def testDistributionOfStatelessRandomNormal(self):
    """Use Anderson-Darling test to test distribution appears normal."""
    with self.test_session() as sess, self.test_scope():
      for dtype in self._random_types():
        seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
        n = 1000
        x = stateless.stateless_random_normal(
            shape=[n], seed=seed_t, dtype=dtype)
        y = sess.run(x, {seed_t: [25252, 314159]})
        # The constant 2.492 is the 5% critical value for the Anderson-Darling
        # test where the mean and variance are known. This test is probabilistic
        # so to avoid flakiness the seed is fixed.
        self.assertTrue(self._anderson_darling(y) < 2.492)


if __name__ == '__main__':
  test.main()
