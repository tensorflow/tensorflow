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

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.kernel_tests.random import util as \
random_test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import stateless_random_ops as stateless
from tensorflow.python.platform import test


class StatelessRandomOpsTest(xla_test.XLATestCase):
  """Test cases for stateless random-number generator operators."""

  def _random_types(self, include_int=False):
    allowed_types = {dtypes.float32, dtypes.float64, dtypes.bfloat16}
    if include_int:
      allowed_types.update({dtypes.int32, dtypes.int64})
    return self.all_tf_types & allowed_types

  def testDeterminism(self):
    # Stateless values should be equal iff the seeds are equal (roughly)
    with self.cached_session(), self.test_scope():
      seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
      seeds = [(x, y) for x in range(5) for y in range(5)] * 3  # pylint: disable=g-complex-comprehension
      for stateless_op in [
          stateless.stateless_random_uniform, stateless.stateless_random_normal
      ]:
        for shape in (), (3,), (2, 5):
          for dtype in self._random_types():
            # Skip bfloat16. The result of bfloat16 is truncated from 32-bit
            # result. With different seeds, the 32-bit results are different,
            # but the truncated 16-bit results might be the same.
            if dtype == dtypes.bfloat16:
              continue
            pure = stateless_op(shape, seed=seed_t, dtype=dtype)
            values = [(seed, pure.eval(feed_dict={
                seed_t: seed
            })) for seed in seeds]
            for s0, v0 in values:
              for s1, v1 in values:
                self.assertEqual(s0 == s1, np.all(v0 == v1))

  def testRandomUniformIsInRange(self):
    with self.cached_session() as sess, self.test_scope():
      for dtype in self._random_types(include_int=True):
        maxval = 1
        if dtype.is_integer:
          maxval = 100
        seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
        x = stateless.stateless_random_uniform(
            shape=[1000], seed=seed_t, maxval=maxval, dtype=dtype)
        y = sess.run(x, {seed_t: [0x12345678, 0xabcdef12]})
        self.assertTrue(np.all(y >= 0))
        self.assertTrue(np.all(y < maxval))

  def testDistributionOfStatelessRandomUniform(self):
    """Use Pearson's Chi-squared test to test for uniformity."""
    with self.cached_session() as sess, self.test_scope():
      for dtype in self._random_types(include_int=True):
        seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
        n = 1000
        maxval = 1
        if dtype.is_integer:
          maxval = 100
        x = stateless.stateless_random_uniform(
            shape=[n], seed=seed_t, maxval=maxval, dtype=dtype)
        y = sess.run(x, {seed_t: [565656, 121212]})
        if maxval > 1:
          # Normalize y to range [0, 1).
          y = y.astype(float) / maxval
        # Tests that the values are distributed amongst 10 bins with equal
        # probability. 16.92 is the Chi^2 value for 9 degrees of freedom with
        # p=0.05. This test is probabilistic and would be flaky if the random
        # seed were not fixed.
        self.assertLess(random_test_util.chi_squared(y, 10), 16.92)

  def testRandomNormalIsFinite(self):
    with self.cached_session() as sess, self.test_scope():
      for dtype in self._random_types():
        seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
        x = stateless.stateless_random_normal(
            shape=[10000], seed=seed_t, dtype=dtype)
        y = sess.run(x, {seed_t: [0x12345678, 0xabcdef12]})
        self.assertTrue(np.all(np.isfinite(y)))

  def testDistributionOfStatelessRandomNormal(self):
    """Use Anderson-Darling test to test distribution appears normal."""
    with self.cached_session() as sess, self.test_scope():
      for dtype in self._random_types():
        seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
        n = 1000
        x = stateless.stateless_random_normal(
            shape=[n], seed=seed_t, dtype=dtype)
        y = sess.run(x, {seed_t: [25252, 314159]})
        # The constant 2.492 is the 5% critical value for the Anderson-Darling
        # test where the mean and variance are known. This test is probabilistic
        # so to avoid flakiness the seed is fixed.
        self.assertLess(
            random_test_util.anderson_darling(y.astype(float)), 2.492)

  def testTruncatedNormal(self):
    for dtype in self._random_types():
      with self.cached_session() as sess, self.test_scope():
        seed_t = array_ops.placeholder(dtypes.int32, shape=[2])
        n = 10000000
        x = stateless.stateless_truncated_normal(
            shape=[n], seed=seed_t, dtype=dtype)
        y = sess.run(x, {seed_t: [0x12345678, 0xabcdef12]})
        random_test_util.test_truncated_normal(
            self.assertEqual, self.assertAllClose, dtype, n, y)


if __name__ == '__main__':
  test.main()
