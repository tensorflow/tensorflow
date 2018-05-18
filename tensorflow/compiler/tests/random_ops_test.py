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

import numpy as np

from tensorflow.compiler.tests.xla_test import XLATestCase
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import googletest


class RandomOpsTest(XLATestCase):
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
                      (not np.array_equal(z, w)) or
                      (not np.array_equal(y, w)))

  def testRandomUniformIsNotConstant(self):
    def rng(dtype):
      return random_ops.random_uniform(shape=[2], dtype=dtype,
                                       maxval=1000000)

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
          x = random_ops.random_uniform(shape=[1000], dtype=dtype, minval=-2,
                                        maxval=33)
        y = sess.run(x)
        self.assertTrue((y >= -2).sum() == 1000)
        self.assertTrue((y < 33).sum() == 1000)

  def testTruncatedNormalIsInRange(self):
    count = 10000
    # TODO(b/34339814): implement inverse erf support for non-F32 types.
    for dtype in [dtypes.float32]:
      with self.test_session() as sess:
        with self.test_scope():
          x = random_ops.truncated_normal(shape=[count], dtype=dtype, seed=42)
        y = sess.run(x)
        self.assertTrue((y >= -2).sum() == count)
        self.assertTrue((y <= 2).sum() == count)


if __name__ == '__main__':
  googletest.main()
