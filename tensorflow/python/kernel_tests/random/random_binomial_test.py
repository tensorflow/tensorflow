# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.stateful_random_ops.binomial."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.random import util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import stateful_random_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

# All supported dtypes for binomial().
_SUPPORTED_DTYPES = (dtypes.float16, dtypes.float32, dtypes.float64,
                     dtypes.int32, dtypes.int64)


class RandomBinomialTest(test.TestCase):
  """This is a large test due to the moments computation taking some time."""

  def _Sampler(self, num, counts, probs, dtype, seed=None):

    def func():
      rng = stateful_random_ops.Generator(seed=seed).binomial(
          shape=[10 * num], counts=counts, probs=probs, dtype=dtype)
      ret = array_ops.reshape(rng, [10, num])
      ret = self.evaluate(ret)
      return ret

    return func

  @test_util.run_v2_only
  def testMoments(self):
    try:
      from scipy import stats  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      tf_logging.warn("Cannot test moments: %s", e)
      return
    # The moments test is a z-value test.  This is the largest z-value
    # we want to tolerate. Since the z-test approximates a unit normal
    # distribution, it should almost definitely never exceed 6.
    z_limit = 6.0
    for dt in _SUPPORTED_DTYPES:
      # Test when n * p > 10, and n * p < 10
      for stride in 0, 4, 10:
        for counts in (1., 10., 22., 50.):
          for prob in (0.1, 0.5, 0.8):
            sampler = self._Sampler(int(1e5), counts, prob, dt, seed=12345)
            z_scores = util.test_moment_matching(
                # Use float64 samples.
                sampler().astype(np.float64),
                number_moments=6,
                dist=stats.binom(counts, prob),
                stride=stride,
            )
            self.assertAllLess(z_scores, z_limit)

  @test_util.run_v2_only
  def testSeed(self):
    for dt in dtypes.float16, dtypes.float32, dtypes.float64:
      sx = self._Sampler(1000, counts=10., probs=0.4, dtype=dt, seed=345)
      sy = self._Sampler(1000, counts=10., probs=0.4, dtype=dt, seed=345)
      self.assertAllEqual(sx(), sy())

  def testZeroShape(self):
    rnd = stateful_random_ops.Generator(seed=12345).binomial([0], [], [])
    self.assertEqual([0], rnd.shape.as_list())

  def testShape(self):
    rng = stateful_random_ops.Generator(seed=12345)
    # Scalar parameters.
    rnd = rng.binomial(shape=[10], counts=np.float32(2.), probs=np.float32(0.5))
    self.assertEqual([10], rnd.shape.as_list())

    # Vector parameters.
    rnd = rng.binomial(
        shape=[10],
        counts=array_ops.ones([10], dtype=np.float32),
        probs=0.3 * array_ops.ones([10], dtype=np.float32))
    self.assertEqual([10], rnd.shape.as_list())
    rnd = rng.binomial(
        shape=[2, 5],
        counts=array_ops.ones([2], dtype=np.float32),
        probs=0.4 * array_ops.ones([2], dtype=np.float32))
    self.assertEqual([2, 5], rnd.shape.as_list())

    # Scalar counts, vector probs.
    rnd = rng.binomial(
        shape=[10],
        counts=np.float32(5.),
        probs=0.8 * array_ops.ones([10], dtype=np.float32))
    self.assertEqual([10], rnd.shape.as_list())

    # Vector counts, scalar probs.
    rnd = rng.binomial(
        shape=[10],
        counts=array_ops.ones([10], dtype=np.float32),
        probs=np.float32(0.9))
    self.assertEqual([10], rnd.shape.as_list())


if __name__ == "__main__":
  test.main()
