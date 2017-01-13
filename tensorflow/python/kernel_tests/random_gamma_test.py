# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.random_ops.random_gamma."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


class RandomGammaTest(test.TestCase):
  """This is a medium test due to the moments computation taking some time."""

  def setUp(self):
    np.random.seed(137)
    random_seed.set_random_seed(137)

  def _Sampler(self, num, alpha, beta, dtype, use_gpu, seed=None):

    def func():
      with self.test_session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
        rng = random_ops.random_gamma(
            [num], alpha, beta=beta, dtype=dtype, seed=seed)
        ret = np.empty([10, num])
        for i in xrange(10):
          ret[i, :] = sess.run(rng)
      return ret

    return func

  def testMomentsFloat32(self):
    self._testMoments(dtypes.float32)

  def testMomentsFloat64(self):
    self._testMoments(dtypes.float64)

  def _testMoments(self, dt):
    try:
      from scipy import stats  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      tf_logging.warn("Cannot test moments: %s" % e)
      return

    # Check the given array of samples matches the given theoretical moment
    # function at different orders. The test is considered passing if the
    # z-tests of all statistical moments are all below z_limit.
    # Parameters:
    #   max_moments: the largest moments of the distribution to be tested
    #   stride: the distance between samples to check for statistical properties
    #       0 means the n-th moment of each sample
    #       any other strides tests for spatial correlation between samples;
    #   z_limit: the maximum z-test we would consider the test to pass;

    # The moments test is a z-value test.  This is the largest z-value
    # we want to tolerate. Since the z-test approximates a unit normal
    # distribution, it should almost definitely never exceed 6.
    z_limit = 6.0

    for stride in 0, 1, 4, 17:
      alphas = [0.2, 1.0, 3.0]
      if dt == dtypes.float64:
        alphas = [0.01] + alphas
      for alpha in alphas:
        for scale in 9, 17:
          # Gamma moments only defined for values less than the scale param.
          max_moment = min(6, scale // 2)
          sampler = self._Sampler(
              20000, alpha, 1 / scale, dt, use_gpu=False, seed=12345)
          moments = [0] * (max_moment + 1)
          moments_sample_count = [0] * (max_moment + 1)
          x = np.array(sampler().flat)  # sampler does 10x samples
          for k in range(len(x)):
            moment = 1.
            for i in range(max_moment + 1):
              index = k + i * stride
              if index >= len(x):
                break
              moments[i] += moment
              moments_sample_count[i] += 1
              moment *= x[index]
          for i in range(max_moment + 1):
            moments[i] /= moments_sample_count[i]
          for i in range(1, max_moment + 1):
            g = stats.gamma(alpha, scale=scale)
            if stride == 0:
              moments_i_mean = g.moment(i)
              moments_i_squared = g.moment(2 * i)
            else:
              moments_i_mean = pow(g.moment(1), i)
              moments_i_squared = pow(g.moment(2), i)
            # Calculate moment variance safely:
            # This is just
            #  (moments_i_squared - moments_i_mean**2) / moments_sample_count[i]
            normalized_moments_i_var = (
                moments_i_mean / moments_sample_count[i] *
                (moments_i_squared / moments_i_mean - moments_i_mean))
            # Assume every operation has a small numerical error.
            # It takes i multiplications to calculate one i-th moment.
            error_per_moment = i * np.finfo(dt.as_numpy_dtype).eps
            total_variance = (normalized_moments_i_var + error_per_moment)
            tiny = np.finfo(dt.as_numpy_dtype).tiny
            self.assertGreaterEqual(total_variance, 0)
            if total_variance < tiny:
              total_variance = tiny
            # z_test is approximately a unit normal distribution.
            z_test = abs(
                (moments[i] - moments_i_mean) / math.sqrt(total_variance))
            self.assertLess(z_test, z_limit)

  def _testZeroDensity(self, alpha):
    """Zero isn't in the support of the gamma distribution.

    But quantized floating point math has its limits.
    TODO(bjp): Implement log-gamma sampler for small-shape distributions.

    Args:
      alpha: float shape value to test
    """
    try:
      from scipy import stats  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      tf_logging.warn("Cannot test zero density proportions: %s" % e)
      return
    allowable_zeros = {
        dtypes.float16: stats.gamma(alpha).cdf(np.finfo(np.float16).tiny),
        dtypes.float32: stats.gamma(alpha).cdf(np.finfo(np.float32).tiny),
        dtypes.float64: stats.gamma(alpha).cdf(np.finfo(np.float64).tiny)
    }
    failures = []
    for use_gpu in [False, True]:
      for dt in dtypes.float16, dtypes.float32, dtypes.float64:
        sampler = self._Sampler(
            10000, alpha, 1.0, dt, use_gpu=use_gpu, seed=12345)
        x = sampler()
        allowable = allowable_zeros[dt] * x.size
        allowable = allowable * 2 if allowable < 10 else allowable * 1.05
        if np.sum(x <= 0) > allowable:
          failures += [(use_gpu, dt)]
      self.assertEqual([], failures)

  def testNonZeroSmallShape(self):
    self._testZeroDensity(0.01)

  def testNonZeroSmallishShape(self):
    self._testZeroDensity(0.35)

  # Asserts that different trials (1000 samples per trial) is unlikely
  # to see the same sequence of values. Will catch buggy
  # implementations which uses the same random number seed.
  def testDistinct(self):
    for use_gpu in [False, True]:
      for dt in dtypes.float16, dtypes.float32, dtypes.float64:
        sampler = self._Sampler(1000, 2.0, 1.0, dt, use_gpu=use_gpu)
        x = sampler()
        y = sampler()
        # Number of different samples.
        count = (x == y).sum()
        count_limit = 20 if dt == dtypes.float16 else 10
        if count >= count_limit:
          print(use_gpu, dt)
          print("x = ", x)
          print("y = ", y)
          print("count = ", count)
        self.assertLess(count, count_limit)

  # Checks that the CPU and GPU implementation returns the same results,
  # given the same random seed
  def testCPUGPUMatch(self):
    for dt in dtypes.float16, dtypes.float32, dtypes.float64:
      results = {}
      for use_gpu in [False, True]:
        sampler = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=use_gpu, seed=12345)
        results[use_gpu] = sampler()
      if dt == dtypes.float16:
        self.assertAllClose(results[False], results[True], rtol=1e-3, atol=1e-3)
      else:
        self.assertAllClose(results[False], results[True], rtol=1e-6, atol=1e-6)

  def testSeed(self):
    for use_gpu in [False, True]:
      for dt in dtypes.float16, dtypes.float32, dtypes.float64:
        sx = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=use_gpu, seed=345)
        sy = self._Sampler(1000, 0.0, 1.0, dt, use_gpu=use_gpu, seed=345)
        self.assertAllEqual(sx(), sy())

  def testNoCSE(self):
    """CSE = constant subexpression eliminator.

    SetIsStateful() should prevent two identical random ops from getting
    merged.
    """
    for dtype in dtypes.float16, dtypes.float32, dtypes.float64:
      for use_gpu in [False, True]:
        with self.test_session(use_gpu=use_gpu):
          rnd1 = random_ops.random_gamma([24], 2.0, dtype=dtype)
          rnd2 = random_ops.random_gamma([24], 2.0, dtype=dtype)
          diff = rnd2 - rnd1
          self.assertGreater(np.linalg.norm(diff.eval()), 0.1)

  def testShape(self):
    # Fully known shape.
    rnd = random_ops.random_gamma([150], 2.0)
    self.assertEqual([150], rnd.get_shape().as_list())
    rnd = random_ops.random_gamma([150], 2.0, beta=[3.0, 4.0])
    self.assertEqual([150, 2], rnd.get_shape().as_list())
    rnd = random_ops.random_gamma([150], array_ops.ones([1, 2, 3]))
    self.assertEqual([150, 1, 2, 3], rnd.get_shape().as_list())
    rnd = random_ops.random_gamma([20, 30], array_ops.ones([1, 2, 3]))
    self.assertEqual([20, 30, 1, 2, 3], rnd.get_shape().as_list())
    rnd = random_ops.random_gamma(
        [123], array_ops.placeholder(
            dtypes.float32, shape=(2,)))
    self.assertEqual([123, 2], rnd.get_shape().as_list())
    # Partially known shape.
    rnd = random_ops.random_gamma(
        array_ops.placeholder(
            dtypes.int32, shape=(1,)), array_ops.ones([7, 3]))
    self.assertEqual([None, 7, 3], rnd.get_shape().as_list())
    rnd = random_ops.random_gamma(
        array_ops.placeholder(
            dtypes.int32, shape=(3,)), array_ops.ones([9, 6]))
    self.assertEqual([None, None, None, 9, 6], rnd.get_shape().as_list())
    # Unknown shape.
    rnd = random_ops.random_gamma(
        array_ops.placeholder(dtypes.int32),
        array_ops.placeholder(dtypes.float32))
    self.assertIs(None, rnd.get_shape().ndims)
    rnd = random_ops.random_gamma([50], array_ops.placeholder(dtypes.float32))
    self.assertIs(None, rnd.get_shape().ndims)


if __name__ == "__main__":
  test.main()
