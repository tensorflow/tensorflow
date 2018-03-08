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
"""Tests for tensorflow.ops.random_ops.random_poisson."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

# All supported dtypes for random_poisson().
_SUPPORTED_DTYPES = (dtypes.float16, dtypes.float32, dtypes.float64,
                     dtypes.int32, dtypes.int64)


class RandomPoissonTest(test.TestCase):
  """This is a large test due to the moments computation taking some time."""

  def _Sampler(self, num, lam, dtype, use_gpu, seed=None):

    def func():
      with self.test_session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
        rng = random_ops.random_poisson(lam, [num], dtype=dtype, seed=seed)
        ret = np.empty([10, num])
        for i in xrange(10):
          ret[i, :] = sess.run(rng)
      return ret

    return func

  # TODO(srvasude): Factor this out along with the corresponding moment testing
  # method in random_gamma_test into a single library.
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
      # Test when lam < 10 and when lam >= 10
      for stride in 0, 4, 10:
        for lam in (3., 20):
          max_moment = 5
          sampler = self._Sampler(10000, lam, dt, use_gpu=False, seed=12345)
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
            g = stats.poisson(lam)
            if stride == 0:
              moments_i_mean = g.moment(i)
              moments_i_squared = g.moment(2 * i)
            else:
              moments_i_mean = pow(g.moment(1), i)
              moments_i_squared = pow(g.moment(2), i)
            moments_i_var = (
                moments_i_squared - moments_i_mean * moments_i_mean)
            # Assume every operation has a small numerical error.
            # It takes i multiplications to calculate one i-th moment.
            error_per_moment = i * 1e-6
            total_variance = (
                moments_i_var / moments_sample_count[i] + error_per_moment)
            if not total_variance:
              total_variance = 1e-10
            # z_test is approximately a unit normal distribution.
            z_test = abs(
                (moments[i] - moments_i_mean) / np.sqrt(total_variance))
            self.assertLess(z_test, z_limit)

  # Checks that the CPU and GPU implementation returns the same results,
  # given the same random seed
  def testCPUGPUMatch(self):
    for dt in _SUPPORTED_DTYPES:
      results = {}
      for use_gpu in [False, True]:
        sampler = self._Sampler(1000, 1.0, dt, use_gpu=use_gpu, seed=12345)
        results[use_gpu] = sampler()
      if dt == dtypes.float16:
        self.assertAllClose(results[False], results[True], rtol=1e-3, atol=1e-3)
      else:
        self.assertAllClose(results[False], results[True], rtol=1e-6, atol=1e-6)

  def testSeed(self):
    for dt in dtypes.float16, dtypes.float32, dtypes.float64:
      sx = self._Sampler(1000, 1.0, dt, use_gpu=True, seed=345)
      sy = self._Sampler(1000, 1.0, dt, use_gpu=True, seed=345)
      self.assertAllEqual(sx(), sy())

  def testNoCSE(self):
    """CSE = constant subexpression eliminator.

    SetIsStateful() should prevent two identical random ops from getting
    merged.
    """
    for dtype in dtypes.float16, dtypes.float32, dtypes.float64:
      with self.test_session(use_gpu=True):
        rnd1 = random_ops.random_poisson(2.0, [24], dtype=dtype)
        rnd2 = random_ops.random_poisson(2.0, [24], dtype=dtype)
        diff = rnd2 - rnd1
        # Since these are all positive integers, the norm will
        # be at least 1 if they are different.
        self.assertGreaterEqual(np.linalg.norm(diff.eval()), 1)

  def testZeroShape(self):
    with self.test_session():
      rnd = random_ops.random_poisson([], [], seed=12345)
      self.assertEqual([0], rnd.get_shape().as_list())
      self.assertAllClose(np.array([], dtype=np.float32), rnd.eval())

  def testShape(self):
    # Fully known shape
    rnd = random_ops.random_poisson(2.0, [150], seed=12345)
    self.assertEqual([150], rnd.get_shape().as_list())
    rnd = random_ops.random_poisson(
        lam=array_ops.ones([1, 2, 3]),
        shape=[150],
        seed=12345)
    self.assertEqual([150, 1, 2, 3], rnd.get_shape().as_list())
    rnd = random_ops.random_poisson(
        lam=array_ops.ones([1, 2, 3]),
        shape=[20, 30],
        seed=12345)
    self.assertEqual([20, 30, 1, 2, 3], rnd.get_shape().as_list())
    rnd = random_ops.random_poisson(
        lam=array_ops.placeholder(dtypes.float32, shape=(2,)),
        shape=[12],
        seed=12345)
    self.assertEqual([12, 2], rnd.get_shape().as_list())
    # Partially known shape.
    rnd = random_ops.random_poisson(
        lam=array_ops.ones([7, 3]),
        shape=array_ops.placeholder(dtypes.int32, shape=(1,)),
        seed=12345)
    self.assertEqual([None, 7, 3], rnd.get_shape().as_list())
    rnd = random_ops.random_poisson(
        lam=array_ops.ones([9, 6]),
        shape=array_ops.placeholder(dtypes.int32, shape=(3,)),
        seed=12345)
    self.assertEqual([None, None, None, 9, 6], rnd.get_shape().as_list())
    # Unknown shape.
    rnd = random_ops.random_poisson(
        lam=array_ops.placeholder(dtypes.float32),
        shape=array_ops.placeholder(dtypes.int32),
        seed=12345)
    self.assertIs(None, rnd.get_shape().ndims)
    rnd = random_ops.random_poisson(
        lam=array_ops.placeholder(dtypes.float32),
        shape=[50],
        seed=12345)
    self.assertIs(None, rnd.get_shape().ndims)

  def testDTypeCombinationsV2(self):
    """Tests random_poisson_v2() for all supported dtype combinations."""
    with self.test_session():
      for lam_dt in _SUPPORTED_DTYPES:
        for out_dt in _SUPPORTED_DTYPES:
          random_ops.random_poisson(
              constant_op.constant([1], dtype=lam_dt), [10],
              dtype=out_dt).eval()


if __name__ == "__main__":
  test.main()
