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
"""Tests for initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats
from tensorflow.contrib.distributions.python.ops import logistic
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.platform import test


class LogisticTest(test.TestCase):

  def testReparameterizable(self):
    batch_size = 6
    np_loc = np.array([2.0] * batch_size, dtype=np.float32)
    loc = constant_op.constant(np_loc)
    scale = 1.5
    dist = logistic.Logistic(loc, scale)
    self.assertTrue(
        dist.reparameterization_type == distribution.FULLY_REPARAMETERIZED)

  def testLogisticLogProb(self):
    with self.cached_session():
      batch_size = 6
      np_loc = np.array([2.0] * batch_size, dtype=np.float32)
      loc = constant_op.constant(np_loc)
      scale = 1.5
      x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
      dist = logistic.Logistic(loc, scale)
      expected_log_prob = stats.logistic.logpdf(x, np_loc, scale)

      log_prob = dist.log_prob(x)
      self.assertEqual(log_prob.get_shape(), (6,))
      self.assertAllClose(log_prob.eval(), expected_log_prob)

      prob = dist.prob(x)
      self.assertEqual(prob.get_shape(), (6,))
      self.assertAllClose(prob.eval(), np.exp(expected_log_prob))

  def testLogisticCDF(self):
    with self.cached_session():
      batch_size = 6
      np_loc = np.array([2.0] * batch_size, dtype=np.float32)
      loc = constant_op.constant(np_loc)
      scale = 1.5

      dist = logistic.Logistic(loc, scale)
      x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
      cdf = dist.cdf(x)
      expected_cdf = stats.logistic.cdf(x, np_loc, scale)

      self.assertEqual(cdf.get_shape(), (6,))
      self.assertAllClose(cdf.eval(), expected_cdf)

  def testLogisticLogCDF(self):
    with self.cached_session():
      batch_size = 6
      np_loc = np.array([2.0] * batch_size, dtype=np.float32)
      loc = constant_op.constant(np_loc)
      scale = 1.5

      dist = logistic.Logistic(loc, scale)
      x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
      logcdf = dist.log_cdf(x)
      expected_logcdf = stats.logistic.logcdf(x, np_loc, scale)

      self.assertEqual(logcdf.get_shape(), (6,))
      self.assertAllClose(logcdf.eval(), expected_logcdf)

  def testLogisticSurvivalFunction(self):
    with self.cached_session():
      batch_size = 6
      np_loc = np.array([2.0] * batch_size, dtype=np.float32)
      loc = constant_op.constant(np_loc)
      scale = 1.5

      dist = logistic.Logistic(loc, scale)
      x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
      survival_function = dist.survival_function(x)
      expected_survival_function = stats.logistic.sf(x, np_loc, scale)

      self.assertEqual(survival_function.get_shape(), (6,))
      self.assertAllClose(survival_function.eval(), expected_survival_function)

  def testLogisticLogSurvivalFunction(self):
    with self.cached_session():
      batch_size = 6
      np_loc = np.array([2.0] * batch_size, dtype=np.float32)
      loc = constant_op.constant(np_loc)
      scale = 1.5

      dist = logistic.Logistic(loc, scale)
      x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
      logsurvival_function = dist.log_survival_function(x)
      expected_logsurvival_function = stats.logistic.logsf(x, np_loc, scale)

      self.assertEqual(logsurvival_function.get_shape(), (6,))
      self.assertAllClose(logsurvival_function.eval(),
                          expected_logsurvival_function)

  def testLogisticMean(self):
    with self.cached_session():
      loc = [2.0, 1.5, 1.0]
      scale = 1.5
      expected_mean = stats.logistic.mean(loc, scale)
      dist = logistic.Logistic(loc, scale)
      self.assertAllClose(dist.mean().eval(), expected_mean)

  def testLogisticVariance(self):
    with self.cached_session():
      loc = [2.0, 1.5, 1.0]
      scale = 1.5
      expected_variance = stats.logistic.var(loc, scale)
      dist = logistic.Logistic(loc, scale)
      self.assertAllClose(dist.variance().eval(), expected_variance)

  def testLogisticEntropy(self):
    with self.cached_session():
      batch_size = 3
      np_loc = np.array([2.0] * batch_size, dtype=np.float32)
      loc = constant_op.constant(np_loc)
      scale = 1.5
      expected_entropy = stats.logistic.entropy(np_loc, scale)
      dist = logistic.Logistic(loc, scale)
      self.assertAllClose(dist.entropy().eval(), expected_entropy)

  def testLogisticSample(self):
    with self.cached_session():
      loc = [3.0, 4.0, 2.0]
      scale = 1.0
      dist = logistic.Logistic(loc, scale)
      sample = dist.sample(seed=100)
      self.assertEqual(sample.get_shape(), (3,))
      self.assertAllClose(sample.eval(), [6.22460556, 3.79602098, 2.05084133])

  def testDtype(self):
    loc = constant_op.constant([0.1, 0.4], dtype=dtypes.float32)
    scale = constant_op.constant(1.0, dtype=dtypes.float32)
    dist = logistic.Logistic(loc, scale)
    self.assertEqual(dist.dtype, dtypes.float32)
    self.assertEqual(dist.loc.dtype, dist.scale.dtype)
    self.assertEqual(dist.dtype, dist.sample(5).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    self.assertEqual(dist.loc.dtype, dist.mean().dtype)
    self.assertEqual(dist.loc.dtype, dist.variance().dtype)
    self.assertEqual(dist.loc.dtype, dist.stddev().dtype)
    self.assertEqual(dist.loc.dtype, dist.entropy().dtype)
    self.assertEqual(dist.loc.dtype, dist.prob(0.2).dtype)
    self.assertEqual(dist.loc.dtype, dist.log_prob(0.2).dtype)

    loc = constant_op.constant([0.1, 0.4], dtype=dtypes.float64)
    scale = constant_op.constant(1.0, dtype=dtypes.float64)
    dist64 = logistic.Logistic(loc, scale)
    self.assertEqual(dist64.dtype, dtypes.float64)
    self.assertEqual(dist64.dtype, dist64.sample(5).dtype)


if __name__ == "__main__":
  test.main()
