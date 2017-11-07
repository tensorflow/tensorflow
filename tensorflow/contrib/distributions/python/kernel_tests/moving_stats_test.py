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
"""Tests for computing moving-average statistics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import moving_stats
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

rng = np.random.RandomState(0)


class MovingReduceMeanVarianceTest(test.TestCase):

  def test_assign_moving_mean_variance(self):
    shape = [1, 2]
    true_mean = np.array([[0., 3.]])
    true_stddev = np.array([[1.1, 0.5]])
    with self.test_session() as sess:
      # Start "x" out with this mean.
      mean_var = variables.Variable(array_ops.zeros_like(true_mean))
      variance_var = variables.Variable(array_ops.ones_like(true_stddev))
      x = random_ops.random_normal(shape, dtype=np.float64, seed=0)
      x = true_stddev * x + true_mean
      ema, emv = moving_stats.assign_moving_mean_variance(
          mean_var, variance_var, x, decay=0.99)

      self.assertEqual(ema.dtype.base_dtype, dtypes.float64)
      self.assertEqual(emv.dtype.base_dtype, dtypes.float64)

      # Run 1000 updates; moving averages should be near the true values.
      variables.global_variables_initializer().run()
      for _ in range(2000):
        sess.run([ema, emv])

      [mean_var_, variance_var_, ema_, emv_] = sess.run([
          mean_var, variance_var, ema, emv])
      # Test that variables are passed-through.
      self.assertAllEqual(mean_var_, ema_)
      self.assertAllEqual(variance_var_, emv_)
      # Test that values are as expected.
      self.assertAllClose(true_mean, ema_, rtol=0.005, atol=0.015)
      self.assertAllClose(true_stddev**2., emv_, rtol=0.06, atol=0.)

      # Change the mean, var then update some more. Moving averages should
      # re-converge.
      sess.run([
          mean_var.assign(np.array([[-1., 2.]])),
          variance_var.assign(np.array([[2., 1.]])),
      ])
      for _ in range(2000):
        sess.run([ema, emv])

      [mean_var_, variance_var_, ema_, emv_] = sess.run([
          mean_var, variance_var, ema, emv])
      # Test that variables are passed-through.
      self.assertAllEqual(mean_var_, ema_)
      self.assertAllEqual(variance_var_, emv_)
      # Test that values are as expected.
      self.assertAllClose(true_mean, ema_, rtol=0.005, atol=0.015)
      self.assertAllClose(true_stddev**2., emv_, rtol=0.1, atol=0.)

  def test_moving_mean_variance(self):
    shape = [1, 2]
    true_mean = np.array([[0., 3.]])
    true_stddev = np.array([[1.1, 0.5]])
    with self.test_session() as sess:
      # Start "x" out with this mean.
      x = random_ops.random_normal(shape, dtype=np.float64, seed=0)
      x = true_stddev * x + true_mean
      ema, emv = moving_stats.moving_mean_variance(
          x, decay=0.99)

      self.assertEqual(ema.dtype.base_dtype, dtypes.float64)
      self.assertEqual(emv.dtype.base_dtype, dtypes.float64)

      # Run 1000 updates; moving averages should be near the true values.
      variables.global_variables_initializer().run()
      for _ in range(2000):
        sess.run([ema, emv])

      [ema_, emv_] = sess.run([ema, emv])
      self.assertAllClose(true_mean, ema_, rtol=0.005, atol=0.015)
      self.assertAllClose(true_stddev**2., emv_, rtol=0.06, atol=0.)


class MovingLogExponentialMovingMeanExpTest(test.TestCase):

  def test_assign_log_moving_mean_exp(self):
    shape = [1, 2]
    true_mean = np.array([[0., 3.]])
    true_stddev = np.array([[1.1, 0.5]])
    decay = 0.99
    with self.test_session() as sess:
      # Start "x" out with this mean.
      x = random_ops.random_normal(shape, dtype=np.float64, seed=0)
      x = true_stddev * x + true_mean
      log_mean_exp_var = variables.Variable(array_ops.zeros_like(true_mean))
      variables.global_variables_initializer().run()
      log_mean_exp = moving_stats.assign_log_moving_mean_exp(
          log_mean_exp_var, x, decay=decay)
      expected_ = np.zeros_like(true_mean)
      for _ in range(2000):
        x_, log_mean_exp_ = sess.run([x, log_mean_exp])
        expected_ = np.log(decay * np.exp(expected_) + (1 - decay) * np.exp(x_))
        self.assertAllClose(expected_, log_mean_exp_, rtol=1e-6, atol=1e-9)

if __name__ == "__main__":
  test.main()
