# Copyright 2016 Google Inc. All Rights Reserved.
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
import tensorflow as tf


class ExponentialTest(tf.test.TestCase):

  def testExponentialLogPDF(self):
    with tf.Session():
      batch_size = 6
      lam = tf.constant([2.0] * batch_size)
      lam_v = 2.0
      x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)
      exponential = tf.contrib.distributions.Exponential(lam=lam)
      expected_log_pdf = stats.expon.logpdf(x, scale=1 / lam_v)

      log_pdf = exponential.log_pdf(x)
      self.assertEqual(log_pdf.get_shape(), (6,))
      self.assertAllClose(log_pdf.eval(), expected_log_pdf)

      pdf = exponential.pdf(x)
      self.assertEqual(pdf.get_shape(), (6,))
      self.assertAllClose(pdf.eval(), np.exp(expected_log_pdf))

  def testExponentialCDF(self):
    with tf.Session():
      batch_size = 6
      lam = tf.constant([2.0] * batch_size)
      lam_v = 2.0
      x = np.array([2.5, 2.5, 4.0, 0.1, 1.0, 2.0], dtype=np.float32)

      exponential = tf.contrib.distributions.Exponential(lam=lam)
      expected_cdf = stats.expon.cdf(x, scale=1 / lam_v)

      cdf = exponential.cdf(x)
      self.assertEqual(cdf.get_shape(), (6,))
      self.assertAllClose(cdf.eval(), expected_cdf)

  def testExponentialMean(self):
    with tf.Session():
      lam_v = np.array([1.0, 4.0, 2.5])
      expected_mean = stats.expon.mean(scale=1 / lam_v)
      exponential = tf.contrib.distributions.Exponential(lam=lam_v)
      self.assertEqual(exponential.mean.get_shape(), (3,))
      self.assertAllClose(exponential.mean.eval(), expected_mean)

  def testExponentialVariance(self):
    with tf.Session():
      lam_v = np.array([1.0, 4.0, 2.5])
      expected_variance = stats.expon.var(scale=1 / lam_v)
      exponential = tf.contrib.distributions.Exponential(lam=lam_v)
      self.assertEqual(exponential.variance.get_shape(), (3,))
      self.assertAllClose(exponential.variance.eval(), expected_variance)

  def testExponentialEntropy(self):
    with tf.Session():
      lam_v = np.array([1.0, 4.0, 2.5])
      expected_entropy = stats.expon.entropy(scale=1 / lam_v)
      exponential = tf.contrib.distributions.Exponential(lam=lam_v)
      self.assertEqual(exponential.entropy().get_shape(), (3,))
      self.assertAllClose(exponential.entropy().eval(), expected_entropy)


if __name__ == '__main__':
  tf.test.main()
