# Copyright 2015 Google Inc. All Rights Reserved.
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

import math

import numpy as np
import tensorflow as tf


class GaussianTest(tf.test.TestCase):

  def testGaussianLogLikelihoodPDF(self):
    with tf.Session():
      mu = tf.constant(3.0)
      sigma = tf.constant(math.sqrt(1/0.1))
      mu_v = 3.0
      sigma_v = np.sqrt(1/0.1)
      x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0])
      gaussian = tf.contrib.distributions.Gaussian(mu=mu, sigma=sigma)
      expected_log_pdf = np.log(
          1/np.sqrt(2*np.pi)/sigma_v*np.exp(-1.0/(2*sigma_v**2)*(x-mu_v)**2))

      log_pdf = gaussian.log_pdf(x)
      self.assertAllClose(expected_log_pdf, log_pdf.eval())

      pdf = gaussian.pdf(x)
      self.assertAllClose(np.exp(expected_log_pdf), pdf.eval())

  def testGaussianCDF(self):
    with tf.Session():
      mu = tf.constant(3.0)
      sigma = tf.constant(math.sqrt(1/0.1))
      mu_v = 3.0
      sigma_v = np.sqrt(1/0.1)
      x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0])
      gaussian = tf.contrib.distributions.Gaussian(mu=mu, sigma=sigma)
      erf_fn = np.vectorize(math.erf)

      # From Wikipedia
      expected_cdf = 0.5*(1.0 + erf_fn((x - mu_v)/(sigma_v*np.sqrt(2))))

      cdf = gaussian.cdf(x)
      self.assertAllClose(expected_cdf, cdf.eval())

  def testGaussianSample(self):
    with tf.Session():
      mu = tf.constant(3.0)
      sigma = tf.constant(math.sqrt(1/0.1))
      mu_v = 3.0
      sigma_v = np.sqrt(1/0.1)
      n = tf.constant(10000)
      gaussian = tf.contrib.distributions.Gaussian(mu=mu, sigma=sigma)
      samples = gaussian.sample(n, seed=137)
      sample_values = samples.eval()
      self.assertEqual(sample_values.shape, (10000,))
      self.assertAllClose(sample_values.mean(), mu_v, atol=1e-2)
      self.assertAllClose(sample_values.std(), sigma_v, atol=1e-1)

if __name__ == '__main__':
  tf.test.main()
