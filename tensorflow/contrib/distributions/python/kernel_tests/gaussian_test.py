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

  def testGaussianLogPDF(self):
    with tf.Session():
      batch_size = 6
      mu = tf.constant([3.0] * batch_size)
      sigma = tf.constant([math.sqrt(10.0)] * batch_size)
      mu_v = 3.0
      sigma_v = np.sqrt(10.0)
      x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=np.float32)
      gaussian = tf.contrib.distributions.Gaussian(mu=mu, sigma=sigma)
      expected_log_pdf = np.log(
          1 / np.sqrt(2 * np.pi) / sigma_v
          * np.exp(-1.0 / (2 * sigma_v**2) * (x - mu_v)**2))

      log_pdf = gaussian.log_pdf(x)
      self.assertAllClose(expected_log_pdf, log_pdf.eval())

      pdf = gaussian.pdf(x)
      self.assertAllClose(np.exp(expected_log_pdf), pdf.eval())

  def testGaussianLogPDFMultidimensional(self):
    with tf.Session():
      batch_size = 6
      mu = tf.constant([[3.0, -3.0]] * batch_size)
      sigma = tf.constant([[math.sqrt(10.0), math.sqrt(15.0)]] * batch_size)
      mu_v = np.array([3.0, -3.0])
      sigma_v = np.array([np.sqrt(10.0), np.sqrt(15.0)])
      x = np.array([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=np.float32).T
      gaussian = tf.contrib.distributions.Gaussian(mu=mu, sigma=sigma)
      expected_log_pdf = np.log(
          1 / np.sqrt(2 * np.pi) / sigma_v
          * np.exp(-1.0 / (2 * sigma_v**2) * (x - mu_v)**2))

      log_pdf = gaussian.log_pdf(x)
      log_pdf_values = log_pdf.eval()
      self.assertEqual(log_pdf.get_shape(), (6, 2))
      self.assertAllClose(expected_log_pdf, log_pdf_values)

      pdf = gaussian.pdf(x)
      pdf_values = pdf.eval()
      self.assertEqual(pdf.get_shape(), (6, 2))
      self.assertAllClose(np.exp(expected_log_pdf), pdf_values)

  def testGaussianCDF(self):
    with tf.Session():
      batch_size = 6
      mu = tf.constant([3.0] * batch_size)
      sigma = tf.constant([math.sqrt(10.0)] * batch_size)
      mu_v = 3.0
      sigma_v = np.sqrt(10.0)
      x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=np.float32)

      gaussian = tf.contrib.distributions.Gaussian(mu=mu, sigma=sigma)
      erf_fn = np.vectorize(math.erf)

      # From Wikipedia
      expected_cdf = 0.5 * (1.0 + erf_fn((x - mu_v)/(sigma_v*np.sqrt(2))))

      cdf = gaussian.cdf(x)
      self.assertAllClose(expected_cdf, cdf.eval())

  def testGaussianEntropy(self):
    with tf.Session():
      mu_v = np.array([1.0, 1.0, 1.0])
      sigma_v = np.array([[1.0, 2.0, 3.0]]).T
      gaussian = tf.contrib.distributions.Gaussian(mu=mu_v, sigma=sigma_v)

      sigma_broadcast = mu_v * sigma_v
      expected_entropy = 0.5 * np.log(2*np.pi*np.exp(1)*sigma_broadcast**2)
      self.assertAllClose(expected_entropy, gaussian.entropy().eval())

  def testGaussianSample(self):
    with tf.Session():
      mu = tf.constant(3.0)
      sigma = tf.constant(math.sqrt(10.0))
      mu_v = 3.0
      sigma_v = np.sqrt(10.0)
      n = tf.constant(100000)
      gaussian = tf.contrib.distributions.Gaussian(mu=mu, sigma=sigma)
      samples = gaussian.sample(n, seed=137)
      sample_values = samples.eval()
      self.assertEqual(sample_values.shape, (100000,))
      self.assertAllClose(sample_values.mean(), mu_v, atol=1e-2)
      self.assertAllClose(sample_values.std(), sigma_v, atol=1e-1)

  def testGaussianSampleMultiDimensional(self):
    with tf.Session():
      batch_size = 2
      mu = tf.constant([[3.0, -3.0]] * batch_size)
      sigma = tf.constant([[math.sqrt(10.0), math.sqrt(15.0)]] * batch_size)
      mu_v = [3.0, -3.0]
      sigma_v = [np.sqrt(10.0), np.sqrt(15.0)]
      n = tf.constant(100000)
      gaussian = tf.contrib.distributions.Gaussian(mu=mu, sigma=sigma)
      samples = gaussian.sample(n, seed=137)
      sample_values = samples.eval()
      self.assertEqual(samples.get_shape(), (100000, batch_size, 2))
      self.assertAllClose(sample_values[:, 0, 0].mean(), mu_v[0], atol=1e-2)
      self.assertAllClose(sample_values[:, 0, 0].std(), sigma_v[0], atol=1e-1)
      self.assertAllClose(sample_values[:, 0, 1].mean(), mu_v[1], atol=1e-2)
      self.assertAllClose(sample_values[:, 0, 1].std(), sigma_v[1], atol=1e-1)

  def testNegativeSigmaFails(self):
    with tf.Session():
      gaussian = tf.contrib.distributions.Gaussian(
          mu=[1.],
          sigma=[-5.],
          name='G')
      with self.assertRaisesOpError(
          r'should contain only positive values'):
        gaussian.mean.eval()

if __name__ == '__main__':
  tf.test.main()
