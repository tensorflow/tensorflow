# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
from scipy import stats
import tensorflow as tf


class NormalTest(tf.test.TestCase):

  def testNormalLogPDF(self):
    with self.test_session():
      batch_size = 6
      mu = tf.constant([3.0] * batch_size)
      sigma = tf.constant([math.sqrt(10.0)] * batch_size)
      x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=np.float32)
      normal = tf.contrib.distributions.Normal(mu=mu, sigma=sigma)
      expected_log_pdf = stats.norm(mu.eval(), sigma.eval()).logpdf(x)

      log_pdf = normal.log_pdf(x)
      self.assertAllClose(expected_log_pdf, log_pdf.eval())
      self.assertAllEqual(normal.batch_shape().eval(), log_pdf.get_shape())
      self.assertAllEqual(normal.batch_shape().eval(), log_pdf.eval().shape)
      self.assertAllEqual(normal.get_batch_shape(), log_pdf.get_shape())
      self.assertAllEqual(normal.get_batch_shape(), log_pdf.eval().shape)

      pdf = normal.pdf(x)
      self.assertAllClose(np.exp(expected_log_pdf), pdf.eval())
      self.assertAllEqual(normal.batch_shape().eval(), pdf.get_shape())
      self.assertAllEqual(normal.batch_shape().eval(), pdf.eval().shape)
      self.assertAllEqual(normal.get_batch_shape(), pdf.get_shape())
      self.assertAllEqual(normal.get_batch_shape(), pdf.eval().shape)

  def testNormalLogPDFMultidimensional(self):
    with self.test_session():
      batch_size = 6
      mu = tf.constant([[3.0, -3.0]] * batch_size)
      sigma = tf.constant([[math.sqrt(10.0), math.sqrt(15.0)]] * batch_size)
      x = np.array([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=np.float32).T
      normal = tf.contrib.distributions.Normal(mu=mu, sigma=sigma)
      expected_log_pdf = stats.norm(mu.eval(), sigma.eval()).logpdf(x)

      log_pdf = normal.log_pdf(x)
      log_pdf_values = log_pdf.eval()
      self.assertEqual(log_pdf.get_shape(), (6, 2))
      self.assertAllClose(expected_log_pdf, log_pdf_values)
      self.assertAllEqual(normal.batch_shape().eval(), log_pdf.get_shape())
      self.assertAllEqual(normal.batch_shape().eval(), log_pdf.eval().shape)
      self.assertAllEqual(normal.get_batch_shape(), log_pdf.get_shape())
      self.assertAllEqual(normal.get_batch_shape(), log_pdf.eval().shape)

      pdf = normal.pdf(x)
      pdf_values = pdf.eval()
      self.assertEqual(pdf.get_shape(), (6, 2))
      self.assertAllClose(np.exp(expected_log_pdf), pdf_values)
      self.assertAllEqual(normal.batch_shape().eval(), pdf.get_shape())
      self.assertAllEqual(normal.batch_shape().eval(), pdf_values.shape)
      self.assertAllEqual(normal.get_batch_shape(), pdf.get_shape())
      self.assertAllEqual(normal.get_batch_shape(), pdf_values.shape)

  def testNormalCDF(self):
    with self.test_session():
      batch_size = 6
      mu = tf.constant([3.0] * batch_size)
      sigma = tf.constant([math.sqrt(10.0)] * batch_size)
      x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=np.float32)

      normal = tf.contrib.distributions.Normal(mu=mu, sigma=sigma)
      expected_cdf = stats.norm(mu.eval(), sigma.eval()).cdf(x)

      cdf = normal.cdf(x)
      self.assertAllClose(expected_cdf, cdf.eval())
      self.assertAllEqual(normal.batch_shape().eval(), cdf.get_shape())
      self.assertAllEqual(normal.batch_shape().eval(), cdf.eval().shape)
      self.assertAllEqual(normal.get_batch_shape(), cdf.get_shape())
      self.assertAllEqual(normal.get_batch_shape(), cdf.eval().shape)

  def testNormalEntropyWithScalarInputs(self):
    # Scipy.stats.norm cannot deal with the shapes in the other test.
    with self.test_session():
      mu_v = 2.34
      sigma_v = 4.56
      normal = tf.contrib.distributions.Normal(mu=mu_v, sigma=sigma_v)

      # scipy.stats.norm cannot deal with these shapes.
      expected_entropy = stats.norm(mu_v, sigma_v).entropy()
      entropy = normal.entropy()
      self.assertAllClose(expected_entropy, entropy.eval())
      self.assertAllEqual(normal.batch_shape().eval(), entropy.get_shape())
      self.assertAllEqual(normal.batch_shape().eval(), entropy.eval().shape)
      self.assertAllEqual(normal.get_batch_shape(), entropy.get_shape())
      self.assertAllEqual(normal.get_batch_shape(), entropy.eval().shape)

  def testNormalEntropy(self):
    with self.test_session():
      mu_v = np.array([1.0, 1.0, 1.0])
      sigma_v = np.array([[1.0, 2.0, 3.0]]).T
      normal = tf.contrib.distributions.Normal(mu=mu_v, sigma=sigma_v)

      # scipy.stats.norm cannot deal with these shapes.
      sigma_broadcast = mu_v * sigma_v
      expected_entropy = 0.5 * np.log(2*np.pi*np.exp(1)*sigma_broadcast**2)
      entropy = normal.entropy()
      np.testing.assert_allclose(expected_entropy, entropy.eval())
      self.assertAllEqual(normal.batch_shape().eval(), entropy.get_shape())
      self.assertAllEqual(normal.batch_shape().eval(), entropy.eval().shape)
      self.assertAllEqual(normal.get_batch_shape(), entropy.get_shape())
      self.assertAllEqual(normal.get_batch_shape(), entropy.eval().shape)

  def testNormalMeanAndMode(self):
    with self.test_session():
      # Mu will be broadcast to [7, 7, 7].
      mu = [7.]
      sigma = [11., 12., 13.]

      normal = tf.contrib.distributions.Normal(mu=mu, sigma=sigma)

      self.assertAllEqual((3,), normal.mean().get_shape())
      self.assertAllEqual([7., 7, 7], normal.mean().eval())

      self.assertAllEqual((3,), normal.mode().get_shape())
      self.assertAllEqual([7., 7, 7], normal.mode().eval())

  def testNormalVariance(self):
    with self.test_session():
      # sigma will be broadcast to [7, 7, 7]
      mu = [1., 2., 3.]
      sigma = [7.]

      normal = tf.contrib.distributions.Normal(mu=mu, sigma=sigma)

      self.assertAllEqual((3,), normal.variance().get_shape())
      self.assertAllEqual([49., 49, 49], normal.variance().eval())

  def testNormalStandardDeviation(self):
    with self.test_session():
      # sigma will be broadcast to [7, 7, 7]
      mu = [1., 2., 3.]
      sigma = [7.]

      normal = tf.contrib.distributions.Normal(mu=mu, sigma=sigma)

      self.assertAllEqual((3,), normal.std().get_shape())
      self.assertAllEqual([7., 7, 7], normal.std().eval())

  def testNormalSample(self):
    with self.test_session():
      mu = tf.constant(3.0)
      sigma = tf.constant(math.sqrt(3.0))
      mu_v = 3.0
      sigma_v = np.sqrt(3.0)
      n = tf.constant(100000)
      normal = tf.contrib.distributions.Normal(mu=mu, sigma=sigma)
      samples = normal.sample_n(n)
      sample_values = samples.eval()
      # Note that the standard error for the sample mean is ~ sigma / sqrt(n).
      # The sample variance similarly is dependent on sigma and n.
      # Thus, the tolerances below are very sensitive to number of samples
      # as well as the variances chosen.
      self.assertEqual(sample_values.shape, (100000,))
      self.assertAllClose(sample_values.mean(), mu_v, atol=1e-1)
      self.assertAllClose(sample_values.std(), sigma_v, atol=1e-1)

      expected_samples_shape = (
          tf.TensorShape([n.eval()]).concatenate(
              tf.TensorShape(normal.batch_shape().eval())))

      self.assertAllEqual(expected_samples_shape, samples.get_shape())
      self.assertAllEqual(expected_samples_shape, sample_values.shape)

      expected_samples_shape = (
          tf.TensorShape([n.eval()]).concatenate(
              normal.get_batch_shape()))

      self.assertAllEqual(expected_samples_shape, samples.get_shape())
      self.assertAllEqual(expected_samples_shape, sample_values.shape)

  def testNormalSampleMultiDimensional(self):
    with self.test_session():
      batch_size = 2
      mu = tf.constant([[3.0, -3.0]] * batch_size)
      sigma = tf.constant([[math.sqrt(2.0), math.sqrt(3.0)]] * batch_size)
      mu_v = [3.0, -3.0]
      sigma_v = [np.sqrt(2.0), np.sqrt(3.0)]
      n = tf.constant(100000)
      normal = tf.contrib.distributions.Normal(mu=mu, sigma=sigma)
      samples = normal.sample_n(n)
      sample_values = samples.eval()
      # Note that the standard error for the sample mean is ~ sigma / sqrt(n).
      # The sample variance similarly is dependent on sigma and n.
      # Thus, the tolerances below are very sensitive to number of samples
      # as well as the variances chosen.
      self.assertEqual(samples.get_shape(), (100000, batch_size, 2))
      self.assertAllClose(sample_values[:, 0, 0].mean(), mu_v[0], atol=1e-1)
      self.assertAllClose(sample_values[:, 0, 0].std(), sigma_v[0], atol=1e-1)
      self.assertAllClose(sample_values[:, 0, 1].mean(), mu_v[1], atol=1e-1)
      self.assertAllClose(sample_values[:, 0, 1].std(), sigma_v[1], atol=1e-1)

      expected_samples_shape = (
          tf.TensorShape([n.eval()]).concatenate(
              tf.TensorShape(normal.batch_shape().eval())))
      self.assertAllEqual(expected_samples_shape, samples.get_shape())
      self.assertAllEqual(expected_samples_shape, sample_values.shape)

      expected_samples_shape = (
          tf.TensorShape([n.eval()]).concatenate(normal.get_batch_shape()))
      self.assertAllEqual(expected_samples_shape, samples.get_shape())
      self.assertAllEqual(expected_samples_shape, sample_values.shape)

  def testNegativeSigmaFails(self):
    with self.test_session():
      normal = tf.contrib.distributions.Normal(
          mu=[1.],
          sigma=[-5.],
          name='G')
      with self.assertRaisesOpError('Condition x > 0 did not hold'):
        normal.mean().eval()

  def testNormalShape(self):
    with self.test_session():
      mu = tf.constant([-3.0] * 5)
      sigma = tf.constant(11.0)
      normal = tf.contrib.distributions.Normal(mu=mu, sigma=sigma)

      self.assertEqual(normal.batch_shape().eval(), [5])
      self.assertEqual(normal.get_batch_shape(), tf.TensorShape([5]))
      self.assertAllEqual(normal.event_shape().eval(), [])
      self.assertEqual(normal.get_event_shape(), tf.TensorShape([]))

  def testNormalShapeWithPlaceholders(self):
    mu = tf.placeholder(dtype=tf.float32)
    sigma = tf.placeholder(dtype=tf.float32)
    normal = tf.contrib.distributions.Normal(mu=mu, sigma=sigma)

    with self.test_session() as sess:
      # get_batch_shape should return an "<unknown>" tensor.
      self.assertEqual(normal.get_batch_shape(), tf.TensorShape(None))
      self.assertEqual(normal.get_event_shape(), ())
      self.assertAllEqual(normal.event_shape().eval(), [])
      self.assertAllEqual(
          sess.run(normal.batch_shape(),
                   feed_dict={mu: 5.0, sigma: [1.0, 2.0]}),
          [2])

  def testNormalNormalKL(self):
    with self.test_session() as sess:
      batch_size = 6
      mu_a = np.array([3.0] * batch_size)
      sigma_a = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
      mu_b = np.array([-3.0] * batch_size)
      sigma_b = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

      n_a = tf.contrib.distributions.Normal(mu=mu_a, sigma=sigma_a)
      n_b = tf.contrib.distributions.Normal(mu=mu_b, sigma=sigma_b)

      kl = tf.contrib.distributions.kl(n_a, n_b)
      kl_val = sess.run(kl)

      kl_expected = (
          (mu_a - mu_b)**2 / (2 * sigma_b**2)
          + 0.5 * ((sigma_a**2/sigma_b**2) -
                   1 - 2 * np.log(sigma_a / sigma_b)))

      self.assertEqual(kl.get_shape(), (batch_size,))
      self.assertAllClose(kl_val, kl_expected)


if __name__ == '__main__':
  tf.test.main()
