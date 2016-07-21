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
"""Tests for MultivariateNormal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats
import tensorflow as tf

distributions = tf.contrib.distributions


class MultivariateNormalShapeTest(tf.test.TestCase):

  def _testPDFShapes(self, mvn_dist, mu, sigma):
    with self.test_session() as sess:
      mvn = mvn_dist(mu, sigma)
      x = 2 * tf.ones_like(mu)

      log_pdf = mvn.log_pdf(x)
      pdf = mvn.pdf(x)

      mu_value = np.ones([3, 3, 2])
      sigma_value = np.zeros([3, 3, 2, 2])
      sigma_value[:] = np.identity(2)
      x_value = 2. * np.ones([3, 3, 2])
      feed_dict = {mu: mu_value, sigma: sigma_value}

      scipy_mvn = stats.multivariate_normal(mean=mu_value[(0, 0)],
                                            cov=sigma_value[(0, 0)])
      expected_log_pdf = scipy_mvn.logpdf(x_value[(0, 0)])
      expected_pdf = scipy_mvn.pdf(x_value[(0, 0)])

      log_pdf_evaled, pdf_evaled = sess.run([log_pdf, pdf], feed_dict=feed_dict)
      self.assertAllEqual([3, 3], log_pdf_evaled.shape)
      self.assertAllEqual([3, 3], pdf_evaled.shape)
      self.assertAllClose(expected_log_pdf, log_pdf_evaled[0, 0])
      self.assertAllClose(expected_pdf, pdf_evaled[0, 0])

  def testPDFUnknownSize(self):
    mu = tf.placeholder(tf.float32, shape=(3 * [None]))
    sigma = tf.placeholder(tf.float32, shape=(4 * [None]))
    self._testPDFShapes(distributions.MultivariateNormalFull, mu, sigma)
    self._testPDFShapes(distributions.MultivariateNormalCholesky, mu, sigma)

  def testPDFUnknownShape(self):
    mu = tf.placeholder(tf.float32)
    sigma = tf.placeholder(tf.float32)
    self._testPDFShapes(distributions.MultivariateNormalFull, mu, sigma)
    self._testPDFShapes(distributions.MultivariateNormalCholesky, mu, sigma)


class MultivariateNormalCholeskyTest(tf.test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def _random_chol(self, *shape):
    mat = self._rng.rand(*shape)
    chol = distributions.batch_matrix_diag_transform(
        mat, transform=tf.nn.softplus)
    chol = tf.batch_matrix_band_part(chol, -1, 0)
    sigma = tf.batch_matmul(chol, chol, adj_y=True)
    return chol.eval(), sigma.eval()

  def testNonmatchingMuSigmaFails(self):
    with self.test_session():
      mu = self._rng.rand(2)
      chol, _ = self._random_chol(2, 2, 2)
      mvn = distributions.MultivariateNormalCholesky(mu, chol)
      with self.assertRaisesOpError("mu should have rank 1 less than cov"):
        mvn.mean().eval()

      mu = self._rng.rand(2, 1)
      chol, _ = self._random_chol(2, 2, 2)
      mvn = distributions.MultivariateNormalCholesky(mu, chol)
      with self.assertRaisesOpError("mu.shape and cov.shape.*should match"):
        mvn.mean().eval()

  def testLogPDFScalarBatch(self):
    with self.test_session():
      mu = self._rng.rand(2)
      chol, sigma = self._random_chol(2, 2)
      mvn = distributions.MultivariateNormalCholesky(mu, chol)
      x = self._rng.rand(2)

      log_pdf = mvn.log_pdf(x)
      pdf = mvn.pdf(x)

      scipy_mvn = stats.multivariate_normal(mean=mu, cov=sigma)

      expected_log_pdf = scipy_mvn.logpdf(x)
      expected_pdf = scipy_mvn.pdf(x)
      self.assertEqual((), log_pdf.get_shape())
      self.assertEqual((), pdf.get_shape())
      self.assertAllClose(expected_log_pdf, log_pdf.eval())
      self.assertAllClose(expected_pdf, pdf.eval())

  def testLogPDFXIsHigherRank(self):
    with self.test_session():
      mu = self._rng.rand(2)
      chol, sigma = self._random_chol(2, 2)
      mvn = distributions.MultivariateNormalCholesky(mu, chol)
      x = self._rng.rand(3, 2)

      log_pdf = mvn.log_pdf(x)
      pdf = mvn.pdf(x)

      scipy_mvn = stats.multivariate_normal(mean=mu, cov=sigma)

      expected_log_pdf = scipy_mvn.logpdf(x)
      expected_pdf = scipy_mvn.pdf(x)
      self.assertEqual((3,), log_pdf.get_shape())
      self.assertEqual((3,), pdf.get_shape())
      self.assertAllClose(expected_log_pdf, log_pdf.eval())
      self.assertAllClose(expected_pdf, pdf.eval())

  def testLogPDFXLowerDimension(self):
    with self.test_session():
      mu = self._rng.rand(3, 2)
      chol, sigma = self._random_chol(3, 2, 2)
      mvn = distributions.MultivariateNormalCholesky(mu, chol)
      x = self._rng.rand(2)

      log_pdf = mvn.log_pdf(x)
      pdf = mvn.pdf(x)

      self.assertEqual((3,), log_pdf.get_shape())
      self.assertEqual((3,), pdf.get_shape())

      # scipy can't do batches, so just test one of them.
      scipy_mvn = stats.multivariate_normal(mean=mu[1, :], cov=sigma[1, :, :])
      expected_log_pdf = scipy_mvn.logpdf(x)
      expected_pdf = scipy_mvn.pdf(x)

      self.assertAllClose(expected_log_pdf, log_pdf.eval()[1])
      self.assertAllClose(expected_pdf, pdf.eval()[1])

  def testEntropy(self):
    with self.test_session():
      mu = self._rng.rand(2)
      chol, sigma = self._random_chol(2, 2)
      mvn = distributions.MultivariateNormalCholesky(mu, chol)
      entropy = mvn.entropy()

      scipy_mvn = stats.multivariate_normal(mean=mu, cov=sigma)
      expected_entropy = scipy_mvn.entropy()
      self.assertEqual(entropy.get_shape(), ())
      self.assertAllClose(expected_entropy, entropy.eval())

  def testEntropyMultidimensional(self):
    with self.test_session():
      mu = self._rng.rand(3, 5, 2)
      chol, sigma = self._random_chol(3, 5, 2, 2)
      mvn = distributions.MultivariateNormalCholesky(mu, chol)
      entropy = mvn.entropy()

      # Scipy doesn't do batches, so test one of them.
      expected_entropy = stats.multivariate_normal(
          mean=mu[1, 1, :], cov=sigma[1, 1, :, :]).entropy()
      self.assertEqual(entropy.get_shape(), (3, 5))
      self.assertAllClose(expected_entropy, entropy.eval()[1, 1])

  def testSample(self):
    with self.test_session():
      mu = self._rng.rand(2)
      chol, sigma = self._random_chol(2, 2)

      n = tf.constant(100000)
      mvn = distributions.MultivariateNormalCholesky(mu, chol)
      samples = mvn.sample(n, seed=137)
      sample_values = samples.eval()
      self.assertEqual(samples.get_shape(), (100000, 2))
      self.assertAllClose(sample_values.mean(axis=0), mu, atol=1e-2)
      self.assertAllClose(np.cov(sample_values, rowvar=0), sigma, atol=1e-1)

  def testSampleMultiDimensional(self):
    with self.test_session():
      mu = self._rng.rand(3, 5, 2)
      chol, sigma = self._random_chol(3, 5, 2, 2)

      n = tf.constant(100000)
      mvn = distributions.MultivariateNormalCholesky(mu, chol)
      samples = mvn.sample(n, seed=137)
      sample_values = samples.eval()

      self.assertEqual(samples.get_shape(), (100000, 3, 5, 2))
      self.assertAllClose(
          sample_values[:, 1, 1, :].mean(axis=0),
          mu[1, 1, :], atol=0.05)
      self.assertAllClose(
          np.cov(sample_values[:, 1, 1, :], rowvar=0),
          sigma[1, 1, :, :], atol=1e-1)


if __name__ == "__main__":
  tf.test.main()
