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
"""Tests for MultivariateNormal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class MultivariateNormalTest(tf.test.TestCase):

  def testNonmatchingMuSigmaFails(self):
    with tf.Session():
      mvn = tf.contrib.distributions.MultivariateNormal(
          mu=[1.0, 2.0],
          sigma=[[[1.0, 0.0],
                  [0.0, 1.0]],
                 [[1.0, 0.0],
                  [0.0, 1.0]]])
      with self.assertRaisesOpError(
          r"Rank of mu should be one less than rank of sigma"):
        mvn.mean.eval()

      mvn = tf.contrib.distributions.MultivariateNormal(
          mu=[[1.0], [2.0]],
          sigma=[[[1.0, 0.0],
                  [0.0, 1.0]],
                 [[1.0, 0.0],
                  [0.0, 1.0]]])
      with self.assertRaisesOpError(
          r"mu.shape and sigma.shape\[\:-1\] must match"):
        mvn.mean.eval()

  def testNotPositiveDefiniteSigmaFails(self):
    with tf.Session():
      mvn = tf.contrib.distributions.MultivariateNormal(
          mu=[[1.0, 2.0], [1.0, 2.0]],
          sigma=[[[1.0, 0.0],
                  [0.0, 1.0]],
                 [[1.0, 1.0],
                  [1.0, 1.0]]])
      with self.assertRaisesOpError(
          r"LLT decomposition was not successful."):
        mvn.mean.eval()
      mvn = tf.contrib.distributions.MultivariateNormal(
          mu=[[1.0, 2.0], [1.0, 2.0]],
          sigma=[[[1.0, 0.0],
                  [0.0, 1.0]],
                 [[-1.0, 0.0],
                  [0.0, 1.0]]])
      with self.assertRaisesOpError(
          r"LLT decomposition was not successful."):
        mvn.mean.eval()
      mvn = tf.contrib.distributions.MultivariateNormal(
          mu=[[1.0, 2.0], [1.0, 2.0]],
          sigma_chol=[[[1.0, 0.0],
                       [0.0, 1.0]],
                      [[-1.0, 0.0],
                       [0.0, 1.0]]])
      with self.assertRaisesOpError(
          r"sigma_chol is not positive definite."):
        mvn.mean.eval()

  def testLogPDFScalar(self):
    with tf.Session():
      mu_v = np.array([-3.0, 3.0], dtype=np.float32)
      mu = tf.constant(mu_v)
      sigma_v = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
      sigma = tf.constant(sigma_v)
      x = np.array([-2.5, 2.5], dtype=np.float32)
      mvn = tf.contrib.distributions.MultivariateNormal(mu=mu, sigma=sigma)

      log_pdf = mvn.log_pdf(x)
      pdf = mvn.pdf(x)

      try:
        from scipy import stats  # pylint: disable=g-import-not-at-top
        scipy_mvn = stats.multivariate_normal(mean=mu_v, cov=sigma_v)
        expected_log_pdf = scipy_mvn.logpdf(x)
        expected_pdf = scipy_mvn.pdf(x)
        self.assertAllClose(expected_log_pdf, log_pdf.eval())
        self.assertAllClose(expected_pdf, pdf.eval())
      except ImportError as e:
        tf.logging.warn("Cannot test stats functions: %s" % str(e))

  def testLogPDFScalarSigmaHalf(self):
    with tf.Session():
      mu_v = np.array([-3.0, 3.0, 1.0], dtype=np.float32)
      mu = tf.constant(mu_v)
      sigma_v = np.array([[1.0, 0.1, 0.2],
                          [0.1, 2.0, 0.05],
                          [0.2, 0.05, 3.0]], dtype=np.float32)
      sigma_chol_v = np.linalg.cholesky(sigma_v)
      sigma_chol = tf.constant(sigma_chol_v)
      x = np.array([-2.5, 2.5, 1.0], dtype=np.float32)
      mvn = tf.contrib.distributions.MultivariateNormal(
          mu=mu, sigma_chol=sigma_chol)
      log_pdf = mvn.log_pdf(x)
      pdf = mvn.pdf(x)
      sigma = mvn.sigma

      try:
        from scipy import stats  # pylint: disable=g-import-not-at-top
        scipy_mvn = stats.multivariate_normal(mean=mu_v, cov=sigma_v)
        expected_log_pdf = scipy_mvn.logpdf(x)
        expected_pdf = scipy_mvn.pdf(x)
        self.assertEqual(sigma.get_shape(), (3, 3))
        self.assertAllClose(sigma_v, sigma.eval())
        self.assertAllClose(expected_log_pdf, log_pdf.eval())
        self.assertAllClose(expected_pdf, pdf.eval())
      except ImportError as e:
        tf.logging.warn("Cannot test stats functions: %s" % str(e))

  def testLogPDF(self):
    with tf.Session():
      mu_v = np.array([-3.0, 3.0], dtype=np.float32)
      mu = tf.constant(mu_v)
      sigma_v = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
      sigma = tf.constant(sigma_v)
      x = np.array([[-2.5, 2.5], [4.0, 0.0], [-1.0, 2.0]], dtype=np.float32)
      mvn = tf.contrib.distributions.MultivariateNormal(mu=mu, sigma=sigma)
      log_pdf = mvn.log_pdf(x)
      pdf = mvn.pdf(x)

      try:
        from scipy import stats  # pylint: disable=g-import-not-at-top
        scipy_mvn = stats.multivariate_normal(mean=mu_v, cov=sigma_v)
        expected_log_pdf = scipy_mvn.logpdf(x)
        expected_pdf = scipy_mvn.pdf(x)
        self.assertEqual(log_pdf.get_shape(), (3,))
        self.assertAllClose(expected_log_pdf, log_pdf.eval())
        self.assertAllClose(expected_pdf, pdf.eval())
      except ImportError as e:
        tf.logging.warn("Cannot test stats functions: %s" % str(e))

  def testLogPDFMatchingDimension(self):
    with tf.Session():
      mu_v = np.array([-3.0, 3.0], dtype=np.float32)
      mu = tf.constant(np.vstack(3 * [mu_v]))
      sigma_v = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
      sigma = tf.constant(np.vstack(3 * [sigma_v[np.newaxis, :]]))
      x = np.array([[-2.5, 2.5], [4.0, 0.0], [-1.0, 2.0]], dtype=np.float32)
      mvn = tf.contrib.distributions.MultivariateNormal(mu=mu, sigma=sigma)
      log_pdf = mvn.log_pdf(x)
      pdf = mvn.pdf(x)

      try:
        from scipy import stats  # pylint: disable=g-import-not-at-top
        scipy_mvn = stats.multivariate_normal(mean=mu_v, cov=sigma_v)
        expected_log_pdf = scipy_mvn.logpdf(x)
        expected_pdf = scipy_mvn.pdf(x)
        self.assertEqual(log_pdf.get_shape(), (3,))
        self.assertAllClose(expected_log_pdf, log_pdf.eval())
        self.assertAllClose(expected_pdf, pdf.eval())
      except ImportError as e:
        tf.logging.warn("Cannot test stats functions: %s" % str(e))

  def testLogPDFMultidimensional(self):
    with tf.Session():
      mu_v = np.array([-3.0, 3.0], dtype=np.float32)
      mu = tf.constant(np.vstack(15 * [mu_v]).reshape(3, 5, 2))
      sigma_v = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
      sigma = tf.constant(
          np.vstack(15 * [sigma_v[np.newaxis, :]]).reshape(3, 5, 2, 2))
      x = np.array([-2.5, 2.5], dtype=np.float32)
      mvn = tf.contrib.distributions.MultivariateNormal(mu=mu, sigma=sigma)
      log_pdf = mvn.log_pdf(x)
      pdf = mvn.pdf(x)

      try:
        from scipy import stats  # pylint: disable=g-import-not-at-top
        scipy_mvn = stats.multivariate_normal(mean=mu_v, cov=sigma_v)
        expected_log_pdf = np.vstack(15 * [scipy_mvn.logpdf(x)]).reshape(3, 5)
        expected_pdf = np.vstack(15 * [scipy_mvn.pdf(x)]).reshape(3, 5)
        self.assertEqual(log_pdf.get_shape(), (3, 5))
        self.assertAllClose(expected_log_pdf, log_pdf.eval())
        self.assertAllClose(expected_pdf, pdf.eval())
      except ImportError as e:
        tf.logging.warn("Cannot test stats functions: %s" % str(e))

  def testEntropy(self):
    with tf.Session():
      mu_v = np.array([-3.0, 3.0], dtype=np.float32)
      mu = tf.constant(mu_v)
      sigma_v = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
      sigma = tf.constant(sigma_v)
      mvn = tf.contrib.distributions.MultivariateNormal(mu=mu, sigma=sigma)
      entropy = mvn.entropy()

      try:
        from scipy import stats  # pylint: disable=g-import-not-at-top
        scipy_mvn = stats.multivariate_normal(mean=mu_v, cov=sigma_v)
        expected_entropy = scipy_mvn.entropy()
        self.assertEqual(entropy.get_shape(), ())
        self.assertAllClose(expected_entropy, entropy.eval())
      except ImportError as e:
        tf.logging.warn("Cannot test stats functions: %s" % str(e))

  def testEntropyMultidimensional(self):
    with tf.Session():
      mu_v = np.array([-3.0, 3.0], dtype=np.float32)
      mu = tf.constant(np.vstack(15 * [mu_v]).reshape(3, 5, 2))
      sigma_v = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
      sigma = tf.constant(
          np.vstack(15 * [sigma_v[np.newaxis, :]]).reshape(3, 5, 2, 2))
      mvn = tf.contrib.distributions.MultivariateNormal(mu=mu, sigma=sigma)
      entropy = mvn.entropy()

      try:
        from scipy import stats  # pylint: disable=g-import-not-at-top
        scipy_mvn = stats.multivariate_normal(mean=mu_v, cov=sigma_v)
        expected_entropy = np.vstack(15 * [scipy_mvn.entropy()]).reshape(3, 5)
        self.assertEqual(entropy.get_shape(), (3, 5))
        self.assertAllClose(expected_entropy, entropy.eval())
      except ImportError as e:
        tf.logging.warn("Cannot test stats functions: %s" % str(e))

  def testSample(self):
    with tf.Session():
      mu_v = np.array([-3.0, 3.0], dtype=np.float32)
      mu = tf.constant(mu_v)
      sigma_v = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
      sigma = tf.constant(sigma_v)
      n = tf.constant(100000)
      mvn = tf.contrib.distributions.MultivariateNormal(mu=mu, sigma=sigma)
      samples = mvn.sample(n, seed=137)
      sample_values = samples.eval()
      self.assertEqual(samples.get_shape(), (100000, 2))
      self.assertAllClose(sample_values.mean(axis=0), mu_v, atol=1e-2)
      self.assertAllClose(np.cov(sample_values, rowvar=0), sigma_v, atol=1e-1)

  def testSampleMultiDimensional(self):
    with tf.Session():
      mu_v = np.array([-3.0, 3.0], dtype=np.float32)
      mu = tf.constant(np.vstack(15 * [mu_v]).reshape(3, 5, 2))
      sigma_v = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
      sigma = tf.constant(
          np.vstack(15 * [sigma_v[np.newaxis, :]]).reshape(3, 5, 2, 2))
      n = tf.constant(100000)
      mvn = tf.contrib.distributions.MultivariateNormal(mu=mu, sigma=sigma)
      samples = mvn.sample(n, seed=137)
      sample_values = samples.eval()
      self.assertEqual(samples.get_shape(), (100000, 3, 5, 2))
      sample_values = sample_values.reshape(100000, 15, 2)
      for i in range(15):
        self.assertAllClose(
            sample_values[:, i, :].mean(axis=0), mu_v, atol=1e-2)
        self.assertAllClose(
            np.cov(sample_values[:, i, :], rowvar=0), sigma_v, atol=1e-1)


if __name__ == "__main__":
  tf.test.main()
