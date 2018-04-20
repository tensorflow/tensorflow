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
"""Tests for MultivariateNormalFullCovariance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats
from tensorflow.contrib import distributions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


ds = distributions
rng = np.random.RandomState(42)


class MultivariateNormalFullCovarianceTest(test.TestCase):

  def _random_pd_matrix(self, *shape):
    mat = rng.rand(*shape)
    chol = ds.matrix_diag_transform(mat, transform=nn_ops.softplus)
    chol = array_ops.matrix_band_part(chol, -1, 0)
    return math_ops.matmul(chol, chol, adjoint_b=True).eval()

  def testRaisesIfInitializedWithNonSymmetricMatrix(self):
    with self.test_session():
      mu = [1., 2.]
      sigma = [[1., 0.], [1., 1.]]  # Nonsingular, but not symmetric
      mvn = ds.MultivariateNormalFullCovariance(mu, sigma, validate_args=True)
      with self.assertRaisesOpError("not symmetric"):
        mvn.covariance().eval()

  def testNamePropertyIsSetByInitArg(self):
    with self.test_session():
      mu = [1., 2.]
      sigma = [[1., 0.], [0., 1.]]
      mvn = ds.MultivariateNormalFullCovariance(mu, sigma, name="Billy")
      self.assertEqual(mvn.name, "Billy/")

  def testDoesNotRaiseIfInitializedWithSymmetricMatrix(self):
    with self.test_session():
      mu = rng.rand(10)
      sigma = self._random_pd_matrix(10, 10)
      mvn = ds.MultivariateNormalFullCovariance(mu, sigma, validate_args=True)
      # Should not raise
      mvn.covariance().eval()

  def testLogPDFScalarBatch(self):
    with self.test_session():
      mu = rng.rand(2)
      sigma = self._random_pd_matrix(2, 2)
      mvn = ds.MultivariateNormalFullCovariance(mu, sigma, validate_args=True)
      x = rng.rand(2)

      log_pdf = mvn.log_prob(x)
      pdf = mvn.prob(x)

      scipy_mvn = stats.multivariate_normal(mean=mu, cov=sigma)

      expected_log_pdf = scipy_mvn.logpdf(x)
      expected_pdf = scipy_mvn.pdf(x)
      self.assertEqual((), log_pdf.get_shape())
      self.assertEqual((), pdf.get_shape())
      self.assertAllClose(expected_log_pdf, log_pdf.eval())
      self.assertAllClose(expected_pdf, pdf.eval())

  def testLogPDFScalarBatchCovarianceNotProvided(self):
    with self.test_session():
      mu = rng.rand(2)
      mvn = ds.MultivariateNormalFullCovariance(
          mu, covariance_matrix=None, validate_args=True)
      x = rng.rand(2)

      log_pdf = mvn.log_prob(x)
      pdf = mvn.prob(x)

      # Initialize a scipy_mvn with the default covariance.
      scipy_mvn = stats.multivariate_normal(mean=mu, cov=np.eye(2))

      expected_log_pdf = scipy_mvn.logpdf(x)
      expected_pdf = scipy_mvn.pdf(x)
      self.assertEqual((), log_pdf.get_shape())
      self.assertEqual((), pdf.get_shape())
      self.assertAllClose(expected_log_pdf, log_pdf.eval())
      self.assertAllClose(expected_pdf, pdf.eval())

  def testShapes(self):
    with self.test_session():
      mu = rng.rand(3, 5, 2)
      covariance = self._random_pd_matrix(3, 5, 2, 2)

      mvn = ds.MultivariateNormalFullCovariance(
          mu, covariance, validate_args=True)

      # Shapes known at graph construction time.
      self.assertEqual((2,), tuple(mvn.event_shape.as_list()))
      self.assertEqual((3, 5), tuple(mvn.batch_shape.as_list()))

      # Shapes known at runtime.
      self.assertEqual((2,), tuple(mvn.event_shape_tensor().eval()))
      self.assertEqual((3, 5), tuple(mvn.batch_shape_tensor().eval()))

  def _random_mu_and_sigma(self, batch_shape, event_shape):
    # This ensures sigma is positive def.
    mat_shape = batch_shape + event_shape + event_shape
    mat = rng.randn(*mat_shape)
    perm = np.arange(mat.ndim)
    perm[-2:] = [perm[-1], perm[-2]]
    sigma = np.matmul(mat, np.transpose(mat, perm))

    mu_shape = batch_shape + event_shape
    mu = rng.randn(*mu_shape)

    return mu, sigma

  def testKLBatch(self):
    batch_shape = (2,)
    event_shape = (3,)
    with self.test_session():
      mu_a, sigma_a = self._random_mu_and_sigma(batch_shape, event_shape)
      mu_b, sigma_b = self._random_mu_and_sigma(batch_shape, event_shape)
      mvn_a = ds.MultivariateNormalFullCovariance(
          loc=mu_a,
          covariance_matrix=sigma_a,
          validate_args=True)
      mvn_b = ds.MultivariateNormalFullCovariance(
          loc=mu_b,
          covariance_matrix=sigma_b,
          validate_args=True)

      kl = ds.kl_divergence(mvn_a, mvn_b)
      self.assertEqual(batch_shape, kl.get_shape())

      kl_v = kl.eval()
      expected_kl_0 = _compute_non_batch_kl(mu_a[0, :], sigma_a[0, :, :],
                                            mu_b[0, :], sigma_b[0, :])
      expected_kl_1 = _compute_non_batch_kl(mu_a[1, :], sigma_a[1, :, :],
                                            mu_b[1, :], sigma_b[1, :])
      self.assertAllClose(expected_kl_0, kl_v[0])
      self.assertAllClose(expected_kl_1, kl_v[1])


def _compute_non_batch_kl(mu_a, sigma_a, mu_b, sigma_b):
  """Non-batch KL for N(mu_a, sigma_a), N(mu_b, sigma_b)."""
  # Check using numpy operations
  # This mostly repeats the tensorflow code _kl_mvn_mvn(), but in numpy.
  # So it is important to also check that KL(mvn, mvn) = 0.
  sigma_b_inv = np.linalg.inv(sigma_b)

  t = np.trace(sigma_b_inv.dot(sigma_a))
  q = (mu_b - mu_a).dot(sigma_b_inv).dot(mu_b - mu_a)
  k = mu_a.shape[0]
  l = np.log(np.linalg.det(sigma_b) / np.linalg.det(sigma_a))

  return 0.5 * (t + q - k + l)


if __name__ == "__main__":
  test.main()
