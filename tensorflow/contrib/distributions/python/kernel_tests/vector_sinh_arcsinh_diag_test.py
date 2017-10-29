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
"""Tests for VectorSinhArcsinhDiag."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib import distributions
from tensorflow.contrib.distributions.python.ops import test_util
from tensorflow.python.platform import test

ds = distributions
rng = np.random.RandomState(123)


class VectorSinhArcsinhDiagTest(test_util.VectorDistributionTestHelpers,
                                test.TestCase):

  def test_default_is_same_as_normal(self):
    d = 10
    scale_diag = rng.rand(d)
    scale_identity_multiplier = np.float64(1.0)
    loc = rng.randn(d)
    with self.test_session() as sess:
      norm = ds.MultivariateNormalDiag(
          loc=loc,
          scale_diag=scale_diag,
          scale_identity_multiplier=scale_identity_multiplier,
          validate_args=True)
      sasnorm = ds.VectorSinhArcsinhDiag(
          loc=loc,
          scale_diag=scale_diag,
          scale_identity_multiplier=scale_identity_multiplier,
          validate_args=True)

      x = rng.randn(5, d)
      norm_pdf, sasnorm_pdf = sess.run([norm.prob(x), sasnorm.prob(x)])
      self.assertAllClose(norm_pdf, sasnorm_pdf)

      norm_samps, sasnorm_samps = sess.run(
          [norm.sample(10000, seed=0),
           sasnorm.sample(10000, seed=0)])
      self.assertAllClose(loc, sasnorm_samps.mean(axis=0), atol=0.1)
      self.assertAllClose(
          norm_samps.mean(axis=0), sasnorm_samps.mean(axis=0), atol=0.1)
      self.assertAllClose(
          norm_samps.std(axis=0), sasnorm_samps.std(axis=0), atol=0.1)

  def test_passing_in_laplace_plus_defaults_is_same_as_laplace(self):
    d = 10
    scale_diag = rng.rand(d)
    scale_identity_multiplier = np.float64(1.2)
    loc = rng.randn(d)
    with self.test_session() as sess:
      vlap = ds.VectorLaplaceDiag(
          loc=loc,
          scale_diag=scale_diag,
          scale_identity_multiplier=scale_identity_multiplier,
          validate_args=True)
      sasvlap = ds.VectorSinhArcsinhDiag(
          loc=loc,
          scale_diag=scale_diag,
          scale_identity_multiplier=scale_identity_multiplier,
          distribution=ds.Laplace(np.float64(0.), np.float64(1.)),
          validate_args=True)

      x = rng.randn(5, d)
      vlap_pdf, sasvlap_pdf = sess.run([vlap.prob(x), sasvlap.prob(x)])
      self.assertAllClose(vlap_pdf, sasvlap_pdf)

      vlap_samps, sasvlap_samps = sess.run(
          [vlap.sample(10000, seed=0),
           sasvlap.sample(10000, seed=0)])
      self.assertAllClose(loc, sasvlap_samps.mean(axis=0), atol=0.1)
      self.assertAllClose(
          vlap_samps.mean(axis=0), sasvlap_samps.mean(axis=0), atol=0.1)
      self.assertAllClose(
          vlap_samps.std(axis=0), sasvlap_samps.std(axis=0), atol=0.1)

  def test_tailweight_small_gives_fewer_outliers_than_normal(self):
    d = 10
    scale_diag = rng.rand(d)
    scale_identity_multiplier = np.float64(0.9)
    loc = rng.randn(d)
    with self.test_session() as sess:
      norm = ds.MultivariateNormalDiag(
          loc=loc,
          scale_diag=scale_diag,
          scale_identity_multiplier=scale_identity_multiplier,
          validate_args=True)
      sasnorm = ds.VectorSinhArcsinhDiag(
          loc=loc,
          scale_diag=scale_diag,
          scale_identity_multiplier=scale_identity_multiplier,
          tailweight=0.1,
          validate_args=True)

      # sasnorm.pdf(x) is smaller on outliers (+-10 are outliers)
      x = np.float64([[-10] * d, [10] * d])  # Shape [2, 10]
      norm_lp, sasnorm_lp = sess.run([norm.log_prob(x), sasnorm.log_prob(x)])
      np.testing.assert_array_less(sasnorm_lp, norm_lp)

      # 0.1% quantile and 99.9% quantile are outliers, and should be more
      # extreme in the normal.  The 97.772% quantiles should be the same.
      norm_samps, sasnorm_samps = sess.run(
          [norm.sample(int(5e5), seed=1),
           sasnorm.sample(int(5e5), seed=1)])
      np.testing.assert_array_less(
          np.percentile(norm_samps, 0.1, axis=0),
          np.percentile(sasnorm_samps, 0.1, axis=0))
      np.testing.assert_array_less(
          np.percentile(sasnorm_samps, 99.9, axis=0),
          np.percentile(norm_samps, 99.9, axis=0))
      # 100. * sp.stats.norm.cdf(2.)
      q = 100 * 0.97724986805182079
      self.assertAllClose(
          np.percentile(sasnorm_samps, q, axis=0),
          np.percentile(norm_samps, q, axis=0),
          rtol=0.03)
      self.assertAllClose(
          np.percentile(sasnorm_samps, 100 - q, axis=0),
          np.percentile(norm_samps, 100 - q, axis=0),
          rtol=0.03)

  def test_tailweight_large_gives_more_outliers_than_normal(self):
    d = 10
    scale_diag = rng.rand(d)
    scale_identity_multiplier = np.float64(1.0)
    loc = rng.randn(d)
    with self.test_session() as sess:
      norm = ds.MultivariateNormalDiag(
          loc=loc,
          scale_diag=scale_diag,
          scale_identity_multiplier=scale_identity_multiplier,
          validate_args=True)
      sasnorm = ds.VectorSinhArcsinhDiag(
          loc=loc,
          scale_diag=scale_diag,
          scale_identity_multiplier=scale_identity_multiplier,
          tailweight=3.,
          validate_args=True)

      # norm.pdf(x) is smaller on outliers (+-10 are outliers)
      x = np.float64([[-10] * d, [10] * d])  # Shape [2, 10]
      norm_lp, sasnorm_lp = sess.run([norm.log_prob(x), sasnorm.log_prob(x)])
      np.testing.assert_array_less(norm_lp, sasnorm_lp)

      # 0.1% quantile and 99.9% quantile are outliers, and should be more
      # extreme in the sasnormal.  The 97.772% quantiles should be the same.
      norm_samps, sasnorm_samps = sess.run(
          [norm.sample(int(5e5), seed=2),
           sasnorm.sample(int(5e5), seed=2)])
      np.testing.assert_array_less(
          np.percentile(sasnorm_samps, 0.1, axis=0),
          np.percentile(norm_samps, 0.1, axis=0))
      np.testing.assert_array_less(
          np.percentile(norm_samps, 99.9, axis=0),
          np.percentile(sasnorm_samps, 99.9, axis=0))
      # 100. * sp.stats.norm.cdf(2.)
      q = 100 * 0.97724986805182079
      self.assertAllClose(
          np.percentile(sasnorm_samps, q, axis=0),
          np.percentile(norm_samps, q, axis=0),
          rtol=0.03)
      self.assertAllClose(
          np.percentile(sasnorm_samps, 100 - q, axis=0),
          np.percentile(norm_samps, 100 - q, axis=0),
          rtol=0.03)

  def test_positive_skewness_moves_mean_to_the_right(self):
    d = 10
    scale_diag = rng.rand(d)
    scale_identity_multiplier = np.float64(1.0)
    loc = rng.randn(d)
    with self.test_session() as sess:
      sasnorm = ds.VectorSinhArcsinhDiag(
          loc=loc,
          scale_diag=scale_diag,
          scale_identity_multiplier=scale_identity_multiplier,
          skewness=3.0,
          validate_args=True)

      sasnorm_samps = sess.run(sasnorm.sample(10000, seed=4))
      np.testing.assert_array_less(loc, sasnorm_samps.mean(axis=0))

  def test_consistency_random_parameters_with_batch_dim(self):
    b, d = 5, 2
    scale_diag = rng.rand(b, d)
    scale_identity_multiplier = np.float64(1.1)
    with self.test_session() as sess:
      sasnorm = ds.VectorSinhArcsinhDiag(
          scale_diag=scale_diag,
          scale_identity_multiplier=scale_identity_multiplier,
          skewness=rng.randn(d) * 0.5,
          tailweight=rng.rand(b, d) + 0.7,
          validate_args=True)

      self.run_test_sample_consistent_log_prob(
          sess.run, sasnorm, radius=1.0, center=0., rtol=0.1)
      self.run_test_sample_consistent_log_prob(
          sess.run,
          sasnorm,
          radius=1.0,
          center=-0.15,
          rtol=0.1)
      self.run_test_sample_consistent_log_prob(
          sess.run,
          sasnorm,
          radius=1.0,
          center=0.15,
          rtol=0.1)

  def test_consistency_random_parameters_no_batch_dims(self):
    d = 3
    scale_diag = rng.rand(d)
    scale_identity_multiplier = np.float64(1.1)
    with self.test_session() as sess:
      sasnorm = ds.VectorSinhArcsinhDiag(
          scale_diag=scale_diag,
          scale_identity_multiplier=scale_identity_multiplier,
          skewness=rng.randn(d) * 0.5,
          tailweight=rng.rand(d) + 0.7,
          validate_args=True)

      self.run_test_sample_consistent_log_prob(
          sess.run, sasnorm, radius=1.0, center=0., rtol=0.1)
      self.run_test_sample_consistent_log_prob(
          sess.run,
          sasnorm,
          radius=1.0,
          center=-0.15,
          rtol=0.1)
      self.run_test_sample_consistent_log_prob(
          sess.run,
          sasnorm,
          radius=1.0,
          center=0.15,
          rtol=0.1)

  def test_pdf_reflected_for_negative_skewness(self):
    with self.test_session() as sess:
      sas_pos_skew = ds.VectorSinhArcsinhDiag(
          loc=[0.],
          scale_identity_multiplier=1.,
          skewness=2.,
          validate_args=True)
      sas_neg_skew = ds.VectorSinhArcsinhDiag(
          loc=[0.],
          scale_identity_multiplier=1.,
          skewness=-2.,
          validate_args=True)
      x = np.linspace(-2, 2, num=5).astype(np.float32).reshape(5, 1)
      self.assertAllClose(
          *sess.run([sas_pos_skew.prob(x), sas_neg_skew.prob(x[::-1])]))


if __name__ == "__main__":
  test.main()
