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
"""Tests for SinhArcsinh."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib import distributions
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

ds = distributions
rng = np.random.RandomState(123)


class SinhArcsinhTest(test.TestCase):

  def test_default_is_same_as_normal(self):
    b = 10
    scale = rng.rand(b) + 0.5
    loc = rng.randn(b)
    with self.cached_session() as sess:
      norm = ds.Normal(
          loc=loc,
          scale=scale,
          validate_args=True)
      sasnorm = ds.SinhArcsinh(
          loc=loc,
          scale=scale,
          validate_args=True)

      x = rng.randn(5, b)
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

  def test_broadcast_params_dynamic(self):
    with self.cached_session() as sess:
      loc = array_ops.placeholder(dtypes.float64)
      scale = array_ops.placeholder(dtypes.float64)
      skewness = array_ops.placeholder(dtypes.float64)
      sasnorm = ds.SinhArcsinh(
          loc=loc,
          scale=scale,
          skewness=skewness,
          validate_args=True)

      samp = sess.run(sasnorm.sample(),
                      feed_dict={loc: rng.rand(5),
                                 scale: np.float64(rng.rand()),  # Scalar
                                 skewness: rng.rand(5)})
      self.assertAllEqual((5,), samp.shape)

  def test_passing_in_laplace_plus_defaults_is_same_as_laplace(self):
    b = 10
    scale = rng.rand(b) + 0.5
    loc = rng.randn(b)
    with self.cached_session() as sess:
      lap = ds.Laplace(
          loc=loc,
          scale=scale,
          validate_args=True)
      saslap = ds.SinhArcsinh(
          loc=loc,
          scale=scale,
          distribution=ds.Laplace(np.float64(0), np.float64(1)),
          validate_args=True)

      x = rng.randn(5, b)
      lap_pdf, saslap_pdf = sess.run([lap.prob(x), saslap.prob(x)])
      self.assertAllClose(lap_pdf, saslap_pdf)

      lap_samps, saslap_samps = sess.run(
          [lap.sample(10000, seed=0),
           saslap.sample(10000, seed=0)])
      self.assertAllClose(loc, saslap_samps.mean(axis=0), atol=0.1)
      self.assertAllClose(
          lap_samps.mean(axis=0), saslap_samps.mean(axis=0), atol=0.1)
      self.assertAllClose(
          lap_samps.std(axis=0), saslap_samps.std(axis=0), atol=0.1)

  def test_tailweight_small_gives_fewer_outliers_than_normal(self):
    batch_size = 10
    scale = rng.rand(batch_size) + 0.5
    loc = 0.1 * rng.randn(batch_size)
    with self.cached_session() as sess:
      norm = ds.Normal(
          loc=loc,
          scale=scale,
          validate_args=True)
      sasnorm = ds.SinhArcsinh(
          loc=loc,
          scale=scale,
          tailweight=0.1,
          validate_args=True)

      # sasnorm.pdf(x) is smaller on outliers (+-10 are outliers)
      x = np.float64([[-10] * batch_size, [10] * batch_size])  # Shape [2, 10]
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
    batch_size = 10
    scale = rng.rand(batch_size) + 0.5
    loc = np.float64(0.)
    with self.cached_session() as sess:
      norm = ds.Normal(
          loc=loc,
          scale=scale,
          validate_args=True)
      sasnorm = ds.SinhArcsinh(
          loc=loc,
          scale=scale,
          tailweight=3.,
          validate_args=True)

      # norm.pdf(x) is smaller on outliers (+-10 are outliers)
      x = np.float64([[-10] * batch_size, [10] * batch_size])  # Shape [2, 10]
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
    batch_size = 10
    scale = rng.rand(batch_size) + 0.5
    loc = rng.randn(batch_size)
    with self.cached_session() as sess:
      sasnorm = ds.SinhArcsinh(
          loc=loc,
          scale=scale,
          skewness=3.0,
          validate_args=True)

      sasnorm_samps = sess.run(sasnorm.sample(10000, seed=4))
      np.testing.assert_array_less(loc, sasnorm_samps.mean(axis=0))

  def test_pdf_reflected_for_negative_skewness(self):
    with self.cached_session() as sess:
      sas_pos_skew = ds.SinhArcsinh(
          loc=0.,
          scale=1.,
          skewness=2.,
          validate_args=True)
      sas_neg_skew = ds.SinhArcsinh(
          loc=0.,
          scale=1.,
          skewness=-2.,
          validate_args=True)
      x = np.linspace(-2, 2, num=5).astype(np.float32)
      self.assertAllClose(
          *sess.run([sas_pos_skew.prob(x), sas_neg_skew.prob(x[::-1])]))


if __name__ == "__main__":
  test.main()
