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
"""Tests for Wishart."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import linalg
from tensorflow.contrib import distributions as distributions_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

distributions = distributions_lib


def make_pd(start, n):
  """Deterministically create a positive definite matrix."""
  x = np.tril(linalg.circulant(np.arange(start, start + n)))
  return np.dot(x, x.T)


def chol(x):
  """Compute Cholesky factorization."""
  return linalg.cholesky(x).T


def wishart_var(df, x):
  """Compute Wishart variance for numpy scale matrix."""
  x = np.sqrt(df) * np.asarray(x)
  d = np.expand_dims(np.diag(x), -1)
  return x**2 + np.dot(d, d.T)


class WishartCholeskyTest(test.TestCase):

  def testEntropy(self):
    with self.test_session():
      scale = make_pd(1., 2)
      df = 4
      w = distributions.WishartCholesky(df, chol(scale))
      # sp.stats.wishart(df=4, scale=make_pd(1., 2)).entropy()
      self.assertAllClose(6.301387092430769, w.entropy().eval())

      w = distributions.WishartCholesky(df=1, scale=[[1.]])
      # sp.stats.wishart(df=1,scale=1).entropy()
      self.assertAllClose(0.78375711047393404, w.entropy().eval())

  def testMeanLogDetAndLogNormalizingConstant(self):
    with self.test_session():

      def entropy_alt(w):
        return (
            w.log_normalization()
            - 0.5 * (w.df - w.dimension - 1.) * w.mean_log_det()
            + 0.5 * w.df * w.dimension).eval()

      w = distributions.WishartCholesky(df=4,
                                        scale=chol(make_pd(1., 2)))
      self.assertAllClose(w.entropy().eval(), entropy_alt(w))

      w = distributions.WishartCholesky(df=5, scale=[[1.]])
      self.assertAllClose(w.entropy().eval(), entropy_alt(w))

  def testMean(self):
    with self.test_session():
      scale = make_pd(1., 2)
      df = 4
      w = distributions.WishartCholesky(df, chol(scale))
      self.assertAllEqual(df * scale, w.mean().eval())

  def testMode(self):
    with self.test_session():
      scale = make_pd(1., 2)
      df = 4
      w = distributions.WishartCholesky(df, chol(scale))
      self.assertAllEqual((df - 2. - 1.) * scale, w.mode().eval())

  def testStd(self):
    with self.test_session():
      scale = make_pd(1., 2)
      df = 4
      w = distributions.WishartCholesky(df, chol(scale))
      self.assertAllEqual(chol(wishart_var(df, scale)), w.stddev().eval())

  def testVariance(self):
    with self.test_session():
      scale = make_pd(1., 2)
      df = 4
      w = distributions.WishartCholesky(df, chol(scale))
      self.assertAllEqual(wishart_var(df, scale), w.variance().eval())

  def testSample(self):
    with self.test_session():
      scale = make_pd(1., 2)
      df = 4

      chol_w = distributions.WishartCholesky(
          df, chol(scale), cholesky_input_output_matrices=False)

      x = chol_w.sample(1, seed=42).eval()
      chol_x = [chol(x[0])]

      full_w = distributions.WishartFull(
          df, scale, cholesky_input_output_matrices=False)
      self.assertAllClose(x, full_w.sample(1, seed=42).eval())

      chol_w_chol = distributions.WishartCholesky(
          df, chol(scale), cholesky_input_output_matrices=True)
      self.assertAllClose(chol_x, chol_w_chol.sample(1, seed=42).eval())
      eigen_values = array_ops.matrix_diag_part(
          chol_w_chol.sample(
              1000, seed=42))
      np.testing.assert_array_less(0., eigen_values.eval())

      full_w_chol = distributions.WishartFull(
          df, scale, cholesky_input_output_matrices=True)
      self.assertAllClose(chol_x, full_w_chol.sample(1, seed=42).eval())
      eigen_values = array_ops.matrix_diag_part(
          full_w_chol.sample(
              1000, seed=42))
      np.testing.assert_array_less(0., eigen_values.eval())

      # Check first and second moments.
      df = 4.
      chol_w = distributions.WishartCholesky(
          df=df,
          scale=chol(make_pd(1., 3)),
          cholesky_input_output_matrices=False)
      x = chol_w.sample(10000, seed=42)
      self.assertAllEqual((10000, 3, 3), x.get_shape())

      moment1_estimate = math_ops.reduce_mean(x, reduction_indices=[0]).eval()
      self.assertAllClose(chol_w.mean().eval(), moment1_estimate, rtol=0.05)

      # The Variance estimate uses the squares rather than outer-products
      # because Wishart.Variance is the diagonal of the Wishart covariance
      # matrix.
      variance_estimate = (math_ops.reduce_mean(
          math_ops.square(x), reduction_indices=[0]) -
                           math_ops.square(moment1_estimate)).eval()
      self.assertAllClose(
          chol_w.variance().eval(), variance_estimate, rtol=0.05)

  # Test that sampling with the same seed twice gives the same results.
  def testSampleMultipleTimes(self):
    with self.test_session():
      df = 4.
      n_val = 100

      random_seed.set_random_seed(654321)
      chol_w1 = distributions.WishartCholesky(
          df=df,
          scale=chol(make_pd(1., 3)),
          cholesky_input_output_matrices=False,
          name="wishart1")
      samples1 = chol_w1.sample(n_val, seed=123456).eval()

      random_seed.set_random_seed(654321)
      chol_w2 = distributions.WishartCholesky(
          df=df,
          scale=chol(make_pd(1., 3)),
          cholesky_input_output_matrices=False,
          name="wishart2")
      samples2 = chol_w2.sample(n_val, seed=123456).eval()

      self.assertAllClose(samples1, samples2)

  def testProb(self):
    with self.test_session():
      # Generate some positive definite (pd) matrices and their Cholesky
      # factorizations.
      x = np.array(
          [make_pd(1., 2), make_pd(2., 2), make_pd(3., 2), make_pd(4., 2)])
      chol_x = np.array([chol(x[0]), chol(x[1]), chol(x[2]), chol(x[3])])

      # Since Wishart wasn"t added to SciPy until 0.16, we'll spot check some
      # pdfs with hard-coded results from upstream SciPy.

      log_prob_df_seq = np.array([
          # math.log(stats.wishart.pdf(x[0], df=2+0, scale=x[0]))
          -3.5310242469692907,
          # math.log(stats.wishart.pdf(x[1], df=2+1, scale=x[1]))
          -7.689907330328961,
          # math.log(stats.wishart.pdf(x[2], df=2+2, scale=x[2]))
          -10.815845159537895,
          # math.log(stats.wishart.pdf(x[3], df=2+3, scale=x[3]))
          -13.640549882916691,
      ])

      # This test checks that batches don't interfere with correctness.
      w = distributions.WishartCholesky(
          df=[2, 3, 4, 5],
          scale=chol_x,
          cholesky_input_output_matrices=True)
      self.assertAllClose(log_prob_df_seq, w.log_prob(chol_x).eval())

      # Now we test various constructions of Wishart with different sample
      # shape.

      log_prob = np.array([
          # math.log(stats.wishart.pdf(x[0], df=4, scale=x[0]))
          -4.224171427529236,
          # math.log(stats.wishart.pdf(x[1], df=4, scale=x[0]))
          -6.3378770664093453,
          # math.log(stats.wishart.pdf(x[2], df=4, scale=x[0]))
          -12.026946850193017,
          # math.log(stats.wishart.pdf(x[3], df=4, scale=x[0]))
          -20.951582705289454,
      ])

      for w in (
          distributions.WishartCholesky(
              df=4,
              scale=chol_x[0],
              cholesky_input_output_matrices=False),
          distributions.WishartFull(
              df=4,
              scale=x[0],
              cholesky_input_output_matrices=False)):
        self.assertAllEqual((2, 2), w.event_shape_tensor().eval())
        self.assertEqual(2, w.dimension.eval())
        self.assertAllClose(log_prob[0], w.log_prob(x[0]).eval())
        self.assertAllClose(log_prob[0:2], w.log_prob(x[0:2]).eval())
        self.assertAllClose(
            np.reshape(log_prob, (2, 2)),
            w.log_prob(np.reshape(x, (2, 2, 2, 2))).eval())
        self.assertAllClose(
            np.reshape(np.exp(log_prob), (2, 2)),
            w.prob(np.reshape(x, (2, 2, 2, 2))).eval())
        self.assertAllEqual((2, 2),
                            w.log_prob(np.reshape(x, (2, 2, 2, 2))).get_shape())

      for w in (
          distributions.WishartCholesky(
              df=4,
              scale=chol_x[0],
              cholesky_input_output_matrices=True),
          distributions.WishartFull(
              df=4,
              scale=x[0],
              cholesky_input_output_matrices=True)):
        self.assertAllEqual((2, 2), w.event_shape_tensor().eval())
        self.assertEqual(2, w.dimension.eval())
        self.assertAllClose(log_prob[0], w.log_prob(chol_x[0]).eval())
        self.assertAllClose(log_prob[0:2], w.log_prob(chol_x[0:2]).eval())
        self.assertAllClose(
            np.reshape(log_prob, (2, 2)),
            w.log_prob(np.reshape(chol_x, (2, 2, 2, 2))).eval())
        self.assertAllClose(
            np.reshape(np.exp(log_prob), (2, 2)),
            w.prob(np.reshape(chol_x, (2, 2, 2, 2))).eval())
        self.assertAllEqual((2, 2),
                            w.log_prob(np.reshape(x, (2, 2, 2, 2))).get_shape())

  def testBatchShape(self):
    with self.test_session() as sess:
      scale = make_pd(1., 2)
      chol_scale = chol(scale)

      w = distributions.WishartCholesky(df=4, scale=chol_scale)
      self.assertAllEqual([], w.batch_shape)
      self.assertAllEqual([], w.batch_shape_tensor().eval())

      w = distributions.WishartCholesky(
          df=[4., 4], scale=np.array([chol_scale, chol_scale]))
      self.assertAllEqual([2], w.batch_shape)
      self.assertAllEqual([2], w.batch_shape_tensor().eval())

      scale_deferred = array_ops.placeholder(dtypes.float32)
      w = distributions.WishartCholesky(df=4, scale=scale_deferred)
      self.assertAllEqual(
          [], sess.run(w.batch_shape_tensor(),
                       feed_dict={scale_deferred: chol_scale}))
      self.assertAllEqual(
          [2],
          sess.run(w.batch_shape_tensor(),
                   feed_dict={scale_deferred: [chol_scale, chol_scale]}))

  def testEventShape(self):
    with self.test_session() as sess:
      scale = make_pd(1., 2)
      chol_scale = chol(scale)

      w = distributions.WishartCholesky(df=4, scale=chol_scale)
      self.assertAllEqual([2, 2], w.event_shape)
      self.assertAllEqual([2, 2], w.event_shape_tensor().eval())

      w = distributions.WishartCholesky(
          df=[4., 4], scale=np.array([chol_scale, chol_scale]))
      self.assertAllEqual([2, 2], w.event_shape)
      self.assertAllEqual([2, 2], w.event_shape_tensor().eval())

      scale_deferred = array_ops.placeholder(dtypes.float32)
      w = distributions.WishartCholesky(df=4, scale=scale_deferred)
      self.assertAllEqual(
          [2, 2],
          sess.run(w.event_shape_tensor(),
                   feed_dict={scale_deferred: chol_scale}))
      self.assertAllEqual(
          [2, 2],
          sess.run(w.event_shape_tensor(),
                   feed_dict={scale_deferred: [chol_scale, chol_scale]}))

  def testValidateArgs(self):
    with self.test_session() as sess:
      df_deferred = array_ops.placeholder(dtypes.float32)
      chol_scale_deferred = array_ops.placeholder(dtypes.float32)
      x = make_pd(1., 3)
      chol_scale = chol(x)

      # Check expensive, deferred assertions.
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "cannot be less than"):
        chol_w = distributions.WishartCholesky(
            df=df_deferred,
            scale=chol_scale_deferred,
            validate_args=True)
        sess.run(chol_w.log_prob(np.asarray(
            x, dtype=np.float32)),
                 feed_dict={df_deferred: 2.,
                            chol_scale_deferred: chol_scale})

      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   "Cholesky decomposition was not successful"):
        chol_w = distributions.WishartFull(
            df=df_deferred, scale=chol_scale_deferred)
        # np.ones((3, 3)) is not positive, definite.
        sess.run(chol_w.log_prob(np.asarray(
            x, dtype=np.float32)),
                 feed_dict={
                     df_deferred: 4.,
                     chol_scale_deferred: np.ones(
                         (3, 3), dtype=np.float32)
                 })

      with self.assertRaisesOpError("scale must be square"):
        chol_w = distributions.WishartCholesky(
            df=4.,
            scale=np.array([[2., 3., 4.], [1., 2., 3.]], dtype=np.float32),
            validate_args=True)
        sess.run(chol_w.scale().eval())

      # Ensure no assertions.
      chol_w = distributions.WishartCholesky(
          df=df_deferred,
          scale=chol_scale_deferred,
          validate_args=False)
      sess.run(chol_w.log_prob(np.asarray(
          x, dtype=np.float32)),
               feed_dict={df_deferred: 4,
                          chol_scale_deferred: chol_scale})
      # Bogus log_prob, but since we have no checks running... c"est la vie.
      sess.run(chol_w.log_prob(np.asarray(
          x, dtype=np.float32)),
               feed_dict={df_deferred: 4,
                          chol_scale_deferred: np.ones((3, 3))})

  def testStaticAsserts(self):
    with self.test_session():
      x = make_pd(1., 3)
      chol_scale = chol(x)

      # Still has these assertions because they're resolveable at graph
      # construction
      with self.assertRaisesRegexp(ValueError, "cannot be less than"):
        distributions.WishartCholesky(
            df=2, scale=chol_scale, validate_args=False)
      with self.assertRaisesRegexp(TypeError, "Argument tril must have dtype"):
        distributions.WishartCholesky(
            df=4.,
            scale=np.asarray(
                chol_scale, dtype=np.int32),
            validate_args=False)


if __name__ == "__main__":
  test.main()
