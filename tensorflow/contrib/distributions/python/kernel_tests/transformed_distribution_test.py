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
"""Tests for TransformedDistribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats

from tensorflow.contrib import distributions
from tensorflow.contrib import linalg
from tensorflow.contrib.distributions.python.ops import bijectors
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

bs = bijectors
ds = distributions
la = linalg


class TransformedDistributionTest(test.TestCase):

  def _cls(self):
    return ds.TransformedDistribution

  def testTransformedDistribution(self):
    g = ops.Graph()
    with g.as_default():
      mu = 3.0
      sigma = 2.0
      # Note: the Jacobian callable only works for this example; more generally
      # you may or may not need a reduce_sum.
      log_normal = self._cls()(
          distribution=ds.Normal(loc=mu, scale=sigma),
          bijector=bs.Exp(event_ndims=0))
      sp_dist = stats.lognorm(s=sigma, scale=np.exp(mu))

      # sample
      sample = log_normal.sample(100000, seed=235)
      self.assertAllEqual([], log_normal.event_shape)
      with self.test_session(graph=g):
        self.assertAllEqual([], log_normal.event_shape_tensor().eval())
        self.assertAllClose(
            sp_dist.mean(), np.mean(sample.eval()), atol=0.0, rtol=0.05)

      # pdf, log_pdf, cdf, etc...
      # The mean of the lognormal is around 148.
      test_vals = np.linspace(0.1, 1000., num=20).astype(np.float32)
      for func in [[log_normal.log_prob, sp_dist.logpdf],
                   [log_normal.prob, sp_dist.pdf],
                   [log_normal.log_cdf, sp_dist.logcdf],
                   [log_normal.cdf, sp_dist.cdf],
                   [log_normal.survival_function, sp_dist.sf],
                   [log_normal.log_survival_function, sp_dist.logsf]]:
        actual = func[0](test_vals)
        expected = func[1](test_vals)
        with self.test_session(graph=g):
          self.assertAllClose(expected, actual.eval(), atol=0, rtol=0.01)

  def testCachedSamplesWithoutInverse(self):
    with self.test_session() as sess:
      mu = 3.0
      sigma = 0.02
      log_normal = self._cls()(
          distribution=ds.Normal(loc=mu, scale=sigma),
          bijector=bs.Exp(event_ndims=0))

      sample = log_normal.sample(1)
      sample_val, log_pdf_val = sess.run([sample, log_normal.log_prob(sample)])
      self.assertAllClose(
          stats.lognorm.logpdf(sample_val, s=sigma, scale=np.exp(mu)),
          log_pdf_val,
          atol=1e-2)

  def testShapeChangingBijector(self):
    with self.test_session():
      softmax = bs.SoftmaxCentered()
      standard_normal = ds.Normal(loc=0., scale=1.)
      multi_logit_normal = self._cls()(
          distribution=standard_normal,
          bijector=softmax)
      x = [[-np.log(3.), 0.],
           [np.log(3), np.log(5)]]
      y = softmax.forward(x).eval()
      expected_log_pdf = (stats.norm(loc=0., scale=1.).logpdf(x) -
                          np.sum(np.log(y), axis=-1))
      self.assertAllClose(expected_log_pdf,
                          multi_logit_normal.log_prob(y).eval())
      self.assertAllClose(
          [1, 2, 3, 2],
          array_ops.shape(multi_logit_normal.sample([1, 2, 3])).eval())
      self.assertAllEqual([2], multi_logit_normal.event_shape)
      self.assertAllEqual([2], multi_logit_normal.event_shape_tensor().eval())

  def testEntropy(self):
    with self.test_session():
      shift = np.array([[-1, 0, 1], [-1, -2, -3]], dtype=np.float32)
      diag = np.array([[1, 2, 3], [2, 3, 2]], dtype=np.float32)
      actual_mvn_entropy = np.concatenate([
          [stats.multivariate_normal(shift[i], np.diag(diag[i]**2)).entropy()]
          for i in range(len(diag))])
      fake_mvn = self._cls()(
          ds.MultivariateNormalDiag(
              loc=array_ops.zeros_like(shift),
              scale_diag=array_ops.ones_like(diag),
              validate_args=True),
          bs.AffineLinearOperator(
              shift,
              scale=la.LinearOperatorDiag(diag, is_non_singular=True),
              validate_args=True),
          validate_args=True)
      self.assertAllClose(actual_mvn_entropy,
                          fake_mvn.entropy().eval())


class ScalarToMultiTest(test.TestCase):

  def _cls(self):
    return ds.TransformedDistribution

  def setUp(self):
    self._shift = np.array([-1, 0, 1], dtype=np.float32)
    self._tril = np.array([[[1., 0, 0],
                            [2, 1, 0],
                            [3, 2, 1]],
                           [[2, 0, 0],
                            [3, 2, 0],
                            [4, 3, 2]]],
                          dtype=np.float32)

  def _testMVN(self,
               base_distribution_class,
               base_distribution_kwargs,
               batch_shape=(),
               event_shape=(),
               not_implemented_message=None):
    with self.test_session() as sess:
      # Overriding shapes must be compatible w/bijector; most bijectors are
      # batch_shape agnostic and only care about event_ndims.
      # In the case of `Affine`, if we got it wrong then it would fire an
      # exception due to incompatible dimensions.
      batch_shape_pl = array_ops.placeholder(
          dtypes.int32, name="dynamic_batch_shape")
      event_shape_pl = array_ops.placeholder(
          dtypes.int32, name="dynamic_event_shape")
      feed_dict = {batch_shape_pl: np.array(batch_shape, dtype=np.int32),
                   event_shape_pl: np.array(event_shape, dtype=np.int32)}
      fake_mvn_dynamic = self._cls()(
          distribution=base_distribution_class(validate_args=True,
                                               **base_distribution_kwargs),
          bijector=bs.Affine(shift=self._shift, scale_tril=self._tril),
          batch_shape=batch_shape_pl,
          event_shape=event_shape_pl,
          validate_args=True)

      fake_mvn_static = self._cls()(
          distribution=base_distribution_class(validate_args=True,
                                               **base_distribution_kwargs),
          bijector=bs.Affine(shift=self._shift, scale_tril=self._tril),
          batch_shape=batch_shape,
          event_shape=event_shape,
          validate_args=True)

      actual_mean = np.tile(self._shift, [2, 1])  # Affine elided this tile.
      actual_cov = np.matmul(self._tril, np.transpose(self._tril, [0, 2, 1]))

      def actual_mvn_log_prob(x):
        return np.concatenate([
            [stats.multivariate_normal(
                actual_mean[i], actual_cov[i]).logpdf(x[:, i, :])]
            for i in range(len(actual_cov))]).T

      actual_mvn_entropy = np.concatenate([
          [stats.multivariate_normal(
              actual_mean[i], actual_cov[i]).entropy()]
          for i in range(len(actual_cov))])

      self.assertAllEqual([3], fake_mvn_static.event_shape)
      self.assertAllEqual([2], fake_mvn_static.batch_shape)

      self.assertAllEqual(tensor_shape.TensorShape(None),
                          fake_mvn_dynamic.event_shape)
      self.assertAllEqual(tensor_shape.TensorShape(None),
                          fake_mvn_dynamic.batch_shape)

      x = fake_mvn_static.sample(5, seed=0).eval()
      for unsupported_fn in (fake_mvn_static.log_cdf,
                             fake_mvn_static.cdf,
                             fake_mvn_static.survival_function,
                             fake_mvn_static.log_survival_function):
        with self.assertRaisesRegexp(NotImplementedError,
                                     not_implemented_message):
          unsupported_fn(x)

      num_samples = 5e3
      for fake_mvn, feed_dict in ((fake_mvn_static, {}),
                                  (fake_mvn_dynamic, feed_dict)):
        # Ensure sample works by checking first, second moments.
        y = fake_mvn.sample(int(num_samples), seed=0)
        x = y[0:5, ...]
        sample_mean = math_ops.reduce_mean(y, 0)
        centered_y = array_ops.transpose(y - sample_mean, [1, 2, 0])
        sample_cov = math_ops.matmul(
            centered_y, centered_y, transpose_b=True) / num_samples
        [
            sample_mean_,
            sample_cov_,
            x_,
            fake_event_shape_,
            fake_batch_shape_,
            fake_log_prob_,
            fake_prob_,
            fake_entropy_,
        ] = sess.run([
            sample_mean,
            sample_cov,
            x,
            fake_mvn.event_shape_tensor(),
            fake_mvn.batch_shape_tensor(),
            fake_mvn.log_prob(x),
            fake_mvn.prob(x),
            fake_mvn.entropy(),
        ], feed_dict=feed_dict)

        self.assertAllClose(actual_mean, sample_mean_, atol=0.1, rtol=0.1)
        self.assertAllClose(actual_cov, sample_cov_, atol=0., rtol=0.1)

        # Ensure all other functions work as intended.
        self.assertAllEqual([5, 2, 3], x_.shape)
        self.assertAllEqual([3], fake_event_shape_)
        self.assertAllEqual([2], fake_batch_shape_)
        self.assertAllClose(actual_mvn_log_prob(x_), fake_log_prob_,
                            atol=0., rtol=1e-6)
        self.assertAllClose(np.exp(actual_mvn_log_prob(x_)), fake_prob_,
                            atol=0., rtol=1e-5)
        self.assertAllClose(actual_mvn_entropy, fake_entropy_,
                            atol=0., rtol=1e-6)

  def testScalarBatchScalarEvent(self):
    self._testMVN(
        base_distribution_class=ds.Normal,
        base_distribution_kwargs={"loc": 0., "scale": 1.},
        batch_shape=[2],
        event_shape=[3],
        not_implemented_message="not implemented when overriding event_shape")

  def testScalarBatchNonScalarEvent(self):
    self._testMVN(
        base_distribution_class=ds.MultivariateNormalDiag,
        base_distribution_kwargs={"loc": [0., 0., 0.],
                                  "scale_diag": [1., 1, 1]},
        batch_shape=[2],
        not_implemented_message="not implemented")

    with self.test_session():
      # Can't override event_shape for scalar batch, non-scalar event.
      with self.assertRaisesRegexp(ValueError, "base distribution not scalar"):
        self._cls()(
            distribution=ds.MultivariateNormalDiag(loc=[0.], scale_diag=[1.]),
            bijector=bs.Affine(shift=self._shift, scale_tril=self._tril),
            batch_shape=[2],
            event_shape=[3],
            validate_args=True)

  def testNonScalarBatchScalarEvent(self):
    self._testMVN(
        base_distribution_class=ds.Normal,
        base_distribution_kwargs={"loc": [0., 0], "scale": [1., 1]},
        event_shape=[3],
        not_implemented_message="not implemented when overriding event_shape")

    with self.test_session():
      # Can't override batch_shape for non-scalar batch, scalar event.
      with self.assertRaisesRegexp(ValueError, "base distribution not scalar"):
        self._cls()(
            distribution=ds.Normal(loc=[0.], scale=[1.]),
            bijector=bs.Affine(shift=self._shift, scale_tril=self._tril),
            batch_shape=[2],
            event_shape=[3],
            validate_args=True)

  def testNonScalarBatchNonScalarEvent(self):
    with self.test_session():
      # Can't override event_shape and/or batch_shape for non_scalar batch,
      # non-scalar event.
      with self.assertRaisesRegexp(ValueError, "base distribution not scalar"):
        self._cls()(
            distribution=ds.MultivariateNormalDiag(loc=[[0.]],
                                                   scale_diag=[[1.]]),
            bijector=bs.Affine(shift=self._shift, scale_tril=self._tril),
            batch_shape=[2],
            event_shape=[3],
            validate_args=True)


if __name__ == "__main__":
  test.main()
