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
import tensorflow as tf

bs = tf.contrib.distributions.bijector
ds = tf.contrib.distributions
la = tf.contrib.linalg


class _ChooseLocation(bs.Bijector):
  """A Bijector which chooses between one of two location parameters."""

  def __init__(self, loc, name="ChooseLocation"):
    self._graph_parents = []
    self._name = name
    with self._name_scope("init", values=[loc]):
      self._loc = tf.convert_to_tensor(loc, name="loc")
      super(_ChooseLocation, self).__init__(
          graph_parents=[self._loc],
          is_constant_jacobian=True,
          validate_args=False,
          name=name)

  def _forward(self, x, z):
    return x + self._gather_loc(z)

  def _inverse(self, x, z):
    return x - self._gather_loc(z)

  def _inverse_log_det_jacobian(self, x, z=None):
    return 0.

  def _gather_loc(self, z):
    z = tf.convert_to_tensor(z)
    z = tf.cast((1 + z) / 2, tf.int32)
    return tf.gather(self._loc, z)


class TransformedDistributionTest(tf.test.TestCase):

  def testTransformedDistribution(self):
    g = tf.Graph()
    with g.as_default():
      mu = 3.0
      sigma = 2.0
      # Note: the Jacobian callable only works for this example; more generally
      # you may or may not need a reduce_sum.
      log_normal = ds.TransformedDistribution(
          distribution=ds.Normal(mu=mu, sigma=sigma),
          bijector=bs.Exp(event_ndims=0))
      sp_dist = stats.lognorm(s=sigma, scale=np.exp(mu))

      # sample
      sample = log_normal.sample(100000, seed=235)
      self.assertAllEqual([], log_normal.get_event_shape())
      with self.test_session(graph=g):
        self.assertAllEqual([], log_normal.event_shape().eval())
        self.assertAllClose(sp_dist.mean(), np.mean(sample.eval()),
                            atol=0.0, rtol=0.05)

      # pdf, log_pdf, cdf, etc...
      # The mean of the lognormal is around 148.
      test_vals = np.linspace(0.1, 1000., num=20).astype(np.float32)
      for func in [
          [log_normal.log_prob, sp_dist.logpdf],
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
      log_normal = ds.TransformedDistribution(
          distribution=ds.Normal(mu=mu, sigma=sigma),
          bijector=bs.Exp(event_ndims=0))

      sample = log_normal.sample(1)
      sample_val, log_pdf_val = sess.run([sample, log_normal.log_pdf(sample)])
      self.assertAllClose(
          stats.lognorm.logpdf(sample_val, s=sigma,
                               scale=np.exp(mu)),
          log_pdf_val,
          atol=1e-2)

  def testConditioning(self):
    with self.test_session():
      conditional_normal = ds.TransformedDistribution(
          distribution=ds.Normal(mu=0., sigma=1.),
          bijector=_ChooseLocation(loc=[-100., 100.]))
      z = [-1, +1, -1, -1, +1]
      self.assertAllClose(
          np.sign(conditional_normal.sample(
              5, bijector_kwargs={"z": z}).eval()), z)

  def testShapeChangingBijector(self):
    with self.test_session():
      softmax = bs.SoftmaxCentered()
      standard_normal = ds.Normal(mu=0., sigma=1.)
      multi_logit_normal = ds.TransformedDistribution(
          distribution=standard_normal,
          bijector=softmax)
      x = [[-np.log(3.), 0.],
           [np.log(3), np.log(5)]]
      y = softmax.forward(x).eval()
      expected_log_pdf = (stats.norm(loc=0., scale=1.).logpdf(x) -
                          np.sum(np.log(y), axis=-1))
      self.assertAllClose(expected_log_pdf,
                          multi_logit_normal.log_prob(y).eval())
      self.assertAllClose([1, 2, 3, 2],
                          tf.shape(multi_logit_normal.sample([1, 2, 3])).eval())
      self.assertAllEqual([2], multi_logit_normal.get_event_shape())
      self.assertAllEqual([2], multi_logit_normal.event_shape().eval())

  def testEntropy(self):
    with self.test_session():
      shift = np.array([[-1, 0, 1],
                        [-1, -2, -3]], dtype=np.float32)
      diag = np.array([[1, 2, 3],
                       [2, 3, 2]], dtype=np.float32)
      actual_mvn = ds.MultivariateNormalDiag(
          shift, diag, validate_args=True)
      fake_mvn = ds.TransformedDistribution(
          ds.MultivariateNormalDiag(
              tf.zeros_like(shift),
              tf.ones_like(diag),
              validate_args=True),
          bs.AffineLinearOperator(
              shift,
              scale=la.LinearOperatorDiag(diag, is_non_singular=True),
              validate_args=True),
          validate_args=True)
      self.assertAllClose(actual_mvn.entropy().eval(),
                          fake_mvn.entropy().eval())


class ScalarToMultiTest(tf.test.TestCase):

  def setUp(self):
    self._shift = np.array([-1, 0, 1], dtype=np.float32)
    self._tril = np.array(
        [[[-1., 0, 0],
          [2, 1, 0],
          [3, 2, 1]],
         [[2, 0, 0],
          [3, -2, 0],
          [4, 3, 2]]], dtype=np.float32)

  def _testMVN(self, base_distribution, batch_shape=None,
               event_shape=None, not_implemented_message=None):
    with self.test_session() as sess:
      # Overriding shapes must be compatible w/bijector; most bijectors are
      # batch_shape agnostic and only care about event_ndims.
      # In the case of `Affine`, if we got it wrong then it would fire an
      # exception due to incompatible dimensions.
      fake_mvn = ds.TransformedDistribution(
          distribution=base_distribution[0](validate_args=True,
                                            **base_distribution[1]),
          bijector=bs.Affine(shift=self._shift, scale_tril=self._tril),
          batch_shape=batch_shape,
          event_shape=event_shape,
          validate_args=True)

      actual_mean = np.tile(self._shift, [2, 1])  # Affine elided this tile.
      actual_cov = np.matmul(self._tril, np.transpose(self._tril, [0, 2, 1]))
      actual_mvn = ds.MultivariateNormalFull(mu=actual_mean, sigma=actual_cov)

      # Ensure sample works by checking first, second moments.
      n = 5e3
      y = fake_mvn.sample(int(n), seed=0)
      sample_mean = tf.reduce_mean(y, 0)
      centered_y = tf.transpose(y - sample_mean, [1, 2, 0])
      sample_cov = tf.matmul(centered_y, centered_y, transpose_b=True) / n
      [sample_mean_, sample_cov_] = sess.run([sample_mean, sample_cov])
      self.assertAllClose(actual_mean, sample_mean_, atol=0.1, rtol=0.1)
      self.assertAllClose(actual_cov, sample_cov_, atol=0., rtol=0.1)

      # Ensure all other functions work as intended.
      x = fake_mvn.sample(5, seed=0).eval()
      self.assertAllEqual([5, 2, 3], x.shape)
      self.assertAllEqual(actual_mvn.get_event_shape(),
                          fake_mvn.get_event_shape())
      self.assertAllEqual(actual_mvn.event_shape().eval(),
                          fake_mvn.event_shape().eval())
      self.assertAllEqual(actual_mvn.get_batch_shape(),
                          fake_mvn.get_batch_shape())
      self.assertAllEqual(actual_mvn.batch_shape().eval(),
                          fake_mvn.batch_shape().eval())
      self.assertAllClose(actual_mvn.log_prob(x).eval(),
                          fake_mvn.log_prob(x).eval(),
                          atol=0., rtol=1e-7)
      self.assertAllClose(actual_mvn.prob(x).eval(),
                          fake_mvn.prob(x).eval(),
                          atol=0., rtol=1e-6)
      self.assertAllClose(actual_mvn.entropy().eval(),
                          fake_mvn.entropy().eval(),
                          atol=0., rtol=1e-6)
      for unsupported_fn in (fake_mvn.log_cdf,
                             fake_mvn.cdf,
                             fake_mvn.survival_function,
                             fake_mvn.log_survival_function):
        with self.assertRaisesRegexp(
            NotImplementedError, not_implemented_message):
          self.assertRaisesRegexp(unsupported_fn(x))

  def testScalarBatchScalarEvent(self):
    self._testMVN(
        base_distribution=[ds.Normal, {"mu": 0., "sigma": 1.}],
        batch_shape=[2],
        event_shape=[3],
        not_implemented_message="not implemented when overriding event_shape")

  def testScalarBatchNonScalarEvent(self):
    self._testMVN(
        base_distribution=[ds.MultivariateNormalDiag, {
            "mu": [0., 0., 0.], "diag_stdev": [1., 1, 1]}],
        batch_shape=[2],
        not_implemented_message="not implemented$")

    with self.test_session():
      # Can't override event_shape for scalar batch, non-scalar event.
      with self.assertRaisesRegexp(ValueError, "requires scalar"):
        ds.TransformedDistribution(
            distribution=ds.MultivariateNormalDiag(mu=[0.], diag_stdev=[1.]),
            bijector=bs.Affine(shift=self._shift, scale_tril=self._tril),
            batch_shape=[2],
            event_shape=[3],
            validate_args=True)

  def testNonScalarBatchScalarEvent(self):
    self._testMVN(
        base_distribution=[ds.Normal, {"mu": [0., 0], "sigma": [1., 1]}],
        event_shape=[3],
        not_implemented_message="not implemented when overriding event_shape")

    with self.test_session():
      # Can't override batch_shape for non-scalar batch, scalar event.
      with self.assertRaisesRegexp(ValueError, "requires scalar"):
        ds.TransformedDistribution(
            distribution=ds.Normal(mu=[0.], sigma=[1.]),
            bijector=bs.Affine(shift=self._shift, scale_tril=self._tril),
            batch_shape=[2],
            event_shape=[3],
            validate_args=True)

  def testNonScalarBatchNonScalarEvent(self):
    with self.test_session():
      # Can't override event_shape and/or batch_shape for non_scalar batch,
      # non-scalar event.
      with self.assertRaisesRegexp(ValueError, "requires scalar"):
        ds.TransformedDistribution(
            distribution=ds.MultivariateNormalDiag(mu=[[0.]],
                                                   diag_stdev=[[1.]]),
            bijector=bs.Affine(shift=self._shift, scale_tril=self._tril),
            batch_shape=[2],
            event_shape=[3],
            validate_args=True)


if __name__ == "__main__":
  tf.test.main()
