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

bijectors = tf.contrib.distributions.bijector
distributions = tf.contrib.distributions


class _ChooseLocation(bijectors.Bijector):
  """A Bijector which chooses between one of two location parameters."""

  def __init__(self, loc, name="ChooseLocation"):
    self._parameters = {}
    self._name = name
    with self._name_scope("init", values=[loc]):
      self._loc = tf.convert_to_tensor(loc, name="loc")
      super(_ChooseLocation, self).__init__(
          parameters={"loc": self._loc},
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
      log_normal = distributions.TransformedDistribution(
          distribution=distributions.Normal(mu=mu, sigma=sigma),
          bijector=bijectors.Exp(event_ndims=0))
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
      log_normal = distributions.TransformedDistribution(
          distribution=distributions.Normal(mu=mu, sigma=sigma),
          bijector=bijectors.Exp(event_ndims=0))

      sample = log_normal.sample(1)
      sample_val, log_pdf_val = sess.run([sample, log_normal.log_pdf(sample)])
      self.assertAllClose(
          stats.lognorm.logpdf(sample_val, s=sigma,
                               scale=np.exp(mu)),
          log_pdf_val,
          atol=1e-2)

  def testConditioning(self):
    with self.test_session():
      conditional_normal = distributions.TransformedDistribution(
          distribution=distributions.Normal(mu=0., sigma=1.),
          bijector=_ChooseLocation(loc=[-100., 100.]))
      z = [-1, +1, -1, -1, +1]
      self.assertAllClose(
          np.sign(conditional_normal.sample(
              5, bijector_kwargs={"z": z}).eval()), z)

  def testShapeChangingBijector(self):
    with self.test_session():
      softmax = bijectors.SoftmaxCentered()
      standard_normal = distributions.Normal(mu=0., sigma=1.)
      multi_logit_normal = distributions.TransformedDistribution(
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


if __name__ == "__main__":
  tf.test.main()
