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


class TransformedDistributionTest(tf.test.TestCase):

  def testTransformedDistribution(self):
    g = tf.Graph()
    with g.as_default():
      mu = 3.0
      sigma = 2.0
      # Note: the Jacobian callable only works for this example; more generally
      # you may or may not need a reduce_sum.
      log_normal = tf.contrib.distributions.TransformedDistribution(
          base_dist_cls=tf.contrib.distributions.Normal,
          mu=mu,
          sigma=sigma,
          transform=lambda x: tf.exp(x),
          inverse=lambda y: tf.log(y),
          log_det_jacobian=(lambda x: x))
      sp_dist = stats.lognorm(s=sigma, scale=np.exp(mu))

      # sample
      sample = log_normal.sample_n(100000, seed=235)
      with self.test_session(graph=g):
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
      log_normal = tf.contrib.distributions.TransformedDistribution(
          base_dist_cls=tf.contrib.distributions.Normal,
          mu=mu,
          sigma=sigma,
          transform=lambda x: tf.exp(x),
          inverse=None,
          log_det_jacobian=(lambda x: tf.reduce_sum(x)))

      sample = log_normal.sample_n(1)
      sample_val, log_pdf_val = sess.run([sample, log_normal.log_pdf(sample)])
      self.assertAllClose(
          stats.lognorm.logpdf(sample_val, s=sigma,
                               scale=np.exp(mu)),
          log_pdf_val,
          atol=1e-2)

      with self.assertRaisesRegexp(ValueError,
                                   "was not returned from `sample`"):
        log_normal.log_pdf(tf.constant(3.0))


if __name__ == "__main__":
  tf.test.main()
