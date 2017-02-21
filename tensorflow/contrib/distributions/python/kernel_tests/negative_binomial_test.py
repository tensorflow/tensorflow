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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats
from tensorflow.contrib.distributions.python.ops import negative_binomial
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


# In all tests that follow, we use scipy.stats.nbinom, which
# represents a Negative Binomial distribution, with success and failure
# probabilities flipped.
class NegativeBinomialTest(test.TestCase):

  def testNegativeBinomialShape(self):
    with self.test_session():
      probs = [.1] * 5
      total_count = [2.0] * 5
      negbinom = negative_binomial.NegativeBinomial(
          total_count=total_count, probs=probs)

      self.assertEqual([5], negbinom.batch_shape_tensor().eval())
      self.assertEqual(tensor_shape.TensorShape([5]), negbinom.batch_shape)
      self.assertAllEqual([], negbinom.event_shape_tensor().eval())
      self.assertEqual(tensor_shape.TensorShape([]), negbinom.event_shape)

  def testNegativeBinomialShapeBroadcast(self):
    with self.test_session():
      probs = [[.1, .2, .3]] * 5
      total_count = [[2.]] * 5
      negbinom = negative_binomial.NegativeBinomial(
          total_count=total_count, probs=probs)

      self.assertAllEqual([5, 3], negbinom.batch_shape_tensor().eval())
      self.assertAllEqual(
          tensor_shape.TensorShape([5, 3]), negbinom.batch_shape)
      self.assertAllEqual([], negbinom.event_shape_tensor().eval())
      self.assertAllEqual(tensor_shape.TensorShape([]), negbinom.event_shape)

  def testLogits(self):
    logits = [[0., 9., -0.5]]
    with self.test_session():
      negbinom = negative_binomial.NegativeBinomial(
          total_count=3., logits=logits)
      self.assertEqual([1, 3], negbinom.probs.get_shape())
      self.assertEqual([1, 3], negbinom.logits.get_shape())
      self.assertAllClose(logits, negbinom.logits.eval())

  def testInvalidP(self):
    invalid_ps = [-.01, 0., -2.,]
    with self.test_session():
      with self.assertRaisesOpError("Condition x >= 0"):
        negbinom = negative_binomial.NegativeBinomial(
            5., probs=invalid_ps, validate_args=True)
        negbinom.probs.eval()

    invalid_ps = [1.01, 2., 1.001,]
    with self.test_session():
      with self.assertRaisesOpError("probs has components greater than 1."):
        negbinom = negative_binomial.NegativeBinomial(
            5., probs=invalid_ps, validate_args=True)
        negbinom.probs.eval()

  def testInvalidNegativeCount(self):
    invalid_rs = [-.01, 0., -2.,]
    with self.test_session():
      with self.assertRaisesOpError("Condition x > 0"):
        negbinom = negative_binomial.NegativeBinomial(
            total_count=invalid_rs, probs=0.1, validate_args=True)
        negbinom.total_count.eval()

  def testNegativeBinomialLogCdf(self):
    with self.test_session():
      batch_size = 6
      probs = [.2] * batch_size
      probs_v = .2
      total_count = 5.
      x = np.array([2., 3., 4., 5., 6., 7.], dtype=np.float32)
      negbinom = negative_binomial.NegativeBinomial(
          total_count=total_count, probs=probs)
      expected_log_cdf = stats.nbinom.logcdf(x, n=total_count, p=1 - probs_v)
      log_cdf = negbinom.log_cdf(x)
      self.assertEqual([6], log_cdf.get_shape())
      self.assertAllClose(expected_log_cdf, log_cdf.eval())

      cdf = negbinom.cdf(x)
      self.assertEqual([6], cdf.get_shape())
      self.assertAllClose(np.exp(expected_log_cdf), cdf.eval())

  def testNegativeBinomialLogCdfValidateArgs(self):
    with self.test_session():
      batch_size = 6
      probs = [.9] * batch_size
      total_count = 5.
      with self.assertRaisesOpError("Condition x >= 0"):
        negbinom = negative_binomial.NegativeBinomial(
            total_count=total_count, probs=probs, validate_args=True)
        negbinom.log_cdf(-1.).eval()

  def testNegativeBinomialLogPmf(self):
    with self.test_session():
      batch_size = 6
      probs = [.2] * batch_size
      probs_v = .2
      total_count = 5.
      x = np.array([2., 3., 4., 5., 6., 7.], dtype=np.float32)
      negbinom = negative_binomial.NegativeBinomial(
          total_count=total_count, probs=probs)
      expected_log_pmf = stats.nbinom.logpmf(x, n=total_count, p=1 - probs_v)
      log_pmf = negbinom.log_prob(x)
      self.assertEqual([6], log_pmf.get_shape())
      self.assertAllClose(expected_log_pmf, log_pmf.eval())

      pmf = negbinom.prob(x)
      self.assertEqual([6], pmf.get_shape())
      self.assertAllClose(np.exp(expected_log_pmf), pmf.eval())

  def testNegativeBinomialLogPmfValidateArgs(self):
    with self.test_session():
      batch_size = 6
      probs = [.9] * batch_size
      total_count = 5.
      x = np.array([2.5, 3.2, 4.3, 5.1, 6., 7.], dtype=np.float32)
      negbinom = negative_binomial.NegativeBinomial(
          total_count=total_count, probs=probs, validate_args=True)

      with self.assertRaisesOpError("Condition x == y"):
        log_pmf = negbinom.log_prob(x)
        log_pmf.eval()

      with self.assertRaisesOpError("Condition x >= 0"):
        log_pmf = negbinom.log_prob([-1.])
        log_pmf.eval()

      negbinom = negative_binomial.NegativeBinomial(
          total_count=total_count, probs=probs, validate_args=False)
      log_pmf = negbinom.log_prob(x)
      self.assertEqual([6], log_pmf.get_shape())
      pmf = negbinom.prob(x)
      self.assertEqual([6], pmf.get_shape())

  def testNegativeBinomialLogPmfMultidimensional(self):
    with self.test_session():
      batch_size = 6
      probs = constant_op.constant([[.2, .3, .5]] * batch_size)
      probs_v = np.array([.2, .3, .5])
      total_count = 5.
      x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=np.float32).T
      negbinom = negative_binomial.NegativeBinomial(
          total_count=total_count, probs=probs)
      expected_log_pmf = stats.nbinom.logpmf(
          x, n=total_count, p=1 - probs_v)
      log_pmf = negbinom.log_prob(x)
      log_pmf_values = log_pmf.eval()
      self.assertEqual([6, 3], log_pmf.get_shape())
      self.assertAllClose(expected_log_pmf, log_pmf_values)

      pmf = negbinom.prob(x)
      pmf_values = pmf.eval()
      self.assertEqual([6, 3], pmf.get_shape())
      self.assertAllClose(np.exp(expected_log_pmf), pmf_values)

  def testNegativeBinomialMean(self):
    with self.test_session():
      total_count = 5.
      probs = np.array([.1, .3, .25], dtype=np.float32)
      negbinom = negative_binomial.NegativeBinomial(
          total_count=total_count, probs=probs)
      expected_means = stats.nbinom.mean(n=total_count, p=1 - probs)
      self.assertEqual([3], negbinom.mean().get_shape())
      self.assertAllClose(expected_means, negbinom.mean().eval())

  def testNegativeBinomialVariance(self):
    with self.test_session():
      total_count = 5.
      probs = np.array([.1, .3, .25], dtype=np.float32)
      negbinom = negative_binomial.NegativeBinomial(
          total_count=total_count, probs=probs)
      expected_vars = stats.nbinom.var(n=total_count, p=1 - probs)
      self.assertEqual([3], negbinom.variance().get_shape())
      self.assertAllClose(expected_vars, negbinom.variance().eval())

  def testNegativeBinomialStddev(self):
    with self.test_session():
      total_count = 5.
      probs = np.array([.1, .3, .25], dtype=np.float32)
      negbinom = negative_binomial.NegativeBinomial(
          total_count=total_count, probs=probs)
      expected_stds = stats.nbinom.std(n=total_count, p=1 - probs)
      self.assertEqual([3], negbinom.stddev().get_shape())
      self.assertAllClose(expected_stds, negbinom.stddev().eval())

  def testNegativeBinomialSample(self):
    with self.test_session() as sess:
      probs = [.3, .9]
      total_count = [4., 11.]
      n = int(100e3)
      negbinom = negative_binomial.NegativeBinomial(
          total_count=total_count, probs=probs)

      samples = negbinom.sample(n, seed=12345)
      self.assertEqual([n, 2], samples.get_shape())

      sample_mean = math_ops.reduce_mean(samples, axis=0)
      sample_var = math_ops.reduce_mean(
          (samples - sample_mean[array_ops.newaxis, ...])**2., axis=0)
      sample_min = math_ops.reduce_min(samples)
      [sample_mean_, sample_var_, sample_min_] = sess.run([
          sample_mean, sample_var, sample_min])
      self.assertAllEqual(np.ones(sample_min_.shape, dtype=np.bool),
                          sample_min_ >= 0.0)
      for i in range(2):
        self.assertAllClose(sample_mean_[i],
                            stats.nbinom.mean(total_count[i], 1 - probs[i]),
                            atol=0.,
                            rtol=.02)
        self.assertAllClose(sample_var_[i],
                            stats.nbinom.var(total_count[i], 1 - probs[i]),
                            atol=0.,
                            rtol=.02)


if __name__ == "__main__":
  test.main()
