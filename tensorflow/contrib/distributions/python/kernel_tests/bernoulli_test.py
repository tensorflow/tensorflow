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
"""Tests for the Bernoulli distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.special
import tensorflow as tf


def make_bernoulli(batch_shape, dtype=tf.int32):
  p = np.random.uniform(size=list(batch_shape))
  p = tf.constant(p, dtype=tf.float32)
  return tf.contrib.distributions.Bernoulli(p=p, dtype=dtype)


def entropy(p):
  q = 1. - p
  return -q * np.log(q) - p * np.log(p)


class BernoulliTest(tf.test.TestCase):

  def testP(self):
    p = [0.2, 0.4]
    dist = tf.contrib.distributions.Bernoulli(p=p)
    with self.test_session():
      self.assertAllClose(p, dist.p.eval())

  def testLogits(self):
    logits = [-42., 42.]
    dist = tf.contrib.distributions.Bernoulli(logits=logits)
    with self.test_session():
      self.assertAllClose(logits, dist.logits.eval())

    with self.test_session():
      self.assertAllClose(scipy.special.expit(logits), dist.p.eval())

    p = [0.01, 0.99, 0.42]
    dist = tf.contrib.distributions.Bernoulli(p=p)
    with self.test_session():
      self.assertAllClose(scipy.special.logit(p), dist.logits.eval())

  def testInvalidP(self):
    invalid_ps = [1.01, 2.]
    for p in invalid_ps:
      with self.test_session():
        with self.assertRaisesOpError("p has components greater than 1"):
          dist = tf.contrib.distributions.Bernoulli(p=p, validate_args=True)
          dist.p.eval()

    invalid_ps = [-0.01, -3.]
    for p in invalid_ps:
      with self.test_session():
        with self.assertRaisesOpError("Condition x >= 0"):
          dist = tf.contrib.distributions.Bernoulli(p=p, validate_args=True)
          dist.p.eval()

    valid_ps = [0.0, 0.5, 1.0]
    for p in valid_ps:
      with self.test_session():
        dist = tf.contrib.distributions.Bernoulli(p=p)
        self.assertEqual(p, dist.p.eval())  # Should not fail

  def testShapes(self):
    with self.test_session():
      for batch_shape in ([], [1], [2, 3, 4]):
        dist = make_bernoulli(batch_shape)
        self.assertAllEqual(batch_shape, dist.get_batch_shape().as_list())
        self.assertAllEqual(batch_shape, dist.batch_shape().eval())
        self.assertAllEqual([], dist.get_event_shape().as_list())
        self.assertAllEqual([], dist.event_shape().eval())

  def testDtype(self):
    dist = make_bernoulli([])
    self.assertEqual(dist.dtype, tf.int32)
    self.assertEqual(dist.dtype, dist.sample(5).dtype)
    self.assertEqual(dist.dtype, dist.mode().dtype)
    self.assertEqual(dist.p.dtype, dist.mean().dtype)
    self.assertEqual(dist.p.dtype, dist.variance().dtype)
    self.assertEqual(dist.p.dtype, dist.std().dtype)
    self.assertEqual(dist.p.dtype, dist.entropy().dtype)
    self.assertEqual(dist.p.dtype, dist.pmf(0).dtype)
    self.assertEqual(dist.p.dtype, dist.log_pmf(0).dtype)

    dist64 = make_bernoulli([], tf.int64)
    self.assertEqual(dist64.dtype, tf.int64)
    self.assertEqual(dist64.dtype, dist64.sample(5).dtype)
    self.assertEqual(dist64.dtype, dist64.mode().dtype)

  def _testPmf(self, **kwargs):
    dist = tf.contrib.distributions.Bernoulli(**kwargs)
    with self.test_session():
      # pylint: disable=bad-continuation
      xs = [
          0,
          [1],
          [1, 0],
          [[1, 0]],
          [[1, 0], [1, 1]],
      ]
      expected_pmfs = [
          [[0.8, 0.6], [0.7, 0.4]],
          [[0.2, 0.4], [0.3, 0.6]],
          [[0.2, 0.6], [0.3, 0.4]],
          [[0.2, 0.6], [0.3, 0.4]],
          [[0.2, 0.6], [0.3, 0.6]],
      ]
      # pylint: enable=bad-continuation

      for x, expected_pmf in zip(xs, expected_pmfs):
        self.assertAllClose(dist.pmf(x).eval(), expected_pmf)
        self.assertAllClose(dist.log_pmf(x).eval(), np.log(expected_pmf))

  def testPmfCorrectBroadcastDynamicShape(self):
    with self.test_session():
      p = tf.placeholder(dtype=tf.float32)
      dist = tf.contrib.distributions.Bernoulli(p=p)
      event1 = [1, 0, 1]
      event2 = [[1, 0, 1]]
      self.assertAllClose(dist.pmf(event1).eval({p: [0.2, 0.3, 0.4]}),
                          [0.2, 0.7, 0.4])
      self.assertAllClose(dist.pmf(event2).eval({p: [0.2, 0.3, 0.4]}),
                          [[0.2, 0.7, 0.4]])

  def testPmfWithP(self):
    p = [[0.2, 0.4], [0.3, 0.6]]
    self._testPmf(p=p)
    self._testPmf(logits=scipy.special.logit(p))

  def testBroadcasting(self):
    with self.test_session():
      p = tf.placeholder(tf.float32)
      dist = tf.contrib.distributions.Bernoulli(p=p)
      self.assertAllClose(np.log(0.5), dist.log_pmf(1).eval({p: 0.5}))
      self.assertAllClose(np.log([0.5, 0.5, 0.5]),
                          dist.log_pmf([1, 1, 1]).eval({p: 0.5}))
      self.assertAllClose(np.log([0.5, 0.5, 0.5]),
                          dist.log_pmf(1).eval({p: [0.5, 0.5, 0.5]}))

  def testPmfShapes(self):
    with self.test_session():
      p = tf.placeholder(tf.float32, shape=[None, 1])
      dist = tf.contrib.distributions.Bernoulli(p=p)
      self.assertEqual(2, len(dist.log_pmf(1).eval({p: [[0.5], [0.5]]}).shape))

    with self.test_session():
      dist = tf.contrib.distributions.Bernoulli(p=0.5)
      self.assertEqual(2, len(dist.log_pmf([[1], [1]]).eval().shape))

    with self.test_session():
      dist = tf.contrib.distributions.Bernoulli(p=0.5)
      self.assertEqual((), dist.log_pmf(1).get_shape())
      self.assertEqual((1), dist.log_pmf([1]).get_shape())
      self.assertEqual((2, 1), dist.log_pmf([[1], [1]]).get_shape())

    with self.test_session():
      dist = tf.contrib.distributions.Bernoulli(p=[[0.5], [0.5]])
      self.assertEqual((2, 1), dist.log_pmf(1).get_shape())

  def testBoundaryConditions(self):
    with self.test_session():
      dist = tf.contrib.distributions.Bernoulli(p=1.0)
      self.assertAllClose(np.nan, dist.log_pmf(0).eval())
      self.assertAllClose([np.nan], [dist.log_pmf(1).eval()])

  def testEntropyNoBatch(self):
    p = 0.2
    dist = tf.contrib.distributions.Bernoulli(p=p)
    with self.test_session():
      self.assertAllClose(dist.entropy().eval(), entropy(p))

  def testEntropyWithBatch(self):
    p = [[0.1, 0.7], [0.2, 0.6]]
    dist = tf.contrib.distributions.Bernoulli(p=p, validate_args=False)
    with self.test_session():
      self.assertAllClose(dist.entropy().eval(), [[entropy(0.1), entropy(0.7)],
                                                  [entropy(0.2), entropy(0.6)]])

  def testSampleN(self):
    with self.test_session():
      p = [0.2, 0.6]
      dist = tf.contrib.distributions.Bernoulli(p=p)
      n = 100000
      samples = dist.sample(n)
      samples.set_shape([n, 2])
      self.assertEqual(samples.dtype, tf.int32)
      sample_values = samples.eval()
      self.assertTrue(np.all(sample_values >= 0))
      self.assertTrue(np.all(sample_values <= 1))
      # Note that the standard error for the sample mean is ~ sqrt(p * (1 - p) /
      # n). This means that the tolerance is very sensitive to the value of p
      # as well as n.
      self.assertAllClose(p, np.mean(sample_values, axis=0), atol=1e-2)
      self.assertEqual(set([0, 1]), set(sample_values.flatten()))
      # In this test we're just interested in verifying there isn't a crash
      # owing to mismatched types. b/30940152
      dist = tf.contrib.distributions.Bernoulli(np.log([.2, .4]))
      self.assertAllEqual(
          (1, 2), dist.sample(1, seed=42).get_shape().as_list())

  def testSampleActsLikeSampleN(self):
    with self.test_session() as sess:
      p = [0.2, 0.6]
      dist = tf.contrib.distributions.Bernoulli(p=p)
      n = 1000
      seed = 42
      self.assertAllEqual(dist.sample(n, seed).eval(),
                          dist.sample(n, seed).eval())
      n = tf.placeholder(tf.int32)
      sample, sample = sess.run([dist.sample(n, seed),
                                 dist.sample(n, seed)],
                                feed_dict={n: 1000})
      self.assertAllEqual(sample, sample)

  def testMean(self):
    with self.test_session():
      p = np.array([[0.2, 0.7], [0.5, 0.4]], dtype=np.float32)
      dist = tf.contrib.distributions.Bernoulli(p=p)
      self.assertAllEqual(dist.mean().eval(), p)

  def testVarianceAndStd(self):
    var = lambda p: p * (1. - p)
    with self.test_session():
      p = [[0.2, 0.7], [0.5, 0.4]]
      dist = tf.contrib.distributions.Bernoulli(p=p)
      self.assertAllClose(dist.variance().eval(),
                          np.array([[var(0.2), var(0.7)], [var(0.5), var(0.4)]],
                                   dtype=np.float32))
      self.assertAllClose(dist.std().eval(),
                          np.array([[np.sqrt(var(0.2)), np.sqrt(var(0.7))],
                                    [np.sqrt(var(0.5)), np.sqrt(var(0.4))]],
                                   dtype=np.float32))

  def testBernoulliWithSigmoidP(self):
    p = np.array([8.3, 4.2])
    dist = tf.contrib.distributions.BernoulliWithSigmoidP(p=p)
    with self.test_session():
      self.assertAllClose(tf.nn.sigmoid(p).eval(), dist.p.eval())

  def testBernoulliBernoulliKL(self):
    with self.test_session() as sess:
      batch_size = 6
      a_p = np.array([0.5] * batch_size, dtype=np.float32)
      b_p = np.array([0.4] * batch_size, dtype=np.float32)

      a = tf.contrib.distributions.Bernoulli(p=a_p)
      b = tf.contrib.distributions.Bernoulli(p=b_p)

      kl = tf.contrib.distributions.kl(a, b)
      kl_val = sess.run(kl)

      kl_expected = (
          a_p * np.log(a_p / b_p) +
          (1. - a_p) * np.log((1. - a_p) / (1. - b_p)))

      self.assertEqual(kl.get_shape(), (batch_size,))
      self.assertAllClose(kl_val, kl_expected)


if __name__ == "__main__":
  tf.test.main()
