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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats
import tensorflow as tf


class BinomialTest(tf.test.TestCase):

  def testSimpleShapes(self):
    with self.test_session():
      p = np.float32(np.random.beta(1, 1))
      binom = tf.contrib.distributions.Binomial(n=1., p=p)
      self.assertAllEqual([], binom.event_shape().eval())
      self.assertAllEqual([], binom.batch_shape().eval())
      self.assertEqual(tf.TensorShape([]), binom.get_event_shape())
      self.assertEqual(tf.TensorShape([]), binom.get_batch_shape())

  def testComplexShapes(self):
    with self.test_session():
      p = np.random.beta(1, 1, size=(3, 2)).astype(np.float32)
      n = [[3., 2], [4, 5], [6, 7]]
      binom = tf.contrib.distributions.Binomial(n=n, p=p)
      self.assertAllEqual([], binom.event_shape().eval())
      self.assertAllEqual([3, 2], binom.batch_shape().eval())
      self.assertEqual(tf.TensorShape([]), binom.get_event_shape())
      self.assertEqual(tf.TensorShape([3, 2]), binom.get_batch_shape())

  def testNProperty(self):
    p = [[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]]
    n = [[3.], [4]]
    with self.test_session():
      binom = tf.contrib.distributions.Binomial(n=n, p=p)
      self.assertEqual((2, 1), binom.n.get_shape())
      self.assertAllClose(n, binom.n.eval())

  def testPProperty(self):
    p = [[0.1, 0.2, 0.7]]
    with self.test_session():
      binom = tf.contrib.distributions.Binomial(n=3., p=p)
      self.assertEqual((1, 3), binom.p.get_shape())
      self.assertEqual((1, 3), binom.logits.get_shape())
      self.assertAllClose(p, binom.p.eval())

  def testLogitsProperty(self):
    logits = [[0., 9., -0.5]]
    with self.test_session():
      binom = tf.contrib.distributions.Binomial(n=3., logits=logits)
      self.assertEqual((1, 3), binom.p.get_shape())
      self.assertEqual((1, 3), binom.logits.get_shape())
      self.assertAllClose(logits, binom.logits.eval())

  def testPmfNandCountsAgree(self):
    p = [[0.1, 0.2, 0.7]]
    n = [[5.]]
    with self.test_session():
      binom = tf.contrib.distributions.Binomial(n=n, p=p, validate_args=True)
      binom.pmf([2., 3, 2]).eval()
      binom.pmf([3., 1, 2]).eval()
      with self.assertRaisesOpError("Condition x >= 0.*"):
        binom.pmf([-1., 4, 2]).eval()
      with self.assertRaisesOpError("Condition x <= y.*"):
        binom.pmf([7., 3, 0]).eval()

  def testPmf_non_integer_counts(self):
    p = [[0.1, 0.2, 0.7]]
    n = [[5.]]
    with self.test_session():
      # No errors with integer n.
      binom = tf.contrib.distributions.Binomial(n=n, p=p, validate_args=True)
      binom.pmf([2., 3, 2]).eval()
      binom.pmf([3., 1, 2]).eval()
      # Both equality and integer checking fail.
      with self.assertRaisesOpError("Condition x == y.*"):
        binom.pmf([1.0, 2.5, 1.5]).eval()

      binom = tf.contrib.distributions.Binomial(n=n, p=p, validate_args=False)
      binom.pmf([1., 2., 3.]).eval()
      # Non-integer arguments work.
      binom.pmf([1.0, 2.5, 1.5]).eval()

  def testPmfBothZeroBatches(self):
    with self.test_session():
      # Both zero-batches.  No broadcast
      p = 0.5
      counts = 1.
      pmf = tf.contrib.distributions.Binomial(n=1., p=p).pmf(counts)
      self.assertAllClose(0.5, pmf.eval())
      self.assertEqual((), pmf.get_shape())

  def testPmfBothZeroBatchesNontrivialN(self):
    with self.test_session():
      # Both zero-batches.  No broadcast
      p = 0.1
      counts = 3.
      binom = tf.contrib.distributions.Binomial(n=5., p=p)
      pmf = binom.pmf(counts)
      self.assertAllClose(stats.binom.pmf(counts, n=5., p=p), pmf.eval())
      self.assertEqual((), pmf.get_shape())

  def testPmfPStretchedInBroadcastWhenSameRank(self):
    with self.test_session():
      p = [[0.1, 0.9]]
      counts = [[1., 2.]]
      pmf = tf.contrib.distributions.Binomial(n=3., p=p).pmf(counts)
      self.assertAllClose(stats.binom.pmf(counts, n=3., p=p), pmf.eval())
      self.assertEqual((1, 2), pmf.get_shape())

  def testPmfPStretchedInBroadcastWhenLowerRank(self):
    with self.test_session():
      p = [0.1, 0.4]
      counts = [[1.], [0.]]
      pmf = tf.contrib.distributions.Binomial(n=1., p=p).pmf(counts)
      self.assertAllClose([[0.1, 0.4], [0.9, 0.6]], pmf.eval())
      self.assertEqual((2, 2), pmf.get_shape())

  def testBinomialMean(self):
    with self.test_session():
      n = 5.
      p = [0.1, 0.2, 0.7]
      binom = tf.contrib.distributions.Binomial(n=n, p=p)
      expected_means = stats.binom.mean(n, p)
      self.assertEqual((3,), binom.mean().get_shape())
      self.assertAllClose(expected_means, binom.mean().eval())

  def testBinomialVariance(self):
    with self.test_session():
      n = 5.
      p = [0.1, 0.2, 0.7]
      binom = tf.contrib.distributions.Binomial(n=n, p=p)
      expected_variances = stats.binom.var(n, p)
      self.assertEqual((3,), binom.variance().get_shape())
      self.assertAllClose(expected_variances, binom.variance().eval())

  def testBinomialMode(self):
    with self.test_session():
      n = 5.
      p = [0.1, 0.2, 0.7]
      binom = tf.contrib.distributions.Binomial(n=n, p=p)
      expected_modes = [0., 1, 4]
      self.assertEqual((3,), binom.mode().get_shape())
      self.assertAllClose(expected_modes, binom.mode().eval())

  def testBinomialMultipleMode(self):
    with self.test_session():
      n = 9.
      p = [0.1, 0.2, 0.7]
      binom = tf.contrib.distributions.Binomial(n=n, p=p)
      # For the case where (n + 1) * p is an integer, the modes are:
      # (n + 1) * p and (n + 1) * p - 1. In this case, we get back
      # the larger of the two modes.
      expected_modes = [1., 2, 7]
      self.assertEqual((3,), binom.mode().get_shape())
      self.assertAllClose(expected_modes, binom.mode().eval())

if __name__ == "__main__":
  tf.test.main()
