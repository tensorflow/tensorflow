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


class BetaTest(tf.test.TestCase):

  def testSimpleShapes(self):
    with self.test_session():
      a = np.random.rand(3)
      b = np.random.rand(3)
      dist = tf.contrib.distributions.Beta(a, b)
      self.assertAllEqual([], dist.event_shape().eval())
      self.assertAllEqual([3], dist.batch_shape().eval())
      self.assertEqual(tf.TensorShape([]), dist.get_event_shape())
      self.assertEqual(tf.TensorShape([3]), dist.get_batch_shape())

  def testComplexShapes(self):
    with self.test_session():
      a = np.random.rand(3, 2, 2)
      b = np.random.rand(3, 2, 2)
      dist = tf.contrib.distributions.Beta(a, b)
      self.assertAllEqual([], dist.event_shape().eval())
      self.assertAllEqual([3, 2, 2], dist.batch_shape().eval())
      self.assertEqual(tf.TensorShape([]), dist.get_event_shape())
      self.assertEqual(tf.TensorShape([3, 2, 2]), dist.get_batch_shape())

  def testComplexShapes_broadcast(self):
    with self.test_session():
      a = np.random.rand(3, 2, 2)
      b = np.random.rand(2, 2)
      dist = tf.contrib.distributions.Beta(a, b)
      self.assertAllEqual([], dist.event_shape().eval())
      self.assertAllEqual([3, 2, 2], dist.batch_shape().eval())
      self.assertEqual(tf.TensorShape([]), dist.get_event_shape())
      self.assertEqual(tf.TensorShape([3, 2, 2]), dist.get_batch_shape())

  def testAlphaProperty(self):
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    with self.test_session():
      dist = tf.contrib.distributions.Beta(a, b)
      self.assertEqual([1, 3], dist.a.get_shape())
      self.assertAllClose(a, dist.a.eval())

  def testBetaProperty(self):
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    with self.test_session():
      dist = tf.contrib.distributions.Beta(a, b)
      self.assertEqual([1, 3], dist.b.get_shape())
      self.assertAllClose(b, dist.b.eval())

  def testPdfXProper(self):
    a = [[1., 2, 3]]
    b = [[2., 4, 3]]
    with self.test_session():
      dist = tf.contrib.distributions.Beta(a, b)
      dist.pdf([.1, .3, .6]).eval()
      dist.pdf([.2, .3, .5]).eval()
      # Either condition can trigger.
      with self.assertRaisesOpError('(Condition x > 0.*|Condition x < y.*)'):
        dist.pdf([-1., 1, 1]).eval()
      with self.assertRaisesOpError('Condition x.*'):
        dist.pdf([0., 1, 1]).eval()
      with self.assertRaisesOpError('Condition x < y.*'):
        dist.pdf([.1, .2, 1.2]).eval()

  def testPdfTwoBatches(self):
    with self.test_session():
      a = [1., 2]
      b = [1., 2]
      x = [.5, .5]
      dist = tf.contrib.distributions.Beta(a, b)
      pdf = dist.pdf(x)
      self.assertAllClose([1., 3./2], pdf.eval())
      self.assertEqual((2,), pdf.get_shape())

  def testPdfTwoBatchesNontrivialX(self):
    with self.test_session():
      a = [1., 2]
      b = [1., 2]
      x = [.3, .7]
      dist = tf.contrib.distributions.Beta(a, b)
      pdf = dist.pdf(x)
      self.assertAllClose([1, 63./50], pdf.eval())
      self.assertEqual((2,), pdf.get_shape())

  def testPdfUniformZeroBatch(self):
    with self.test_session():
      # This is equivalent to a uniform distribution
      a = 1.
      b = 1.
      x = np.array([.1, .2, .3, .5, .8], dtype=np.float32)
      dist = tf.contrib.distributions.Beta(a, b)
      pdf = dist.pdf(x)
      self.assertAllClose([1.] * 5, pdf.eval())
      self.assertEqual((5,), pdf.get_shape())

  def testPdfAlphaStretchedInBroadcastWhenSameRank(self):
    with self.test_session():
      a = [[1., 2]]
      b = [[1., 2]]
      x = [[.5, .5], [.3, .7]]
      dist = tf.contrib.distributions.Beta(a, b)
      pdf = dist.pdf(x)
      self.assertAllClose([[1., 3./2], [1., 63./50]], pdf.eval())
      self.assertEqual((2, 2), pdf.get_shape())

  def testPdfAlphaStretchedInBroadcastWhenLowerRank(self):
    with self.test_session():
      a = [1., 2]
      b = [1., 2]
      x = [[.5, .5], [.2, .8]]
      pdf = tf.contrib.distributions.Beta(a, b).pdf(x)
      self.assertAllClose([[1., 3./2], [1., 24./25]], pdf.eval())
      self.assertEqual((2, 2), pdf.get_shape())

  def testPdfXStretchedInBroadcastWhenSameRank(self):
    with self.test_session():
      a = [[1., 2], [2., 3]]
      b = [[1., 2], [2., 3]]
      x = [[.5, .5]]
      pdf = tf.contrib.distributions.Beta(a, b).pdf(x)
      self.assertAllClose([[1., 3./2], [3./2, 15./8]], pdf.eval())
      self.assertEqual((2, 2), pdf.get_shape())

  def testPdfXStretchedInBroadcastWhenLowerRank(self):
    with self.test_session():
      a = [[1., 2], [2., 3]]
      b = [[1., 2], [2., 3]]
      x = [.5, .5]
      pdf = tf.contrib.distributions.Beta(a, b).pdf(x)
      self.assertAllClose([[1., 3./2], [3./2, 15./8]], pdf.eval())
      self.assertEqual((2, 2), pdf.get_shape())

  def testBetaMean(self):
    with tf.Session():
      a = [1., 2, 3]
      b = [2., 4, 1.2]
      expected_mean = stats.beta.mean(a, b)
      dist = tf.contrib.distributions.Beta(a, b)
      self.assertEqual(dist.mean().get_shape(), (3,))
      self.assertAllClose(expected_mean, dist.mean().eval())

  def testBetaVariance(self):
    with tf.Session():
      a = [1., 2, 3]
      b = [2., 4, 1.2]
      expected_variance = stats.beta.var(a, b)
      dist = tf.contrib.distributions.Beta(a, b)
      self.assertEqual(dist.variance().get_shape(), (3,))
      self.assertAllClose(expected_variance, dist.variance().eval())

  def testBetaMode(self):
    with tf.Session():
      a = np.array([1.1, 2, 3])
      b = np.array([2., 4, 1.2])
      expected_mode = (a - 1)/(a + b - 2)
      dist = tf.contrib.distributions.Beta(a, b)
      self.assertEqual(dist.mode().get_shape(), (3,))
      self.assertAllClose(expected_mode, dist.mode().eval())

  def testBetaMode_invalid(self):
    with tf.Session():
      a = np.array([1., 2, 3])
      b = np.array([2., 4, 1.2])
      dist = tf.contrib.distributions.Beta(a, b)
      with self.assertRaisesOpError('Condition x < y.*'):
        dist.mode().eval()

      a = np.array([2., 2, 3])
      b = np.array([1., 4, 1.2])
      dist = tf.contrib.distributions.Beta(a, b)
      with self.assertRaisesOpError('Condition x < y.*'):
        dist.mode().eval()

  def testBetaMode_enable_allow_nan_stats(self):
    with tf.Session():
      a = np.array([1., 2, 3])
      b = np.array([2., 4, 1.2])
      dist = tf.contrib.distributions.Beta(a, b, allow_nan_stats=True)

      expected_mode = (a - 1)/(a + b - 2)
      expected_mode[0] = np.nan
      self.assertEqual((3,), dist.mode().get_shape())
      self.assertAllClose(expected_mode, dist.mode().eval())

      a = np.array([2., 2, 3])
      b = np.array([1., 4, 1.2])
      dist = tf.contrib.distributions.Beta(a, b, allow_nan_stats=True)

      expected_mode = (a - 1)/(a + b - 2)
      expected_mode[0] = np.nan
      self.assertEqual((3,), dist.mode().get_shape())
      self.assertAllClose(expected_mode, dist.mode().eval())

  def testBetaEntropy(self):
    with tf.Session():
      a = [1., 2, 3]
      b = [2., 4, 1.2]
      expected_entropy = stats.beta.entropy(a, b)
      dist = tf.contrib.distributions.Beta(a, b)
      self.assertEqual(dist.entropy().get_shape(), (3,))
      self.assertAllClose(expected_entropy, dist.entropy().eval())

  def testBetaSample(self):
    with self.test_session():
      a = 1.
      b = 2.
      beta = tf.contrib.distributions.Beta(a, b)
      n = tf.constant(100000)
      samples = beta.sample_n(n)
      sample_values = samples.eval()
      self.assertEqual(sample_values.shape, (100000,))
      self.assertFalse(np.any(sample_values < 0.0))
      self.assertLess(
          stats.kstest(
              # Beta is a univariate distribution.
              sample_values, stats.beta(a=1., b=2.).cdf)[0],
          0.01)
      # The standard error of the sample mean is 1 / (sqrt(18 * n))
      self.assertAllClose(sample_values.mean(axis=0),
                          stats.beta.mean(a, b),
                          atol=1e-2)
      self.assertAllClose(np.cov(sample_values, rowvar=0),
                          stats.beta.var(a, b),
                          atol=1e-1)

  def testBetaSampleMultidimensional(self):
    with self.test_session():
      # TODO(srvasude): Remove the 1.1 when Gamma sampler doesn't
      # return 0 when a < 1.
      a = np.random.rand(3, 2, 2).astype(np.float32) + 1.1
      b = np.random.rand(3, 2, 2).astype(np.float32) + 1.1
      beta = tf.contrib.distributions.Beta(a, b)
      n = tf.constant(100000)
      samples = beta.sample_n(n)
      sample_values = samples.eval()
      self.assertEqual(sample_values.shape, (100000, 3, 2, 2))
      self.assertFalse(np.any(sample_values < 0.0))
      self.assertAllClose(
          sample_values[:, 1, :].mean(axis=0),
          stats.beta.mean(a, b)[1, :],
          atol=1e-1)

if __name__ == '__main__':
  tf.test.main()
