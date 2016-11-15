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


class DirichletTest(tf.test.TestCase):

  def testSimpleShapes(self):
    with self.test_session():
      alpha = np.random.rand(3)
      dist = tf.contrib.distributions.Dirichlet(alpha)
      self.assertEqual(3, dist.event_shape().eval())
      self.assertAllEqual([], dist.batch_shape().eval())
      self.assertEqual(tf.TensorShape([3]), dist.get_event_shape())
      self.assertEqual(tf.TensorShape([]), dist.get_batch_shape())

  def testComplexShapes(self):
    with self.test_session():
      alpha = np.random.rand(3, 2, 2)
      dist = tf.contrib.distributions.Dirichlet(alpha)
      self.assertEqual(2, dist.event_shape().eval())
      self.assertAllEqual([3, 2], dist.batch_shape().eval())
      self.assertEqual(tf.TensorShape([2]), dist.get_event_shape())
      self.assertEqual(tf.TensorShape([3, 2]), dist.get_batch_shape())

  def testAlphaProperty(self):
    alpha = [[1., 2, 3]]
    with self.test_session():
      dist = tf.contrib.distributions.Dirichlet(alpha)
      self.assertEqual([1, 3], dist.alpha.get_shape())
      self.assertAllClose(alpha, dist.alpha.eval())

  def testPdfXProper(self):
    alpha = [[1., 2, 3]]
    with self.test_session():
      dist = tf.contrib.distributions.Dirichlet(alpha, validate_args=True)
      dist.pdf([.1, .3, .6]).eval()
      dist.pdf([.2, .3, .5]).eval()
      # Either condition can trigger.
      with self.assertRaisesOpError("Condition x > 0.*|Condition x < y.*"):
        dist.pdf([-1., 1, 1]).eval()
      with self.assertRaisesOpError("Condition x > 0.*"):
        dist.pdf([0., .1, .9]).eval()
      with self.assertRaisesOpError("Condition x ~= y.*"):
        dist.pdf([.1, .2, .8]).eval()

  def testPdfZeroBatches(self):
    with self.test_session():
      alpha = [1., 2]
      x = [.5, .5]
      dist = tf.contrib.distributions.Dirichlet(alpha)
      pdf = dist.pdf(x)
      self.assertAllClose(1., pdf.eval())
      self.assertEqual((), pdf.get_shape())

  def testPdfZeroBatchesNontrivialX(self):
    with self.test_session():
      alpha = [1., 2]
      x = [.3, .7]
      dist = tf.contrib.distributions.Dirichlet(alpha)
      pdf = dist.pdf(x)
      self.assertAllClose(7./5, pdf.eval())
      self.assertEqual((), pdf.get_shape())

  def testPdfUniformZeroBatches(self):
    with self.test_session():
      # Corresponds to a uniform distribution
      alpha = [1., 1, 1]
      x = [[.2, .5, .3], [.3, .4, .3]]
      dist = tf.contrib.distributions.Dirichlet(alpha)
      pdf = dist.pdf(x)
      self.assertAllClose([2., 2.], pdf.eval())
      self.assertEqual((2), pdf.get_shape())

  def testPdfAlphaStretchedInBroadcastWhenSameRank(self):
    with self.test_session():
      alpha = [[1., 2]]
      x = [[.5, .5], [.3, .7]]
      dist = tf.contrib.distributions.Dirichlet(alpha)
      pdf = dist.pdf(x)
      self.assertAllClose([1., 7./5], pdf.eval())
      self.assertEqual((2), pdf.get_shape())

  def testPdfAlphaStretchedInBroadcastWhenLowerRank(self):
    with self.test_session():
      alpha = [1., 2]
      x = [[.5, .5], [.2, .8]]
      pdf = tf.contrib.distributions.Dirichlet(alpha).pdf(x)
      self.assertAllClose([1., 8./5], pdf.eval())
      self.assertEqual((2), pdf.get_shape())

  def testPdfXStretchedInBroadcastWhenSameRank(self):
    with self.test_session():
      alpha = [[1., 2], [2., 3]]
      x = [[.5, .5]]
      pdf = tf.contrib.distributions.Dirichlet(alpha).pdf(x)
      self.assertAllClose([1., 3./2], pdf.eval())
      self.assertEqual((2), pdf.get_shape())

  def testPdfXStretchedInBroadcastWhenLowerRank(self):
    with self.test_session():
      alpha = [[1., 2], [2., 3]]
      x = [.5, .5]
      pdf = tf.contrib.distributions.Dirichlet(alpha).pdf(x)
      self.assertAllClose([1., 3./2], pdf.eval())
      self.assertEqual((2), pdf.get_shape())

  def testDirichletMean(self):
    with self.test_session():
      alpha = [1., 2, 3]
      expected_mean = stats.dirichlet.mean(alpha)
      dirichlet = tf.contrib.distributions.Dirichlet(alpha=alpha)
      self.assertEqual(dirichlet.mean().get_shape(), (3,))
      self.assertAllClose(dirichlet.mean().eval(), expected_mean)

  def testDirichletVariance(self):
    with self.test_session():
      alpha = [1., 2, 3]
      denominator = np.sum(alpha)**2 * (np.sum(alpha) + 1)
      expected_variance = np.diag(stats.dirichlet.var(alpha))
      expected_variance += [
          [0., -2, -3], [-2, 0, -6], [-3, -6, 0]] / denominator
      dirichlet = tf.contrib.distributions.Dirichlet(alpha=alpha)
      self.assertEqual(dirichlet.variance().get_shape(), (3, 3))
      self.assertAllClose(dirichlet.variance().eval(), expected_variance)

  def testDirichletMode(self):
    with self.test_session():
      alpha = np.array([1.1, 2, 3])
      expected_mode = (alpha - 1)/(np.sum(alpha) - 3)
      dirichlet = tf.contrib.distributions.Dirichlet(alpha=alpha)
      self.assertEqual(dirichlet.mode().get_shape(), (3,))
      self.assertAllClose(dirichlet.mode().eval(), expected_mode)

  def testDirichletModeInvalid(self):
    with self.test_session():
      alpha = np.array([1., 2, 3])
      dirichlet = tf.contrib.distributions.Dirichlet(
          alpha=alpha, allow_nan_stats=False)
      with self.assertRaisesOpError("Condition x < y.*"):
        dirichlet.mode().eval()

  def testDirichletModeEnableAllowNanStats(self):
    with self.test_session():
      alpha = np.array([1., 2, 3])
      dirichlet = tf.contrib.distributions.Dirichlet(
          alpha=alpha, allow_nan_stats=True)
      expected_mode = (alpha - 1)/(np.sum(alpha) - 3)
      expected_mode[0] = np.nan

      self.assertEqual(dirichlet.mode().get_shape(), (3,))
      self.assertAllClose(dirichlet.mode().eval(), expected_mode)

  def testDirichletEntropy(self):
    with self.test_session():
      alpha = [1., 2, 3]
      expected_entropy = stats.dirichlet.entropy(alpha)
      dirichlet = tf.contrib.distributions.Dirichlet(alpha=alpha)
      self.assertEqual(dirichlet.entropy().get_shape(), ())
      self.assertAllClose(dirichlet.entropy().eval(), expected_entropy)

  def testDirichletSample(self):
    with self.test_session():
      alpha = [1., 2]
      dirichlet = tf.contrib.distributions.Dirichlet(alpha)
      n = tf.constant(100000)
      samples = dirichlet.sample(n)
      sample_values = samples.eval()
      self.assertEqual(sample_values.shape, (100000, 2))
      self.assertTrue(np.all(sample_values > 0.0))
      self.assertLess(
          stats.kstest(
              # Beta is a univariate distribution.
              sample_values[:, 0], stats.beta(a=1., b=2.).cdf)[0],
          0.01)

if __name__ == "__main__":
  tf.test.main()
