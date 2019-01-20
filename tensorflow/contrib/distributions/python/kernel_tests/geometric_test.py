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
"""Tests for the Geometric distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats
from tensorflow.contrib.distributions.python.ops import geometric
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


# In all tests that follow, we use scipy.stats.geom, which
# represents the "Shifted" Geometric distribution. Hence, loc=-1 is passed
# in to each scipy function for testing.
class GeometricTest(test.TestCase):

  def testGeometricShape(self):
    with self.cached_session():
      probs = constant_op.constant([.1] * 5)
      geom = geometric.Geometric(probs=probs)

      self.assertEqual([5,], geom.batch_shape_tensor().eval())
      self.assertAllEqual([], geom.event_shape_tensor().eval())
      self.assertEqual(tensor_shape.TensorShape([5]), geom.batch_shape)
      self.assertEqual(tensor_shape.TensorShape([]), geom.event_shape)

  def testInvalidP(self):
    invalid_ps = [-.01, -0.01, -2.]
    with self.cached_session():
      with self.assertRaisesOpError("Condition x >= 0"):
        geom = geometric.Geometric(probs=invalid_ps, validate_args=True)
        geom.probs.eval()

    invalid_ps = [1.1, 3., 5.]
    with self.cached_session():
      with self.assertRaisesOpError("Condition x <= y"):
        geom = geometric.Geometric(probs=invalid_ps, validate_args=True)
        geom.probs.eval()

  def testGeomLogPmf(self):
    with self.cached_session():
      batch_size = 6
      probs = constant_op.constant([.2] * batch_size)
      probs_v = .2
      x = np.array([2., 3., 4., 5., 6., 7.], dtype=np.float32)
      geom = geometric.Geometric(probs=probs)
      expected_log_prob = stats.geom.logpmf(x, probs_v, loc=-1)
      log_prob = geom.log_prob(x)
      self.assertEqual([6,], log_prob.get_shape())
      self.assertAllClose(expected_log_prob, log_prob.eval())

      pmf = geom.prob(x)
      self.assertEqual([6,], pmf.get_shape())
      self.assertAllClose(np.exp(expected_log_prob), pmf.eval())

  def testGeometricLogPmf_validate_args(self):
    with self.cached_session():
      batch_size = 6
      probs = constant_op.constant([.9] * batch_size)
      x = array_ops.placeholder(dtypes.float32, shape=[6])
      feed_dict = {x: [2.5, 3.2, 4.3, 5.1, 6., 7.]}
      geom = geometric.Geometric(probs=probs, validate_args=True)

      with self.assertRaisesOpError("Condition x == y"):
        log_prob = geom.log_prob(x)
        log_prob.eval(feed_dict=feed_dict)

      with self.assertRaisesOpError("Condition x >= 0"):
        log_prob = geom.log_prob(np.array([-1.], dtype=np.float32))
        log_prob.eval()

      geom = geometric.Geometric(probs=probs)
      log_prob = geom.log_prob(x)
      self.assertEqual([6,], log_prob.get_shape())
      pmf = geom.prob(x)
      self.assertEqual([6,], pmf.get_shape())

  def testGeometricLogPmfMultidimensional(self):
    with self.cached_session():
      batch_size = 6
      probs = constant_op.constant([[.2, .3, .5]] * batch_size)
      probs_v = np.array([.2, .3, .5])
      x = np.array([[2., 3., 4., 5., 6., 7.]], dtype=np.float32).T
      geom = geometric.Geometric(probs=probs)
      expected_log_prob = stats.geom.logpmf(x, probs_v, loc=-1)
      log_prob = geom.log_prob(x)
      log_prob_values = log_prob.eval()
      self.assertEqual([6, 3], log_prob.get_shape())
      self.assertAllClose(expected_log_prob, log_prob_values)

      pmf = geom.prob(x)
      pmf_values = pmf.eval()
      self.assertEqual([6, 3], pmf.get_shape())
      self.assertAllClose(np.exp(expected_log_prob), pmf_values)

  def testGeometricCDF(self):
    with self.cached_session():
      batch_size = 6
      probs = constant_op.constant([[.2, .4, .5]] * batch_size)
      probs_v = np.array([.2, .4, .5])
      x = np.array([[2., 3., 4., 5.5, 6., 7.]], dtype=np.float32).T

      geom = geometric.Geometric(probs=probs)
      expected_cdf = stats.geom.cdf(x, probs_v, loc=-1)

      cdf = geom.cdf(x)
      self.assertEqual([6, 3], cdf.get_shape())
      self.assertAllClose(expected_cdf, cdf.eval())

  def testGeometricEntropy(self):
    with self.cached_session():
      probs_v = np.array([.1, .3, .25], dtype=np.float32)
      geom = geometric.Geometric(probs=probs_v)
      expected_entropy = stats.geom.entropy(probs_v, loc=-1)
      self.assertEqual([3], geom.entropy().get_shape())
      self.assertAllClose(expected_entropy, geom.entropy().eval())

  def testGeometricMean(self):
    with self.cached_session():
      probs_v = np.array([.1, .3, .25])
      geom = geometric.Geometric(probs=probs_v)
      expected_means = stats.geom.mean(probs_v, loc=-1)
      self.assertEqual([3], geom.mean().get_shape())
      self.assertAllClose(expected_means, geom.mean().eval())

  def testGeometricVariance(self):
    with self.cached_session():
      probs_v = np.array([.1, .3, .25])
      geom = geometric.Geometric(probs=probs_v)
      expected_vars = stats.geom.var(probs_v, loc=-1)
      self.assertEqual([3], geom.variance().get_shape())
      self.assertAllClose(expected_vars, geom.variance().eval())

  def testGeometricStddev(self):
    with self.cached_session():
      probs_v = np.array([.1, .3, .25])
      geom = geometric.Geometric(probs=probs_v)
      expected_stddevs = stats.geom.std(probs_v, loc=-1)
      self.assertEqual([3], geom.stddev().get_shape())
      self.assertAllClose(geom.stddev().eval(), expected_stddevs)

  def testGeometricMode(self):
    with self.cached_session():
      probs_v = np.array([.1, .3, .25])
      geom = geometric.Geometric(probs=probs_v)
      self.assertEqual([3,], geom.mode().get_shape())
      self.assertAllClose([0.] * 3, geom.mode().eval())

  def testGeometricSample(self):
    with self.cached_session():
      probs_v = [.3, .9]
      probs = constant_op.constant(probs_v)
      n = constant_op.constant(100000)
      geom = geometric.Geometric(probs=probs)

      samples = geom.sample(n, seed=12345)
      self.assertEqual([100000, 2], samples.get_shape())

      sample_values = samples.eval()
      self.assertFalse(np.any(sample_values < 0.0))
      for i in range(2):
        self.assertAllClose(sample_values[:, i].mean(),
                            stats.geom.mean(probs_v[i], loc=-1),
                            rtol=.02)
        self.assertAllClose(sample_values[:, i].var(),
                            stats.geom.var(probs_v[i], loc=-1),
                            rtol=.02)

  def testGeometricSampleMultiDimensional(self):
    with self.cached_session():
      batch_size = 2
      probs_v = [.3, .9]
      probs = constant_op.constant([probs_v] * batch_size)

      geom = geometric.Geometric(probs=probs)

      n = 400000
      samples = geom.sample(n, seed=12345)
      self.assertEqual([n, batch_size, 2], samples.get_shape())

      sample_values = samples.eval()

      self.assertFalse(np.any(sample_values < 0.0))
      for i in range(2):
        self.assertAllClose(sample_values[:, 0, i].mean(),
                            stats.geom.mean(probs_v[i], loc=-1),
                            rtol=.02)
        self.assertAllClose(sample_values[:, 0, i].var(),
                            stats.geom.var(probs_v[i], loc=-1),
                            rtol=.02)
        self.assertAllClose(sample_values[:, 1, i].mean(),
                            stats.geom.mean(probs_v[i], loc=-1),
                            rtol=.02)
        self.assertAllClose(sample_values[:, 1, i].var(),
                            stats.geom.var(probs_v[i], loc=-1),
                            rtol=.02)

  def testGeometricAtBoundary(self):
    with self.cached_session():
      geom = geometric.Geometric(probs=1., validate_args=True)

      x = np.array([0., 2., 3., 4., 5., 6., 7.], dtype=np.float32)
      expected_log_prob = stats.geom.logpmf(x, [1.], loc=-1)
      # Scipy incorrectly returns nan.
      expected_log_prob[np.isnan(expected_log_prob)] = 0.

      log_prob = geom.log_prob(x)
      self.assertEqual([7,], log_prob.get_shape())
      self.assertAllClose(expected_log_prob, log_prob.eval())

      pmf = geom.prob(x)
      self.assertEqual([7,], pmf.get_shape())
      self.assertAllClose(np.exp(expected_log_prob), pmf.eval())

      expected_log_cdf = stats.geom.logcdf(x, 1., loc=-1)

      log_cdf = geom.log_cdf(x)
      self.assertEqual([7,], log_cdf.get_shape())
      self.assertAllClose(expected_log_cdf, log_cdf.eval())

      cdf = geom.cdf(x)
      self.assertEqual([7,], cdf.get_shape())
      self.assertAllClose(np.exp(expected_log_cdf), cdf.eval())

      expected_mean = stats.geom.mean(1., loc=-1)
      self.assertEqual([], geom.mean().get_shape())
      self.assertAllClose(expected_mean, geom.mean().eval())

      expected_variance = stats.geom.var(1., loc=-1)
      self.assertEqual([], geom.variance().get_shape())
      self.assertAllClose(expected_variance, geom.variance().eval())

      with self.assertRaisesOpError("Entropy is undefined"):
        geom.entropy().eval()


if __name__ == "__main__":
  test.main()
