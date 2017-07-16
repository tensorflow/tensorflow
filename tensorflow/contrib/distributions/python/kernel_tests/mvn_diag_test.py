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
"""Tests for MultivariateNormal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import stats
from tensorflow.contrib import distributions
from tensorflow.contrib.distributions.python.ops import bijectors
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


ds = distributions


class MultivariateNormalDiagTest(test.TestCase):
  """Well tested because this is a simple override of the base class."""

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testScalarParams(self):
    mu = -1.
    diag = -5.
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "at least 1 dimension"):
        ds.MultivariateNormalDiag(mu, diag)

  def testVectorParams(self):
    mu = [-1.]
    diag = [-5.]
    with self.test_session():
      dist = ds.MultivariateNormalDiag(mu, diag, validate_args=True)
      self.assertAllEqual([3, 1], dist.sample(3).get_shape())

  def testDistWithBatchShapeOneThenTransformedThroughSoftplus(self):
    # This complex combination of events resulted in a loss of static shape
    # information when tensor_util.constant_value(self._needs_rotation) was
    # being used incorrectly (resulting in always rotating).
    # Batch shape = [1], event shape = [3]
    mu = array_ops.zeros((1, 3))
    diag = array_ops.ones((1, 3))
    with self.test_session():
      base_dist = ds.MultivariateNormalDiag(mu, diag, validate_args=True)
      dist = ds.TransformedDistribution(
          base_dist,
          validate_args=True,
          bijector=bijectors.Softplus(event_ndims=1))
      samps = dist.sample(5)  # Shape [5, 1, 3].
      self.assertAllEqual([5, 1], dist.log_prob(samps).get_shape())

  def testMean(self):
    mu = [-1., 1]
    diag = [1., -5]
    with self.test_session():
      dist = ds.MultivariateNormalDiag(mu, diag, validate_args=True)
      self.assertAllEqual(mu, dist.mean().eval())

  def testMeanWithBroadcastLoc(self):
    mu = [-1.]
    diag = [1., -5]
    with self.test_session():
      dist = ds.MultivariateNormalDiag(mu, diag, validate_args=True)
      self.assertAllEqual([-1., -1.], dist.mean().eval())

  def testEntropy(self):
    mu = [-1., 1]
    diag = [-1., 5]
    diag_mat = np.diag(diag)
    scipy_mvn = stats.multivariate_normal(mean=mu, cov=diag_mat**2)
    with self.test_session():
      dist = ds.MultivariateNormalDiag(mu, diag, validate_args=True)
      self.assertAllClose(scipy_mvn.entropy(), dist.entropy().eval(), atol=1e-4)

  def testSample(self):
    mu = [-1., 1]
    diag = [1., -2]
    with self.test_session():
      dist = ds.MultivariateNormalDiag(mu, diag, validate_args=True)
      samps = dist.sample(int(1e3), seed=0).eval()
      cov_mat = array_ops.matrix_diag(diag).eval()**2

      self.assertAllClose(mu, samps.mean(axis=0),
                          atol=0., rtol=0.05)
      self.assertAllClose(cov_mat, np.cov(samps.T),
                          atol=0.05, rtol=0.05)

  def testSingularScaleRaises(self):
    mu = [-1., 1]
    diag = [1., 0]
    with self.test_session():
      dist = ds.MultivariateNormalDiag(mu, diag, validate_args=True)
      with self.assertRaisesOpError("Singular"):
        dist.sample().eval()

  def testSampleWithBroadcastScale(self):
    # mu corresponds to a 2-batch of 3-variate normals
    mu = np.zeros([2, 3])

    # diag corresponds to no batches of 3-variate normals
    diag = np.ones([3])

    with self.test_session():
      dist = ds.MultivariateNormalDiag(mu, diag, validate_args=True)

      mean = dist.mean()
      self.assertAllEqual([2, 3], mean.get_shape())
      self.assertAllClose(mu, mean.eval())

      n = int(1e3)
      samps = dist.sample(n, seed=0).eval()
      cov_mat = array_ops.matrix_diag(diag).eval()**2
      sample_cov = np.matmul(samps.transpose([1, 2, 0]),
                             samps.transpose([1, 0, 2])) / n

      self.assertAllClose(mu, samps.mean(axis=0),
                          atol=0.10, rtol=0.05)
      self.assertAllClose([cov_mat, cov_mat], sample_cov,
                          atol=0.10, rtol=0.05)

  def testCovariance(self):
    with self.test_session():
      mvn = ds.MultivariateNormalDiag(
          loc=array_ops.zeros([2, 3], dtype=dtypes.float32))
      self.assertAllClose(
          np.diag(np.ones([3], dtype=np.float32)),
          mvn.covariance().eval())

      mvn = ds.MultivariateNormalDiag(
          loc=array_ops.zeros([3], dtype=dtypes.float32),
          scale_identity_multiplier=[3., 2.])
      self.assertAllEqual([2], mvn.batch_shape)
      self.assertAllEqual([3], mvn.event_shape)
      self.assertAllClose(
          np.array([[[3., 0, 0],
                     [0, 3, 0],
                     [0, 0, 3]],
                    [[2, 0, 0],
                     [0, 2, 0],
                     [0, 0, 2]]])**2.,
          mvn.covariance().eval())

      mvn = ds.MultivariateNormalDiag(
          loc=array_ops.zeros([3], dtype=dtypes.float32),
          scale_diag=[[3., 2, 1], [4, 5, 6]])
      self.assertAllEqual([2], mvn.batch_shape)
      self.assertAllEqual([3], mvn.event_shape)
      self.assertAllClose(
          np.array([[[3., 0, 0],
                     [0, 2, 0],
                     [0, 0, 1]],
                    [[4, 0, 0],
                     [0, 5, 0],
                     [0, 0, 6]]])**2.,
          mvn.covariance().eval())

  def testVariance(self):
    with self.test_session():
      mvn = ds.MultivariateNormalDiag(
          loc=array_ops.zeros([2, 3], dtype=dtypes.float32))
      self.assertAllClose(
          np.ones([3], dtype=np.float32),
          mvn.variance().eval())

      mvn = ds.MultivariateNormalDiag(
          loc=array_ops.zeros([3], dtype=dtypes.float32),
          scale_identity_multiplier=[3., 2.])
      self.assertAllClose(
          np.array([[3., 3, 3],
                    [2, 2, 2]])**2.,
          mvn.variance().eval())

      mvn = ds.MultivariateNormalDiag(
          loc=array_ops.zeros([3], dtype=dtypes.float32),
          scale_diag=[[3., 2, 1],
                      [4, 5, 6]])
      self.assertAllClose(
          np.array([[3., 2, 1],
                    [4, 5, 6]])**2.,
          mvn.variance().eval())

  def testStddev(self):
    with self.test_session():
      mvn = ds.MultivariateNormalDiag(
          loc=array_ops.zeros([2, 3], dtype=dtypes.float32))
      self.assertAllClose(
          np.ones([3], dtype=np.float32),
          mvn.stddev().eval())

      mvn = ds.MultivariateNormalDiag(
          loc=array_ops.zeros([3], dtype=dtypes.float32),
          scale_identity_multiplier=[3., 2.])
      self.assertAllClose(
          np.array([[3., 3, 3],
                    [2, 2, 2]]),
          mvn.stddev().eval())

      mvn = ds.MultivariateNormalDiag(
          loc=array_ops.zeros([3], dtype=dtypes.float32),
          scale_diag=[[3., 2, 1], [4, 5, 6]])
      self.assertAllClose(
          np.array([[3., 2, 1],
                    [4, 5, 6]]),
          mvn.stddev().eval())

  def testMultivariateNormalDiagWithSoftplusScale(self):
    mu = [-1.0, 1.0]
    diag = [-1.0, -2.0]
    with self.test_session():
      dist = ds.MultivariateNormalDiagWithSoftplusScale(
          mu, diag, validate_args=True)
      samps = dist.sample(1000, seed=0).eval()
      cov_mat = array_ops.matrix_diag(nn_ops.softplus(diag)).eval()**2

      self.assertAllClose(mu, samps.mean(axis=0), atol=0.1)
      self.assertAllClose(cov_mat, np.cov(samps.T), atol=0.1)


if __name__ == "__main__":
  test.main()
