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
"""Tests for VectorExponentialLinearOperator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib import distributions
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


ds = distributions


class VectorExponentialDiagTest(test.TestCase):
  """Well tested because this is a simple override of the base class."""

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def testScalarParams(self):
    mu = -1.
    diag = -5.
    with self.cached_session():
      with self.assertRaisesRegexp(ValueError, "at least 1 dimension"):
        ds.VectorExponentialDiag(mu, diag)

  def testVectorParams(self):
    mu = [-1.]
    diag = [-5.]
    with self.cached_session():
      dist = ds.VectorExponentialDiag(mu, diag, validate_args=True)
      self.assertAllEqual([3, 1], dist.sample(3).get_shape())

  def testMean(self):
    mu = [-1., 1]
    diag = [1., -5]
    with self.cached_session():
      dist = ds.VectorExponentialDiag(mu, diag, validate_args=True)
      self.assertAllEqual([-1. + 1., 1. - 5.], dist.mean().eval())

  def testMode(self):
    mu = [-1.]
    diag = [1., -5]
    with self.cached_session():
      dist = ds.VectorExponentialDiag(mu, diag, validate_args=True)
      self.assertAllEqual([-1., -1.], dist.mode().eval())

  def testMeanWithBroadcastLoc(self):
    mu = [-1.]
    diag = [1., -5]
    with self.cached_session():
      dist = ds.VectorExponentialDiag(mu, diag, validate_args=True)
      self.assertAllEqual([-1. + 1, -1. - 5], dist.mean().eval())

  def testSample(self):
    mu = [-2., 1]
    diag = [1., -2]
    with self.cached_session():
      dist = ds.VectorExponentialDiag(mu, diag, validate_args=True)
      samps = dist.sample(int(1e4), seed=0).eval()
      cov_mat = array_ops.matrix_diag(diag).eval()**2

      self.assertAllClose([-2 + 1, 1. - 2], samps.mean(axis=0),
                          atol=0., rtol=0.05)
      self.assertAllClose(cov_mat, np.cov(samps.T),
                          atol=0.05, rtol=0.05)

  def testSingularScaleRaises(self):
    mu = [-1., 1]
    diag = [1., 0]
    with self.cached_session():
      dist = ds.VectorExponentialDiag(mu, diag, validate_args=True)
      with self.assertRaisesOpError("Singular"):
        dist.sample().eval()

  def testSampleWithBroadcastScale(self):
    # mu corresponds to a 2-batch of 3-variate normals
    mu = np.zeros([2, 3])

    # diag corresponds to no batches of 3-variate normals
    diag = np.ones([3])

    with self.cached_session():
      dist = ds.VectorExponentialDiag(mu, diag, validate_args=True)

      mean = dist.mean()
      self.assertAllEqual([2, 3], mean.get_shape())
      self.assertAllClose(mu + diag, mean.eval())

      n = int(1e4)
      samps = dist.sample(n, seed=0).eval()
      samps_centered = samps - samps.mean(axis=0)
      cov_mat = array_ops.matrix_diag(diag).eval()**2
      sample_cov = np.matmul(samps_centered.transpose([1, 2, 0]),
                             samps_centered.transpose([1, 0, 2])) / n

      self.assertAllClose(mu + diag, samps.mean(axis=0),
                          atol=0.10, rtol=0.05)
      self.assertAllClose([cov_mat, cov_mat], sample_cov,
                          atol=0.10, rtol=0.05)

  def testCovariance(self):
    with self.cached_session():
      vex = ds.VectorExponentialDiag(
          loc=array_ops.ones([2, 3], dtype=dtypes.float32))
      self.assertAllClose(
          np.diag(np.ones([3], dtype=np.float32)),
          vex.covariance().eval())

      vex = ds.VectorExponentialDiag(
          loc=array_ops.ones([3], dtype=dtypes.float32),
          scale_identity_multiplier=[3., 2.])
      self.assertAllEqual([2], vex.batch_shape)
      self.assertAllEqual([3], vex.event_shape)
      self.assertAllClose(
          np.array([[[3., 0, 0],
                     [0, 3, 0],
                     [0, 0, 3]],
                    [[2, 0, 0],
                     [0, 2, 0],
                     [0, 0, 2]]])**2.,
          vex.covariance().eval())

      vex = ds.VectorExponentialDiag(
          loc=array_ops.ones([3], dtype=dtypes.float32),
          scale_diag=[[3., 2, 1], [4, 5, 6]])
      self.assertAllEqual([2], vex.batch_shape)
      self.assertAllEqual([3], vex.event_shape)
      self.assertAllClose(
          np.array([[[3., 0, 0],
                     [0, 2, 0],
                     [0, 0, 1]],
                    [[4, 0, 0],
                     [0, 5, 0],
                     [0, 0, 6]]])**2.,
          vex.covariance().eval())

  def testVariance(self):
    with self.cached_session():
      vex = ds.VectorExponentialDiag(
          loc=array_ops.zeros([2, 3], dtype=dtypes.float32))
      self.assertAllClose(
          np.ones([3], dtype=np.float32),
          vex.variance().eval())

      vex = ds.VectorExponentialDiag(
          loc=array_ops.ones([3], dtype=dtypes.float32),
          scale_identity_multiplier=[3., 2.])
      self.assertAllClose(
          np.array([[3., 3, 3],
                    [2., 2, 2]])**2.,
          vex.variance().eval())

      vex = ds.VectorExponentialDiag(
          loc=array_ops.ones([3], dtype=dtypes.float32),
          scale_diag=[[3., 2, 1],
                      [4., 5, 6]])
      self.assertAllClose(
          np.array([[3., 2, 1],
                    [4., 5, 6]])**2.,
          vex.variance().eval())

  def testStddev(self):
    with self.cached_session():
      vex = ds.VectorExponentialDiag(
          loc=array_ops.zeros([2, 3], dtype=dtypes.float32))
      self.assertAllClose(
          np.ones([3], dtype=np.float32),
          vex.stddev().eval())

      vex = ds.VectorExponentialDiag(
          loc=array_ops.zeros([3], dtype=dtypes.float32),
          scale_identity_multiplier=[3., 2.])
      self.assertAllClose(
          np.array([[3., 3, 3],
                    [2., 2, 2]]),
          vex.stddev().eval())

      vex = ds.VectorExponentialDiag(
          loc=array_ops.zeros([3], dtype=dtypes.float32),
          scale_diag=[[3., 2, 1], [4, 5, 6]])
      self.assertAllClose(
          np.array([[3., 2, 1],
                    [4., 5, 6]]),
          vex.stddev().eval())


if __name__ == "__main__":
  test.main()
