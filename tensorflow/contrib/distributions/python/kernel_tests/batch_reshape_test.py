# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for BatchReshape."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import batch_reshape as batch_reshape_lib
from tensorflow.contrib.distributions.python.ops import mvn_diag as mvn_lib
from tensorflow.contrib.distributions.python.ops import poisson as poisson_lib
from tensorflow.contrib.distributions.python.ops import wishart as wishart_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.platform import test


class _BatchReshapeTest(object):

  def make_wishart(self, dims, new_batch_shape, old_batch_shape):
    new_batch_shape_ph = (
        constant_op.constant(np.int32(new_batch_shape)) if self.is_static_shape
        else array_ops.placeholder_with_default(
            np.int32(new_batch_shape), shape=None))

    scale = self.dtype([
        [[1., 0.5],
         [0.5, 1.]],
        [[0.5, 0.25],
         [0.25, 0.75]],
    ])
    scale = np.reshape(np.concatenate([scale, scale], axis=0),
                       old_batch_shape + [dims, dims])
    scale_ph = array_ops.placeholder_with_default(
        scale, shape=scale.shape if self.is_static_shape else None)
    wishart = wishart_lib.WishartFull(df=5, scale=scale_ph)
    reshape_wishart = batch_reshape_lib.BatchReshape(
        distribution=wishart,
        batch_shape=new_batch_shape_ph,
        validate_args=True)

    return wishart, reshape_wishart

  def test_matrix_variate_sample_and_log_prob(self):
    dims = 2
    new_batch_shape = [4]
    old_batch_shape = [2, 2]
    wishart, reshape_wishart = self.make_wishart(
        dims, new_batch_shape, old_batch_shape)

    batch_shape = reshape_wishart.batch_shape_tensor()
    event_shape = reshape_wishart.event_shape_tensor()

    expected_sample_shape = [3, 1] + new_batch_shape + [dims, dims]
    x = wishart.sample([3, 1], seed=42)
    expected_sample = array_ops.reshape(x, expected_sample_shape)
    actual_sample = reshape_wishart.sample([3, 1], seed=42)

    expected_log_prob_shape = [3, 1] + new_batch_shape
    expected_log_prob = array_ops.reshape(
        wishart.log_prob(x), expected_log_prob_shape)
    actual_log_prob = reshape_wishart.log_prob(expected_sample)

    with self.test_session() as sess:
      [
          batch_shape_,
          event_shape_,
          expected_sample_, actual_sample_,
          expected_log_prob_, actual_log_prob_,
      ] = sess.run([
          batch_shape,
          event_shape,
          expected_sample, actual_sample,
          expected_log_prob, actual_log_prob,
      ])

    self.assertAllEqual(new_batch_shape, batch_shape_)
    self.assertAllEqual([dims, dims], event_shape_)
    self.assertAllClose(expected_sample_, actual_sample_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_log_prob_, actual_log_prob_,
                        atol=0., rtol=1e-6)
    if not self.is_static_shape:
      return
    self.assertAllEqual(new_batch_shape, reshape_wishart.batch_shape)
    self.assertAllEqual([dims, dims], reshape_wishart.event_shape)
    self.assertAllEqual(expected_sample_shape, actual_sample.shape)
    self.assertAllEqual(expected_log_prob_shape, actual_log_prob.shape)

  def test_matrix_variate_stats(self):
    dims = 2
    new_batch_shape = [4]
    old_batch_shape = [2, 2]
    wishart, reshape_wishart = self.make_wishart(
        dims, new_batch_shape, old_batch_shape)

    expected_scalar_stat_shape = new_batch_shape
    expected_matrix_stat_shape = new_batch_shape + [dims, dims]

    expected_entropy = array_ops.reshape(
        wishart.entropy(), expected_scalar_stat_shape)
    actual_entropy = reshape_wishart.entropy()

    expected_mean = array_ops.reshape(
        wishart.mean(), expected_matrix_stat_shape)
    actual_mean = reshape_wishart.mean()

    expected_mode = array_ops.reshape(
        wishart.mode(), expected_matrix_stat_shape)
    actual_mode = reshape_wishart.mode()

    expected_stddev = array_ops.reshape(
        wishart.stddev(), expected_matrix_stat_shape)
    actual_stddev = reshape_wishart.stddev()

    expected_variance = array_ops.reshape(
        wishart.variance(), expected_matrix_stat_shape)
    actual_variance = reshape_wishart.variance()

    with self.test_session() as sess:
      [
          expected_entropy_, actual_entropy_,
          expected_mean_, actual_mean_,
          expected_mode_, actual_mode_,
          expected_stddev_, actual_stddev_,
          expected_variance_, actual_variance_,
      ] = sess.run([
          expected_entropy, actual_entropy,
          expected_mean, actual_mean,
          expected_mode, actual_mode,
          expected_stddev, actual_stddev,
          expected_variance, actual_variance,
      ])

    self.assertAllClose(expected_entropy_, actual_entropy_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_mean_, actual_mean_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_mode_, actual_mode_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_stddev_, actual_stddev_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_variance_, actual_variance_,
                        atol=0., rtol=1e-6)
    if not self.is_static_shape:
      return
    self.assertAllEqual(expected_scalar_stat_shape, actual_entropy.shape)
    self.assertAllEqual(expected_matrix_stat_shape, actual_mean.shape)
    self.assertAllEqual(expected_matrix_stat_shape, actual_mode.shape)
    self.assertAllEqual(expected_matrix_stat_shape, actual_stddev.shape)
    self.assertAllEqual(expected_matrix_stat_shape, actual_variance.shape)

  def make_normal(self, new_batch_shape, old_batch_shape):
    new_batch_shape_ph = (
        constant_op.constant(np.int32(new_batch_shape)) if self.is_static_shape
        else array_ops.placeholder_with_default(
            np.int32(new_batch_shape), shape=None))

    scale = self.dtype(0.5 + np.arange(
        np.prod(old_batch_shape)).reshape(old_batch_shape))
    scale_ph = array_ops.placeholder_with_default(
        scale, shape=scale.shape if self.is_static_shape else None)
    normal = normal_lib.Normal(loc=self.dtype(0), scale=scale_ph)
    reshape_normal = batch_reshape_lib.BatchReshape(
        distribution=normal,
        batch_shape=new_batch_shape_ph,
        validate_args=True)
    return normal, reshape_normal

  def test_scalar_variate_sample_and_log_prob(self):
    new_batch_shape = [2, 2]
    old_batch_shape = [4]

    normal, reshape_normal = self.make_normal(
        new_batch_shape, old_batch_shape)

    batch_shape = reshape_normal.batch_shape_tensor()
    event_shape = reshape_normal.event_shape_tensor()

    expected_sample_shape = new_batch_shape
    x = normal.sample(seed=52)
    expected_sample = array_ops.reshape(x, expected_sample_shape)
    actual_sample = reshape_normal.sample(seed=52)

    expected_log_prob_shape = new_batch_shape
    expected_log_prob = array_ops.reshape(
        normal.log_prob(x), expected_log_prob_shape)
    actual_log_prob = reshape_normal.log_prob(expected_sample)

    with self.test_session() as sess:
      [
          batch_shape_,
          event_shape_,
          expected_sample_, actual_sample_,
          expected_log_prob_, actual_log_prob_,
      ] = sess.run([
          batch_shape,
          event_shape,
          expected_sample, actual_sample,
          expected_log_prob, actual_log_prob,
      ])
    self.assertAllEqual(new_batch_shape, batch_shape_)
    self.assertAllEqual([], event_shape_)
    self.assertAllClose(expected_sample_, actual_sample_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_log_prob_, actual_log_prob_,
                        atol=0., rtol=1e-6)
    if not self.is_static_shape:
      return
    self.assertAllEqual(new_batch_shape, reshape_normal.batch_shape)
    self.assertAllEqual([], reshape_normal.event_shape)
    self.assertAllEqual(expected_sample_shape, actual_sample.shape)
    self.assertAllEqual(expected_log_prob_shape, actual_log_prob.shape)

  def test_scalar_variate_stats(self):
    new_batch_shape = [2, 2]
    old_batch_shape = [4]

    normal, reshape_normal = self.make_normal(new_batch_shape, old_batch_shape)

    expected_scalar_stat_shape = new_batch_shape

    expected_entropy = array_ops.reshape(
        normal.entropy(), expected_scalar_stat_shape)
    actual_entropy = reshape_normal.entropy()

    expected_mean = array_ops.reshape(
        normal.mean(), expected_scalar_stat_shape)
    actual_mean = reshape_normal.mean()

    expected_mode = array_ops.reshape(
        normal.mode(), expected_scalar_stat_shape)
    actual_mode = reshape_normal.mode()

    expected_stddev = array_ops.reshape(
        normal.stddev(), expected_scalar_stat_shape)
    actual_stddev = reshape_normal.stddev()

    expected_variance = array_ops.reshape(
        normal.variance(), expected_scalar_stat_shape)
    actual_variance = reshape_normal.variance()

    with self.test_session() as sess:
      [
          expected_entropy_, actual_entropy_,
          expected_mean_, actual_mean_,
          expected_mode_, actual_mode_,
          expected_stddev_, actual_stddev_,
          expected_variance_, actual_variance_,
      ] = sess.run([
          expected_entropy, actual_entropy,
          expected_mean, actual_mean,
          expected_mode, actual_mode,
          expected_stddev, actual_stddev,
          expected_variance, actual_variance,
      ])
    self.assertAllClose(expected_entropy_, actual_entropy_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_mean_, actual_mean_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_mode_, actual_mode_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_stddev_, actual_stddev_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_variance_, actual_variance_,
                        atol=0., rtol=1e-6)
    if not self.is_static_shape:
      return
    self.assertAllEqual(expected_scalar_stat_shape, actual_entropy.shape)
    self.assertAllEqual(expected_scalar_stat_shape, actual_mean.shape)
    self.assertAllEqual(expected_scalar_stat_shape, actual_mode.shape)
    self.assertAllEqual(expected_scalar_stat_shape, actual_stddev.shape)
    self.assertAllEqual(expected_scalar_stat_shape, actual_variance.shape)

  def make_mvn(self, dims, new_batch_shape, old_batch_shape):
    new_batch_shape_ph = (
        constant_op.constant(np.int32(new_batch_shape)) if self.is_static_shape
        else array_ops.placeholder_with_default(
            np.int32(new_batch_shape), shape=None))

    scale = np.ones(old_batch_shape + [dims], self.dtype)
    scale_ph = array_ops.placeholder_with_default(
        scale, shape=scale.shape if self.is_static_shape else None)
    mvn = mvn_lib.MultivariateNormalDiag(scale_diag=scale_ph)
    reshape_mvn = batch_reshape_lib.BatchReshape(
        distribution=mvn,
        batch_shape=new_batch_shape_ph,
        validate_args=True)
    return mvn, reshape_mvn

  def test_vector_variate_sample_and_log_prob(self):
    dims = 3
    new_batch_shape = [2, 1]
    old_batch_shape = [2]
    mvn, reshape_mvn = self.make_mvn(
        dims, new_batch_shape, old_batch_shape)

    batch_shape = reshape_mvn.batch_shape_tensor()
    event_shape = reshape_mvn.event_shape_tensor()

    expected_sample_shape = [3] + new_batch_shape + [dims]
    x = mvn.sample(3, seed=62)
    expected_sample = array_ops.reshape(x, expected_sample_shape)
    actual_sample = reshape_mvn.sample(3, seed=62)

    expected_log_prob_shape = [3] + new_batch_shape
    expected_log_prob = array_ops.reshape(
        mvn.log_prob(x), expected_log_prob_shape)
    actual_log_prob = reshape_mvn.log_prob(expected_sample)

    with self.test_session() as sess:
      [
          batch_shape_,
          event_shape_,
          expected_sample_, actual_sample_,
          expected_log_prob_, actual_log_prob_,
      ] = sess.run([
          batch_shape,
          event_shape,
          expected_sample, actual_sample,
          expected_log_prob, actual_log_prob,
      ])
    self.assertAllEqual(new_batch_shape, batch_shape_)
    self.assertAllEqual([dims], event_shape_)
    self.assertAllClose(expected_sample_, actual_sample_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_log_prob_, actual_log_prob_,
                        atol=0., rtol=1e-6)
    if not self.is_static_shape:
      return
    self.assertAllEqual(new_batch_shape, reshape_mvn.batch_shape)
    self.assertAllEqual([dims], reshape_mvn.event_shape)
    self.assertAllEqual(expected_sample_shape, actual_sample.shape)
    self.assertAllEqual(expected_log_prob_shape, actual_log_prob.shape)

  def test_vector_variate_stats(self):
    dims = 3
    new_batch_shape = [2, 1]
    old_batch_shape = [2]
    mvn, reshape_mvn = self.make_mvn(
        dims, new_batch_shape, old_batch_shape)

    expected_scalar_stat_shape = new_batch_shape

    expected_entropy = array_ops.reshape(
        mvn.entropy(), expected_scalar_stat_shape)
    actual_entropy = reshape_mvn.entropy()

    expected_vector_stat_shape = new_batch_shape + [dims]

    expected_mean = array_ops.reshape(
        mvn.mean(), expected_vector_stat_shape)
    actual_mean = reshape_mvn.mean()

    expected_mode = array_ops.reshape(
        mvn.mode(), expected_vector_stat_shape)
    actual_mode = reshape_mvn.mode()

    expected_stddev = array_ops.reshape(
        mvn.stddev(), expected_vector_stat_shape)
    actual_stddev = reshape_mvn.stddev()

    expected_variance = array_ops.reshape(
        mvn.variance(), expected_vector_stat_shape)
    actual_variance = reshape_mvn.variance()

    expected_matrix_stat_shape = new_batch_shape + [dims, dims]

    expected_covariance = array_ops.reshape(
        mvn.covariance(), expected_matrix_stat_shape)
    actual_covariance = reshape_mvn.covariance()

    with self.test_session() as sess:
      [
          expected_entropy_, actual_entropy_,
          expected_mean_, actual_mean_,
          expected_mode_, actual_mode_,
          expected_stddev_, actual_stddev_,
          expected_variance_, actual_variance_,
          expected_covariance_, actual_covariance_,
      ] = sess.run([
          expected_entropy, actual_entropy,
          expected_mean, actual_mean,
          expected_mode, actual_mode,
          expected_stddev, actual_stddev,
          expected_variance, actual_variance,
          expected_covariance, actual_covariance,
      ])
    self.assertAllClose(expected_entropy_, actual_entropy_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_mean_, actual_mean_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_mode_, actual_mode_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_stddev_, actual_stddev_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_variance_, actual_variance_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_covariance_, actual_covariance_,
                        atol=0., rtol=1e-6)
    if not self.is_static_shape:
      return
    self.assertAllEqual(expected_scalar_stat_shape, actual_entropy.shape)
    self.assertAllEqual(expected_vector_stat_shape, actual_mean.shape)
    self.assertAllEqual(expected_vector_stat_shape, actual_mode.shape)
    self.assertAllEqual(expected_vector_stat_shape, actual_stddev.shape)
    self.assertAllEqual(expected_vector_stat_shape, actual_variance.shape)
    self.assertAllEqual(expected_matrix_stat_shape, actual_covariance.shape)

  def test_bad_reshape_size(self):
    dims = 2
    new_batch_shape = [2, 3]
    old_batch_shape = [2]   # 2 != 2*3

    new_batch_shape_ph = (
        constant_op.constant(np.int32(new_batch_shape)) if self.is_static_shape
        else array_ops.placeholder_with_default(
            np.int32(new_batch_shape), shape=None))

    scale = np.ones(old_batch_shape + [dims], self.dtype)
    scale_ph = array_ops.placeholder_with_default(
        scale, shape=scale.shape if self.is_static_shape else None)
    mvn = mvn_lib.MultivariateNormalDiag(scale_diag=scale_ph)

    if self.is_static_shape:
      with self.assertRaisesRegexp(
          ValueError, (r"`batch_shape` size \(6\) must match "
                       r"`distribution\.batch_shape` size \(2\)")):
        batch_reshape_lib.BatchReshape(
            distribution=mvn,
            batch_shape=new_batch_shape_ph,
            validate_args=True)

    else:
      with self.test_session():
        with self.assertRaisesOpError(r"`batch_shape` size must match "
                                      r"`distributions.batch_shape` size"):
          batch_reshape_lib.BatchReshape(
              distribution=mvn,
              batch_shape=new_batch_shape_ph,
              validate_args=True).sample().eval()

  def test_non_positive_shape(self):
    dims = 2
    new_batch_shape = [-1, -2]   # -1*-2=2 so will pass size check.
    old_batch_shape = [2]

    new_batch_shape_ph = (
        constant_op.constant(np.int32(new_batch_shape)) if self.is_static_shape
        else array_ops.placeholder_with_default(
            np.int32(new_batch_shape), shape=None))

    scale = np.ones(old_batch_shape + [dims], self.dtype)
    scale_ph = array_ops.placeholder_with_default(
        scale, shape=scale.shape if self.is_static_shape else None)
    mvn = mvn_lib.MultivariateNormalDiag(scale_diag=scale_ph)

    if self.is_static_shape:
      with self.assertRaisesRegexp(ValueError, r".*must be positive.*"):
        batch_reshape_lib.BatchReshape(
            distribution=mvn,
            batch_shape=new_batch_shape_ph,
            validate_args=True)

    else:
      with self.test_session():
        with self.assertRaisesOpError(r".*must be positive.*"):
          batch_reshape_lib.BatchReshape(
              distribution=mvn,
              batch_shape=new_batch_shape_ph,
              validate_args=True).sample().eval()

  def test_non_vector_shape(self):
    dims = 2
    new_batch_shape = 2
    old_batch_shape = [2]

    new_batch_shape_ph = (
        constant_op.constant(np.int32(new_batch_shape)) if self.is_static_shape
        else array_ops.placeholder_with_default(
            np.int32(new_batch_shape), shape=None))

    scale = np.ones(old_batch_shape + [dims], self.dtype)
    scale_ph = array_ops.placeholder_with_default(
        scale, shape=scale.shape if self.is_static_shape else None)
    mvn = mvn_lib.MultivariateNormalDiag(scale_diag=scale_ph)

    if self.is_static_shape:
      with self.assertRaisesRegexp(ValueError, r".*must be a vector.*"):
        batch_reshape_lib.BatchReshape(
            distribution=mvn,
            batch_shape=new_batch_shape_ph,
            validate_args=True)

    else:
      with self.test_session():
        with self.assertRaisesOpError(r".*must be a vector.*"):
          batch_reshape_lib.BatchReshape(
              distribution=mvn,
              batch_shape=new_batch_shape_ph,
              validate_args=True).sample().eval()

  def test_broadcasting_explicitly_unsupported(self):
    old_batch_shape = [4]
    new_batch_shape = [1, 4, 1]
    rate_ = self.dtype([1, 10, 2, 20])

    rate = array_ops.placeholder_with_default(
        rate_,
        shape=old_batch_shape if self.is_static_shape else None)
    poisson_4 = poisson_lib.Poisson(rate)
    new_batch_shape_ph = (
        constant_op.constant(np.int32(new_batch_shape)) if self.is_static_shape
        else array_ops.placeholder_with_default(
            np.int32(new_batch_shape), shape=None))
    poisson_141_reshaped = batch_reshape_lib.BatchReshape(
        poisson_4, new_batch_shape_ph, validate_args=True)

    x_4 = self.dtype([2, 12, 3, 23])
    x_114 = self.dtype([2, 12, 3, 23]).reshape(1, 1, 4)

    if self.is_static_shape:
      with self.assertRaisesRegexp(NotImplementedError,
                                   "too few event dims"):
        poisson_141_reshaped.log_prob(x_4)
      with self.assertRaisesRegexp(NotImplementedError,
                                   "unexpected batch and event shape"):
        poisson_141_reshaped.log_prob(x_114)
      return

    with self.assertRaisesOpError("too few event dims"):
      with self.test_session():
        poisson_141_reshaped.log_prob(x_4).eval()

    with self.assertRaisesOpError("unexpected batch and event shape"):
      with self.test_session():
        poisson_141_reshaped.log_prob(x_114).eval()


class BatchReshapeStaticTest(_BatchReshapeTest, test.TestCase):

  dtype = np.float32
  is_static_shape = True


class BatchReshapeDynamicTest(_BatchReshapeTest, test.TestCase):

  dtype = np.float64
  is_static_shape = False


if __name__ == "__main__":
  test.main()
