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
"""Tests for batch_norm related functionality in tensorflow.ops.nn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


@test_util.with_c_api
class BatchNormalizationTest(test.TestCase):

  def _npBatchNorm(self, x, m, v, beta, gamma, epsilon,
                   scale_after_normalization, shift_after_normalization):
    y = (x - m) / np.sqrt(v + epsilon)
    y = y * gamma if scale_after_normalization else y
    return y + beta if shift_after_normalization else y

  def _opsBatchNorm(self, x, m, v, beta, gamma, epsilon,
                    scale_after_normalization, shift_after_normalization):
    y = (x - m) * math_ops.rsqrt(v + epsilon)
    if scale_after_normalization:
      y = gamma * y
    return y + beta if shift_after_normalization else y

  def _tfBatchNormV1(self, x, m, v, beta, gamma, epsilon,
                     scale_after_normalization):
    """Original implementation."""
    test_util.set_producer_version(ops.get_default_graph(), 8)
    return gen_nn_ops._batch_norm_with_global_normalization(
        x, m, v, beta, gamma, epsilon, scale_after_normalization)

  def _tfBatchNormV1BW(self, x, m, v, beta, gamma, epsilon,
                       scale_after_normalization):
    """Re-implementation of the original kernel for backward compatibility."""
    return nn_impl.batch_norm_with_global_normalization(
        x, m, v, beta, gamma, epsilon, scale_after_normalization)

  def _tfBatchNormV2(self, x, m, v, beta, gamma, epsilon,
                     scale_after_normalization, shift_after_normalization):
    """New implementation."""
    return nn_impl.batch_normalization(x, m, v, beta if
                                       shift_after_normalization else None,
                                       gamma if scale_after_normalization else
                                       None, epsilon)

  def testBatchNorm(self):
    x_shape = [3, 5, 4, 2]
    param_shape = [2]
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    m_val = np.random.random_sample(param_shape).astype(np.float32)
    v_val = np.random.random_sample(param_shape).astype(np.float32)
    beta_val = np.random.random_sample(param_shape).astype(np.float32)
    gamma_val = np.random.random_sample(param_shape).astype(np.float32)
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu) as sess:
        x = constant_op.constant(x_val, name="x")
        m = constant_op.constant(m_val, name="m")
        v = constant_op.constant(v_val, name="v")
        beta = constant_op.constant(beta_val, name="beta")
        gamma = constant_op.constant(gamma_val, name="gamma")
        epsilon = 0.001
        for scale_after_normalization in [True, False]:
          for shift_after_normalization in [True, False]:
            bn2 = self._tfBatchNormV2(x, m, v, beta, gamma, epsilon,
                                      scale_after_normalization,
                                      shift_after_normalization)
            bn1bw = self._tfBatchNormV1BW(x, m, v, beta, gamma, epsilon,
                                          scale_after_normalization)
            bn1 = self._tfBatchNormV1(x, m, v, beta, gamma, epsilon,
                                      scale_after_normalization)
            on = self._opsBatchNorm(x, m, v, beta, gamma, epsilon,
                                    scale_after_normalization,
                                    shift_after_normalization)
            np_bn = self._npBatchNorm(x_val, m_val, v_val, beta_val, gamma_val,
                                      epsilon, scale_after_normalization,
                                      shift_after_normalization)
            tf_bn_v2, tf_bn_v1bw, tf_bn_v1, ops_bn = sess.run(
                [bn2, bn1bw, bn1, on])
            self.assertAllClose(np_bn, ops_bn, atol=0.00001)
            self.assertAllClose(np_bn, tf_bn_v2, atol=0.00001)
            self.assertAllClose(tf_bn_v2, ops_bn, atol=0.00001)
            # shift_after_normalization=False is not supported in v1.
            if shift_after_normalization:
              self.assertAllClose(np_bn, tf_bn_v1bw, atol=0.00001)
              self.assertAllClose(np_bn, tf_bn_v1, atol=0.00001)
              self.assertAllClose(tf_bn_v1, ops_bn, atol=0.00001)
              self.assertAllClose(tf_bn_v1bw, ops_bn, atol=0.00001)

  def _testBatchNormGradient(self,
                             param_index,
                             tag,
                             scale_after_normalization,
                             shift_after_normalization,
                             version,
                             err_tolerance=1e-11):
    x_shape = [3, 5, 4, 5]
    param_shape = [5]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float64)
    m_val = np.random.random_sample(param_shape).astype(np.float64)
    v_val = np.random.random_sample(param_shape).astype(np.float64)
    beta_val = np.random.random_sample(param_shape).astype(np.float64)
    gamma_val = np.random.random_sample(param_shape).astype(np.float64)
    with self.test_session():
      x = constant_op.constant(x_val, name="x")
      m = constant_op.constant(m_val, name="m")
      v = constant_op.constant(v_val, name="v")
      beta = constant_op.constant(beta_val, name="beta")
      gamma = constant_op.constant(gamma_val, name="gamma")
      epsilon = 0.001
      if version == 1:
        output = self._tfBatchNormV1(x, m, v, beta, gamma, epsilon,
                                     scale_after_normalization)
      elif version == 2:
        output = self._tfBatchNormV2(x, m, v, beta, gamma, epsilon,
                                     scale_after_normalization,
                                     shift_after_normalization)
      else:
        print("Invalid version", version)
        raise ValueError()
      all_params = [x, m, v, beta, gamma]
      all_shapes = [x_shape, param_shape, param_shape, param_shape, param_shape]
      err = gradient_checker.compute_gradient_error(all_params[param_index],
                                                    all_shapes[param_index],
                                                    output, x_shape)
    print("Batch normalization v%d %s gradient %s scale and %s shift err = " %
          (version, tag, "with" if scale_after_normalization else "without",
           "with" if shift_after_normalization else "without"), err)
    self.assertLess(err, err_tolerance)

  def _testBatchNormGradientInAllNeedConfigs(self,
                                             param_index,
                                             tag,
                                             err_tolerance=1e-11):
    for scale_after_normalization in [True, False]:
      for shift_after_normalization in [True, False]:
        # shift_after_normalization=False is not supported in version 1.
        for v in ([1, 2] if shift_after_normalization else [2]):
          self._testBatchNormGradient(param_index, tag,
                                      scale_after_normalization,
                                      shift_after_normalization, v,
                                      err_tolerance)

  def testBatchNormInputGradient(self):
    self._testBatchNormGradientInAllNeedConfigs(0, "x")

  def testBatchNormMeanGradient(self):
    self._testBatchNormGradientInAllNeedConfigs(1, "mean")

  def testBatchNormVarianceGradient(self):
    self._testBatchNormGradientInAllNeedConfigs(
        2, "variance", err_tolerance=1e-03)

  def testBatchNormBetaGradient(self):
    # Since beta does not exist when scale_after_normalization=False, we only
    # test for scale_after_normalization=True.
    for scale_after_normalization in [True, False]:
      for v in [1, 2]:
        self._testBatchNormGradient(3, "beta", scale_after_normalization, True,
                                    v)

  def testBatchNormGammaGradient(self):
    # If scale_after_normalization is False, backprop for gamma in v1
    # will be 0. In version 2 of the API, if scale_after_normalization is False,
    # gamma is not used at all, and the gradient is None, which displeases the
    # gradient checker.
    for scale_after_normalization in [True, False]:
      self._testBatchNormGradient(4, "gamma", scale_after_normalization, True,
                                  1)
    for shift_after_normalization in [True, False]:
      self._testBatchNormGradient(4, "gamma", True, shift_after_normalization,
                                  2)

  def testBatchNormGradImpl(self):
    x_shape = [7, 5, 4, 6]
    param_shape = [6]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    m_val = np.random.random_sample(param_shape).astype(np.float32)
    v_val = np.random.random_sample(param_shape).astype(np.float32)
    beta_val = np.random.random_sample(param_shape).astype(np.float32)
    gamma_val = np.random.random_sample(param_shape).astype(np.float32)
    backprop_val = np.random.random_sample(x_shape).astype(np.float32)
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu) as sess:
        x = constant_op.constant(x_val, name="x")
        m = constant_op.constant(m_val, name="m")
        v = constant_op.constant(v_val, name="v")
        beta = constant_op.constant(beta_val, name="beta")
        gamma = constant_op.constant(gamma_val, name="gamma")
        backprop = constant_op.constant(backprop_val, name="backprop")
        epsilon = 0.001
        for scale_after_normalization in [True, False]:
          # _batch_norm_with_global_normalization_grad is deprecated in v9
          test_util.set_producer_version(ops.get_default_graph(), 8)
          grad = gen_nn_ops.batch_norm_with_global_normalization_grad(
              x, m, v, gamma, backprop, epsilon, scale_after_normalization)
          dx, dm, dv, db, dg = grad
          self.assertEqual(grad.dx, dx)
          self.assertEqual(grad.dm, dm)
          self.assertEqual(grad.dv, dv)
          self.assertEqual(grad.db, db)
          self.assertEqual(grad.dg, dg)

          on = self._opsBatchNorm(x, m, v, beta, gamma, epsilon,
                                  scale_after_normalization, True)
          odx, odm, odv, odb, odg = gradients_impl.gradients(
              [on], [x, m, v, beta, gamma], [backprop])
          if scale_after_normalization:
            all_grads = sess.run([dx, dm, dv, db, dg, odx, odm, odv, odb, odg])
            to_check = ["dx", "dm", "dv", "db", "dg"]
          else:
            all_grads = sess.run([dx, dm, dv, db, odx, odm, odv, odb])
            to_check = ["dx", "dm", "dv", "db"]
          for i, _ in enumerate(to_check):
            self.assertAllClose(
                all_grads[i + len(to_check)], all_grads[i], atol=0.000001)

  def testBatchNormKeepDims(self):
    """Test for tf.nn.moments(..., keep_dims=True / False).

    Make sure that parameters with shape (1, 1, 1, depth) yield the same
    result as parameters with shape (depth)
    """
    x_shape = (3, 5, 4, 2)
    param_shape = (2)
    keep_dims_param_shape = (1, 1, 1, 2)
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    m_val = np.random.random_sample(param_shape).astype(np.float32)
    v_val = np.random.random_sample(param_shape).astype(np.float32)
    beta_val = np.random.random_sample(param_shape).astype(np.float32)
    gamma_val = np.random.random_sample(param_shape).astype(np.float32)
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu) as sess:
        x = constant_op.constant(x_val, name="x")
        m = constant_op.constant(m_val, name="m")
        v = constant_op.constant(v_val, name="v")
        beta = constant_op.constant(beta_val, name="beta")
        gamma = constant_op.constant(gamma_val, name="gamma")
        keep_dims_m = array_ops.reshape(
            m, keep_dims_param_shape, name="keep_dims_m")
        keep_dims_v = array_ops.reshape(
            v, keep_dims_param_shape, name="keep_dims_v")
        keep_dims_beta = array_ops.reshape(
            beta, keep_dims_param_shape, name="keep_dims_beta")
        keep_dims_gamma = array_ops.reshape(
            gamma, keep_dims_param_shape, name="keep_dims_gamma")
        epsilon = 0.001
        for scale_after_normalization in [True, False]:
          for shift_after_normalization in [True, False]:
            bn = self._tfBatchNormV2(x, m, v, beta, gamma, epsilon,
                                     scale_after_normalization,
                                     shift_after_normalization)
            keep_dims_bn = self._tfBatchNormV2(x, keep_dims_m, keep_dims_v,
                                               keep_dims_beta, keep_dims_gamma,
                                               epsilon,
                                               scale_after_normalization,
                                               shift_after_normalization)
            tf_batch_norm, keep_dims_tf_batch_norm = sess.run(
                [bn, keep_dims_bn])
            self.assertEquals(x_shape, tf_batch_norm.shape)
            self.assertEquals(x_shape, keep_dims_tf_batch_norm.shape)
            self.assertAllClose(
                tf_batch_norm, keep_dims_tf_batch_norm, atol=0.000001)

  def _testBatchNormArbitraryShapes(self, x_shape, param_shape, atol=0.0001):
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    m_val = np.random.random_sample(param_shape).astype(np.float32)
    v_val = np.random.random_sample(param_shape).astype(np.float32)
    beta_val = np.random.random_sample(param_shape).astype(np.float32)
    gamma_val = np.random.random_sample(param_shape).astype(np.float32)
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu) as sess:
        x = constant_op.constant(x_val, name="x")
        m = constant_op.constant(m_val, name="m")
        v = constant_op.constant(v_val, name="v")
        beta = constant_op.constant(beta_val, name="beta")
        gamma = constant_op.constant(gamma_val, name="gamma")
        epsilon = 0.001
        for scale_after_normalization in [True, False]:
          for shift_after_normalization in [True, False]:
            bn = self._tfBatchNormV2(x, m, v, beta, gamma, epsilon,
                                     scale_after_normalization,
                                     shift_after_normalization)
            np_batch_norm = self._npBatchNorm(x_val, m_val, v_val, beta_val,
                                              gamma_val, epsilon,
                                              scale_after_normalization,
                                              shift_after_normalization)
            [tf_batch_norm] = sess.run([bn])
            self.assertEquals(x_shape, np_batch_norm.shape)
            self.assertEquals(x_shape, tf_batch_norm.shape)
            self.assertAllClose(np_batch_norm, tf_batch_norm, atol=atol)

  def testBatchNormArbitraryShapes(self):
    """Test for a variety of shapes and moments.

    Batch normalization is expected to work regardless of the position and
    dimensionality of the 'depth' axis/axes.
    """
    self._testBatchNormArbitraryShapes((3, 3), (1, 3))
    self._testBatchNormArbitraryShapes((3, 3), (3, 1))
    self._testBatchNormArbitraryShapes((3, 2, 4, 5), (1, 2, 1, 1))
    self._testBatchNormArbitraryShapes(
        (2, 3, 2, 4, 5), (1, 1, 1, 4, 5), atol=0.005)


@test_util.with_c_api
class SufficientStatisticsTest(test.TestCase):

  def _npSuffStats(self, x, axes, shift, keep_dims):
    axis = tuple(axes)
    if shift is not None:
      m_ss = np.sum(x - shift, axis=axis, keepdims=keep_dims)
      v_ss = np.sum((x - shift) * (x - shift), axis=axis, keepdims=keep_dims)
    else:
      m_ss = np.sum(x, axis=axis, keepdims=keep_dims)
      v_ss = np.sum(x * x, axis=axis, keepdims=keep_dims)
    count = 1.0
    for d in xrange(x.ndim):
      if d in set(axes):
        count *= x.shape[d]
    if not keep_dims:
      shift = np.squeeze(shift, axis=axis)
    return count, m_ss, v_ss, shift

  def _opSuffStats(self, x, axes, shift, keep_dims):
    return nn_impl.sufficient_statistics(x, axes, shift, keep_dims)

  def _testSuffStats(self, x_shape, axes, shift, keep_dims, has_shape):
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    np_c, np_m, np_v, np_s = self._npSuffStats(x_val, axes, shift, keep_dims)
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu) as sess:
        if has_shape:
          x = constant_op.constant(x_val, name="x")
          x.set_shape(x_shape)
          op_c, op_m, op_v, op_s = self._opSuffStats(x, axes, shift, keep_dims)
          if shift:
            tf_c, tf_m, tf_v, tf_s = sess.run([op_c, op_m, op_v, op_s])
          else:
            tf_c, tf_m, tf_v = sess.run([op_c, op_m, op_v])
        else:
          x = array_ops.placeholder(
              dtype=dtypes.float32, shape=[None] * len(x_shape), name="x")
          op_c, op_m, op_v, op_s = self._opSuffStats(x, axes, shift, keep_dims)
          if shift:
            tf_c, tf_m, tf_v, tf_s = sess.run([op_c, op_m, op_v, op_s],
                                              feed_dict={x: x_val})
          else:
            tf_c, tf_m, tf_v = sess.run([op_c, op_m, op_v],
                                        feed_dict={x: x_val})
        self.assertAllClose(np_c, tf_c, atol=0.000001)
        self.assertAllClose(np_m, tf_m, atol=0.000001)
        self.assertAllClose(np_v, tf_v, atol=0.000001)
        if shift:
          self.assertAllClose(np_s, tf_s, atol=0.000001)

  def testSuffStats(self):
    for has_shape in [True, False]:
      for keep_dims in [True, False]:
        for shift in [None, 1.0]:
          self._testSuffStats([2, 3], [1], shift, keep_dims, has_shape)
          self._testSuffStats([2, 3], [0], shift, keep_dims, has_shape)
          self._testSuffStats([1, 2, 3], [0, 2], shift, keep_dims, has_shape)


@test_util.with_c_api
class NormalizeMomentsTest(test.TestCase):

  def _npNormalizeMoments(self, counts, mean_ss, variance_ss, shift):
    mean = mean_ss / counts
    variance = variance_ss / counts - mean * mean
    if shift is not None:
      mean += shift
    return mean, variance

  def _opNormalizeMoments(self, counts, mean_ss, variance_ss, shift):
    return nn_impl.normalize_moments(counts, mean_ss, variance_ss, shift)

  def _testNormalizeMoments(self, shape, shift):
    counts = np.ones([1]).astype(np.float32)
    mean_ss = np.random.random_sample(shape).astype(np.float32)
    variance_ss = np.random.random_sample(shape).astype(np.float32)
    variance_ss *= variance_ss
    if shift:
      shift_v = np.random.random_sample(shape).astype(np.float32)
    else:
      shift_v = None
    npm, npv = self._npNormalizeMoments(counts, mean_ss, variance_ss, shift_v)
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu) as sess:
        tf_counts = constant_op.constant(counts, name="counts")
        tf_mean_ss = constant_op.constant(mean_ss, name="mean_ss")
        tf_variance_ss = constant_op.constant(variance_ss, name="variance_ss")
        if shift:
          tf_shift_v = constant_op.constant(shift_v, name="shift")
        else:
          tf_shift_v = None
        opm, opv = self._opNormalizeMoments(tf_counts, tf_mean_ss,
                                            tf_variance_ss, tf_shift_v)
        tfm, tfv = sess.run([opm, opv])
        self.assertAllClose(npm, tfm, atol=0.000001)
        self.assertAllClose(npv, tfv, atol=0.000001)

  def testNormalizeMoments(self):
    for shift in [None, 4.0]:
      self._testNormalizeMoments([3], shift)
      self._testNormalizeMoments([2, 3], shift)


@test_util.with_c_api
class MomentsTest(test.TestCase):

  def _unweighted_moments(self, x, axes, keep_dims=False, extra_out_grads=None):
    # Method to compute moments of `x` wrt `axes`.
    #
    # This is exposed so WeightedMomentsTest can inherit the tests and
    # assertions from MomentsTest; the extra_out_grads argument allows
    # its inherited gradient tests to assert gradients against the
    # weights as well as the input values.

    return nn_impl.moments(x, axes, keep_dims=keep_dims)

  def RunMomentTestWithDynamicShape(self, shape, axes, keep_dims, dtype):
    with self.test_session():
      # shape = [batch, width, height, depth]
      assert len(shape) == 4

      x_numpy = np.random.normal(size=shape).astype(np.float32)
      x = array_ops.placeholder(dtype, shape=[None] * len(shape))

      mean, var = self._unweighted_moments(x, axes, keep_dims=keep_dims)

      num_elements = np.prod([shape[i] for i in axes])

      ax = tuple(axes)
      expected_mean = np.sum(x_numpy, axis=ax,
                             keepdims=keep_dims) / num_elements
      expected_mean_squared = np.multiply(expected_mean, expected_mean)
      expected_x_squared = np.sum(np.multiply(x_numpy, x_numpy),
                                  axis=ax,
                                  keepdims=keep_dims) / num_elements
      expected_variance = expected_x_squared - expected_mean_squared

      # Check that the moments are correct.
      self.assertAllCloseAccordingToType(
          expected_mean, mean.eval(feed_dict={x: x_numpy}))
      self.assertAllCloseAccordingToType(
          expected_variance, var.eval(feed_dict={x: x_numpy}))

  def RunMomentTest(self, shape, axes, keep_dims, dtype):
    with self.test_session():
      # shape = [batch, width, height, depth]
      assert len(shape) == 4

      x_numpy = np.random.normal(size=shape).astype(np.float32)
      x = math_ops.cast(constant_op.constant(x_numpy), dtype=dtype)

      # Compute the expected values at high precision since the method
      # is prone to catastrophic cancellation:
      x_numpy = x_numpy.astype(np.float128)

      mean, var = self._unweighted_moments(x, axes, keep_dims=keep_dims)

      num_elements = np.prod([shape[i] for i in axes])

      ax = tuple(axes)
      expected_mean = np.sum(x_numpy, axis=ax,
                             keepdims=keep_dims) / num_elements
      expected_mean_squared = np.multiply(expected_mean, expected_mean)
      expected_x_squared = np.sum(np.multiply(x_numpy, x_numpy),
                                  axis=ax,
                                  keepdims=keep_dims) / num_elements
      expected_variance = expected_x_squared - expected_mean_squared

      # Check that the moments are correct.
      self.assertAllCloseAccordingToType(expected_mean, mean.eval())
      self.assertAllCloseAccordingToType(expected_variance, var.eval())

  def testBasic(self):
    for keep_dims in [False, True]:
      for dtype in [dtypes.float32, dtypes.float16]:
        self.RunMomentTest(
            shape=[2, 3, 5, 4], axes=[0], keep_dims=keep_dims, dtype=dtype)
        self.RunMomentTestWithDynamicShape(
            shape=[2, 3, 5, 4], axes=[0], keep_dims=keep_dims, dtype=dtype)

  def testGlobalNormalization(self):
    for keep_dims in [False, True]:
      for dtype in [dtypes.float32, dtypes.float16]:
        self.RunMomentTest(
            shape=[2, 3, 5, 4],
            axes=[0, 1, 2],
            keep_dims=keep_dims,
            dtype=dtype)
        self.RunMomentTestWithDynamicShape(
            shape=[2, 3, 5, 4],
            axes=[0, 1, 2],
            keep_dims=keep_dims,
            dtype=dtype)

  def testAxes(self):
    for keep_dims in [False, True]:
      for dtype in [dtypes.float32, dtypes.float16]:
        self.RunMomentTest(
            shape=[2, 3, 5, 4],
            axes=[1, 2, 3],
            keep_dims=keep_dims,
            dtype=dtype)
        self.RunMomentTestWithDynamicShape(
            shape=[2, 3, 5, 4],
            axes=[1, 2, 3],
            keep_dims=keep_dims,
            dtype=dtype)

  def _testGlobalGradient(self, from_y="mean"):
    with self.test_session():
      x_shape = [3, 5, 4, 2]
      x_val = np.random.random_sample(x_shape).astype(np.float64)
      x = constant_op.constant(x_val)
      x.set_shape(x_shape)

      axes = [0, 1, 2]
      y_shape = [2]  # Depth of x

      inputs_to_compute_gradients_for = [x]

      out_mean, out_var = self._unweighted_moments(
          x, axes, extra_out_grads=inputs_to_compute_gradients_for)
      if from_y == "mean":
        y = out_mean
      elif from_y == "var":
        y = out_var

      for (i, v) in enumerate(inputs_to_compute_gradients_for):
        err = gradient_checker.compute_gradient_error(v,
                                                      v.get_shape().as_list(),
                                                      y, y_shape)
        print("Moments %s gradient err vs input %d = %g" % (from_y, i, err))
        self.assertLess(err, 1e-11)

  def testMeanGlobalGradient(self):
    self._testGlobalGradient(from_y="mean")

  def testVarGlobalGradient(self):
    self._testGlobalGradient(from_y="var")


@test_util.with_c_api
class WeightedMomentsTest(MomentsTest):
  """Tests for nn.weighted_moments.

  Note that this test inherits from MomentsTest, inheriting all its
  test methods!

  It modifies MomentsTest in two ways:

  a) By overriding _unweighted_moments, all the codepaths in
     MomentsTest are executed, but with calls to tf.nn.moments()
     replaced by calls to tf.nn.weighted_moments() with a constant
     weight of 1.

  b) By overriding RunMomentTest and RunMomentTestWithDynamicShape,
     this test adds multiple additional calls to
     RunWeightedMomentsTest() to exercise correctness with
     non-constant weights and varying broadcasting situations. (It
     also continues to call MomentsTest.Run(Weighted)?MomentsTest as
     well.)

  """

  def _unweighted_moments(self, x, axes, keep_dims=False, extra_out_grads=None):
    weights = constant_op.constant(1, dtype=x.dtype)
    if extra_out_grads is not None:
      # We want to assert gradients WRT weights as well as X!
      extra_out_grads.append(weights)
    return nn_impl.weighted_moments(x, axes, weights, keep_dims=keep_dims)

  def RunMomentTest(self, shape, axes, keep_dims, dtype, dynshapes=False):
    if not dynshapes:
      super(WeightedMomentsTest, self).RunMomentTest(shape, axes, keep_dims,
                                                     dtype)
    else:
      super(WeightedMomentsTest, self).RunMomentTestWithDynamicShape(shape,
                                                                     axes,
                                                                     keep_dims,
                                                                     dtype)

    # 1:1 weights and inputs
    self.RunWeightedMomentTest(shape, shape, axes, keep_dims, dtype)

    # Various broadcasting combinations
    for idx in range(len(shape)):
      # try broadcasting weights in all positions
      weight_shape = [1] * len(shape)
      weight_shape[idx] = shape[idx]

      self.RunWeightedMomentTest(shape, weight_shape, axes, keep_dims, dtype)

      # Also try broadcasting with a suffix of length n
      weight_shape = shape[-(idx + 1):]
      self.RunWeightedMomentTest(
          shape, weight_shape, axes, keep_dims, dtype, dynshapes=dynshapes)

  def RunMomentTestWithDynamicShape(self, shape, axes, keep_dims, dtype):
    self.RunMomentTest(shape, axes, keep_dims, dtype, dynshapes=True)

  def RunWeightedMomentTest(self,
                            shape,
                            weights_shape,
                            axes,
                            keep_dims,
                            dtype,
                            dynshapes=False):
    with self.test_session() as s:
      x_numpy = np.random.normal(size=shape).astype(np.float32)
      weights_numpy = np.absolute(  # weights must be positive
          np.random.normal(
              size=weights_shape, loc=1.0).astype(np.float32))

      # Expand the numpy version to higher precision
      x_numpy = x_numpy.astype(np.float128)
      weights_numpy = weights_numpy.astype(np.float128)

      x_shape = [None] * len(shape) if dynshapes else shape
      weights_shape = ([None] * len(weights_shape) if dynshapes else
                       weights_shape)

      x = array_ops.placeholder(dtype, shape=x_shape)
      weights = array_ops.placeholder(dtype, shape=weights_shape)

      mean, var = nn_impl.weighted_moments(
          x, axes, weights, keep_dims=keep_dims)

      ax = tuple(axes)

      def _np_weighted_sum(v):
        return np.sum(weights_numpy * v, axis=ax, keepdims=keep_dims)

      weight_sum = _np_weighted_sum(np.ones_like(x_numpy))
      expected_mean = _np_weighted_sum(x_numpy) / weight_sum
      expected_mean_squared = np.multiply(expected_mean, expected_mean)
      expected_x_squared = (_np_weighted_sum(np.multiply(x_numpy, x_numpy)) /
                            weight_sum)
      expected_variance = expected_x_squared - expected_mean_squared

      mean_v, var_v = s.run([mean, var],
                            feed_dict={x: x_numpy,
                                       weights: weights_numpy})

      self.assertAllCloseAccordingToType(expected_mean, mean_v)
      self.assertAllCloseAccordingToType(expected_variance, var_v)


if __name__ == "__main__":
  test.main()
