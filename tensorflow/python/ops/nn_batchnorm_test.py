# Copyright 2015 Google Inc. All Rights Reserved.
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
import tensorflow as tf

from tensorflow.python.ops import gen_nn_ops


class BatchNormalizationTest(tf.test.TestCase):

  def _npBatchNorm(self, x, m, v, beta, gamma, epsilon,
                   scale_after_normalization, shift_after_normalization):
    y = (x - m) / np.sqrt(v + epsilon)
    y = y * gamma if scale_after_normalization else y
    return y + beta if shift_after_normalization else y

  def _opsBatchNorm(self, x, m, v, beta, gamma, epsilon,
                    scale_after_normalization, shift_after_normalization):
    y = (x - m) * tf.rsqrt(v + epsilon)
    if scale_after_normalization:
      y = gamma * y
    return y + beta if shift_after_normalization else y

  def _tfBatchNormV1(self, x, m, v, beta, gamma, epsilon,
                     scale_after_normalization):
    """Original implementation."""
    # _batch_norm_with_global_normalization is deprecated in v9
    tf.get_default_graph().graph_def_versions.producer = 8
    # pylint: disable=protected-access
    return gen_nn_ops._batch_norm_with_global_normalization(
        x, m, v, beta, gamma, epsilon, scale_after_normalization)
    # pylint: enable=protected-access

  def _tfBatchNormV1BW(self, x, m, v, beta, gamma, epsilon,
                       scale_after_normalization):
    """Re-implementation of the original kernel for backward compatibility."""
    return tf.nn.batch_norm_with_global_normalization(
        x, m, v, beta, gamma, epsilon, scale_after_normalization)

  def _tfBatchNormV2(self, x, m, v, beta, gamma, epsilon,
                     scale_after_normalization, shift_after_normalization):
    """New implementation."""
    return tf.nn.batch_normalization(
        x, m, v, beta if shift_after_normalization else None,
        gamma if scale_after_normalization else None, epsilon)

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
        x = tf.constant(x_val, name="x")
        m = tf.constant(m_val, name="m")
        v = tf.constant(v_val, name="v")
        beta = tf.constant(beta_val, name="beta")
        gamma = tf.constant(gamma_val, name="gamma")
        epsilon = 0.001
        for scale_after_normalization in [True, False]:
          for shift_after_normalization in [True, False]:
            bn2 = self._tfBatchNormV2(
                x, m, v, beta, gamma, epsilon, scale_after_normalization,
                shift_after_normalization)
            bn1bw = self._tfBatchNormV1BW(
                x, m, v, beta, gamma, epsilon, scale_after_normalization)
            bn1 = self._tfBatchNormV1(
                x, m, v, beta, gamma, epsilon, scale_after_normalization)
            on = self._opsBatchNorm(
                x, m, v, beta, gamma, epsilon, scale_after_normalization,
                shift_after_normalization)
            np_bn = self._npBatchNorm(
                x_val, m_val, v_val, beta_val, gamma_val, epsilon,
                scale_after_normalization, shift_after_normalization)
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

  def _testBatchNormGradient(self, param_index, tag, scale_after_normalization,
                             shift_after_normalization, version,
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
      x = tf.constant(x_val, name="x")
      m = tf.constant(m_val, name="m")
      v = tf.constant(v_val, name="v")
      beta = tf.constant(beta_val, name="beta")
      gamma = tf.constant(gamma_val, name="gamma")
      epsilon = 0.001
      if version == 1:
        output = self._tfBatchNormV1(
            x, m, v, beta, gamma, epsilon, scale_after_normalization)
      elif version == 2:
        output = self._tfBatchNormV2(
            x, m, v, beta, gamma, epsilon, scale_after_normalization,
            shift_after_normalization)
      else:
        print("Invalid version", version)
        raise ValueError()
      all_params = [x, m, v, beta, gamma]
      all_shapes = [x_shape, param_shape, param_shape, param_shape, param_shape]
      err = tf.test.compute_gradient_error(
          all_params[param_index], all_shapes[param_index], output, x_shape)
    print("Batch normalization v%d %s gradient %s scale and %s shift err = " %
          (version, tag, "with" if scale_after_normalization else "without",
           "with" if shift_after_normalization else "without"),
          err)
    self.assertLess(err, err_tolerance)

  def _testBatchNormGradientInAllNeedConfigs(
      self, param_index, tag, err_tolerance=1e-11):
    for scale_after_normalization in [True, False]:
      for shift_after_normalization in [True, False]:
        # shift_after_normalization=False is not supported in version 1.
        for v in ([1, 2] if shift_after_normalization else [2]):
          self._testBatchNormGradient(
              param_index, tag, scale_after_normalization,
              shift_after_normalization, v, err_tolerance)

  def testBatchNormInputGradient(self):
    self._testBatchNormGradientInAllNeedConfigs(0, "x")

  def testBatchNormMeanGradient(self):
    self._testBatchNormGradientInAllNeedConfigs(1, "mean")

  def testBatchNormVarianceGradient(self):
    self._testBatchNormGradientInAllNeedConfigs(2, "variance",
                                                err_tolerance=1e-03)

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
        x = tf.constant(x_val, name="x")
        m = tf.constant(m_val, name="m")
        v = tf.constant(v_val, name="v")
        beta = tf.constant(beta_val, name="beta")
        gamma = tf.constant(gamma_val, name="gamma")
        backprop = tf.constant(backprop_val, name="backprop")
        epsilon = 0.001
        for scale_after_normalization in [True, False]:
          # _batch_norm_with_global_normalization_grad is deprecated in v9
          tf.get_default_graph().graph_def_versions.producer = 8
          grad = gen_nn_ops._batch_norm_with_global_normalization_grad(
              x, m, v, gamma, backprop, epsilon, scale_after_normalization)
          dx, dm, dv, db, dg = grad
          self.assertEqual(grad.dx, dx)
          self.assertEqual(grad.dm, dm)
          self.assertEqual(grad.dv, dv)
          self.assertEqual(grad.db, db)
          self.assertEqual(grad.dg, dg)

          on = self._opsBatchNorm(
              x, m, v, beta, gamma, epsilon, scale_after_normalization, True)
          odx, odm, odv, odb, odg = tf.gradients(
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
        x = tf.constant(x_val, name="x")
        m = tf.constant(m_val, name="m")
        v = tf.constant(v_val, name="v")
        beta = tf.constant(beta_val, name="beta")
        gamma = tf.constant(gamma_val, name="gamma")
        keep_dims_m = tf.reshape(m, keep_dims_param_shape, name="keep_dims_m")
        keep_dims_v = tf.reshape(v, keep_dims_param_shape, name="keep_dims_v")
        keep_dims_beta = tf.reshape(
            beta, keep_dims_param_shape, name="keep_dims_beta")
        keep_dims_gamma = tf.reshape(
            gamma, keep_dims_param_shape, name="keep_dims_gamma")
        epsilon = 0.001
        for scale_after_normalization in [True, False]:
          for shift_after_normalization in [True, False]:
            bn = self._tfBatchNormV2(
                x, m, v, beta, gamma, epsilon, scale_after_normalization,
                shift_after_normalization)
            keep_dims_bn = self._tfBatchNormV2(
                x, keep_dims_m, keep_dims_v, keep_dims_beta,
                keep_dims_gamma, epsilon, scale_after_normalization,
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
        x = tf.constant(x_val, name="x")
        m = tf.constant(m_val, name="m")
        v = tf.constant(v_val, name="v")
        beta = tf.constant(beta_val, name="beta")
        gamma = tf.constant(gamma_val, name="gamma")
        epsilon = 0.001
        for scale_after_normalization in [True, False]:
          for shift_after_normalization in [True, False]:
            bn = self._tfBatchNormV2(
                x, m, v, beta, gamma, epsilon, scale_after_normalization,
                shift_after_normalization)
            np_batch_norm = self._npBatchNorm(
                x_val, m_val, v_val, beta_val, gamma_val, epsilon,
                scale_after_normalization, shift_after_normalization)
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
    self._testBatchNormArbitraryShapes((2, 3, 2, 4, 5), (1, 1, 1, 4, 5),
                                       atol=0.005)


class SufficientStatisticsTest(tf.test.TestCase):

  def _npSuffStats(self, x, axes, shift, keep_dims):
    axis = tuple(axes)
    if shift:
      shift_value = x[[slice(None) if i not in set(axis) else slice(0, 1)
                       for i in xrange(x.ndim)]]
      m_ss = np.sum(x - shift_value, axis=axis, keepdims=keep_dims)
      v_ss = np.sum(
          (x - shift_value) * (x - shift_value),
          axis=axis,
          keepdims=keep_dims)
    else:
      shift_value = None
      m_ss = np.sum(x, axis=axis, keepdims=keep_dims)
      v_ss = np.sum(x * x, axis=axis, keepdims=keep_dims)
    count = 1.0
    for d in xrange(x.ndim):
      if d in set(axes):
        count *= x.shape[d]
    if not keep_dims:
      shift_value = np.squeeze(shift_value, axis=axis)
    return count, m_ss, v_ss, shift_value

  def _opSuffStats(self, x, axes, shift, keep_dims):
    return tf.nn.sufficient_statistics(x, axes, shift, keep_dims)

  def _testSuffStats(self, x_shape, axes, shift, keep_dims, has_shape):
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    np_c, np_m, np_v, np_s = self._npSuffStats(x_val, axes, shift, keep_dims)
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu) as sess:
        if has_shape:
          x = tf.constant(x_val, name="x")
          x.set_shape(x_shape)
          op_c, op_m, op_v, op_s = self._opSuffStats(x, axes, shift, keep_dims)
          if shift:
            tf_c, tf_m, tf_v, tf_s = sess.run([op_c, op_m, op_v, op_s])
          else:
            tf_c, tf_m, tf_v = sess.run([op_c, op_m, op_v])
        else:
          x = tf.placeholder(dtype=tf.float32,
                             shape=[None] * len(x_shape),
                             name="x")
          op_c, op_m, op_v, op_s = self._opSuffStats(x, axes, shift, keep_dims)
          if shift:
            tf_c, tf_m, tf_v, tf_s = sess.run(
                [op_c, op_m, op_v, op_s],
                feed_dict={x: x_val})
          else:
            tf_c, tf_m, tf_v = sess.run(
                [op_c, op_m, op_v],
                feed_dict={x: x_val})
        self.assertAllClose(np_c, tf_c, atol=0.000001)
        self.assertAllClose(np_m, tf_m, atol=0.000001)
        self.assertAllClose(np_v, tf_v, atol=0.000001)
        if shift:
          self.assertAllClose(np_s, tf_s, atol=0.000001)

  def testSuffStats(self):
    for has_shape in [True, False]:
      for keep_dims in [True, False]:
        for shift in [True, False]:
          self._testSuffStats([2, 3], [1], shift, keep_dims, has_shape)
          self._testSuffStats([2, 3], [0], shift, keep_dims, has_shape)
          self._testSuffStats([1, 2, 3], [0, 2], shift, keep_dims, has_shape)


class NormalizeMomentsTest(tf.test.TestCase):

  def _npNormalizeMoments(self, counts, mean_ss, variance_ss, shift):
    mean = mean_ss / counts
    variance = variance_ss / counts - mean * mean
    if shift is not None:
      mean += shift
    return mean, variance

  def _opNormalizeMoments(self, counts, mean_ss, variance_ss, shift):
    return tf.nn.normalize_moments(counts, mean_ss, variance_ss, shift)

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
        tf_counts = tf.constant(counts, name="counts")
        tf_mean_ss = tf.constant(mean_ss, name="mean_ss")
        tf_variance_ss = tf.constant(variance_ss, name="variance_ss")
        if shift:
          tf_shift_v = tf.constant(shift_v, name="shift")
        else:
          tf_shift_v = None
        opm, opv = self._opNormalizeMoments(tf_counts, tf_mean_ss,
                                            tf_variance_ss, tf_shift_v)
        tfm, tfv = sess.run([opm, opv])
        self.assertAllClose(npm, tfm, atol=0.000001)
        self.assertAllClose(npv, tfv, atol=0.000001)

  def testNormalizeMoments(self):
    for shift in [True, False]:
      self._testNormalizeMoments([3], shift)
      self._testNormalizeMoments([2, 3], shift)


class MomentsTest(tf.test.TestCase):

  def RunMomentTestWithDynamicShape(self, shape, axes, keep_dims):
    with self.test_session():
      # shape = [batch, width, height, depth]
      assert len(shape) == 4

      x_numpy = np.random.normal(size=shape).astype(np.float32)
      x = tf.placeholder(tf.float32, shape=[None] * len(shape))

      mean, var = tf.nn.moments(x, axes, keep_dims=keep_dims)

      num_elements = np.prod([shape[i] for i in axes])

      ax = tuple(axes)
      expected_mean = np.sum(
          x_numpy, axis=ax, keepdims=keep_dims) / num_elements
      expected_mean_squared = np.multiply(expected_mean, expected_mean)
      expected_x_squared = np.sum(
          np.multiply(x_numpy, x_numpy),
          axis=ax,
          keepdims=keep_dims) / num_elements
      expected_variance = expected_x_squared - expected_mean_squared

      # Check that the moments are correct.
      self.assertAllClose(expected_mean, mean.eval(feed_dict={x: x_numpy}))
      self.assertAllClose(expected_variance, var.eval(feed_dict={x: x_numpy}))

  def RunMomentTest(self, shape, axes, keep_dims):
    with self.test_session():
      # shape = [batch, width, height, depth]
      assert len(shape) == 4

      x_numpy = np.random.normal(size=shape).astype(np.float32)
      x = tf.constant(x_numpy)

      mean, var = tf.nn.moments(x, axes, keep_dims=keep_dims)

      num_elements = np.prod([shape[i] for i in axes])

      ax = tuple(axes)
      expected_mean = np.sum(
          x_numpy, axis=ax, keepdims=keep_dims) / num_elements
      expected_mean_squared = np.multiply(expected_mean, expected_mean)
      expected_x_squared = np.sum(
          np.multiply(x_numpy, x_numpy),
          axis=ax,
          keepdims=keep_dims) / num_elements
      expected_variance = expected_x_squared - expected_mean_squared

      # Check that the moments are correct.
      self.assertAllClose(expected_mean, mean.eval())
      self.assertAllClose(expected_variance, var.eval())

  def testBasic(self):
    for keep_dims in [False, True]:
      self.RunMomentTest(shape=[2, 3, 5, 4], axes=[0], keep_dims=keep_dims)
      self.RunMomentTestWithDynamicShape(
          shape=[2, 3, 5, 4], axes=[0], keep_dims=keep_dims)

  def testGlobalNormalization(self):
    for keep_dims in [False, True]:
      self.RunMomentTest(
          shape=[2, 3, 5, 4], axes=[0, 1, 2], keep_dims=keep_dims)
      self.RunMomentTestWithDynamicShape(
          shape=[2, 3, 5, 4], axes=[0, 1, 2], keep_dims=keep_dims)

  def testAxes(self):
    for keep_dims in [False, True]:
      self.RunMomentTest(
          shape=[2, 3, 5, 4], axes=[1, 2, 3], keep_dims=keep_dims)
      self.RunMomentTestWithDynamicShape(
          shape=[2, 3, 5, 4], axes=[1, 2, 3], keep_dims=keep_dims)

  def _testGlobalGradient(self, from_y="mean"):
    with self.test_session():
      x_shape = [3, 5, 4, 2]
      x_val = np.random.random_sample(x_shape).astype(np.float64)
      x = tf.constant(x_val)
      x.set_shape(x_shape)

      axes = [0, 1, 2]
      y_shape = [2]  # Depth of x
      out_mean, out_var = tf.nn.moments(x, axes)
      if from_y == "mean":
        y = out_mean
      elif from_y == "var":
        y = out_var
      err = tf.test.compute_gradient_error(x, x_shape, y, y_shape)
      print("Moments %s gradient err = %g" % (from_y, err))
      self.assertLess(err, 1e-11)

  def testMeanGlobalGradient(self):
    self._testGlobalGradient(from_y="mean")

  def testVarGlobalGradient(self):
    self._testGlobalGradient(from_y="var")

  def testOutputNamesNoKeep(self):
    """Make sure the output names are stable."""
    with self.test_session():
      mean, var = tf.nn.moments(tf.constant([1]), [0], keep_dims=False)
      self.assertEquals(mean.op.name, "moments/normalize/mean")
      self.assertEquals(var.op.name, "moments/normalize/variance")

  def testOutputNamesKeep(self):
    """Make sure the output names are stable."""
    with self.test_session():
      mean, var = tf.nn.moments(tf.constant([1]), [0], keep_dims=True)
      self.assertEquals(mean.op.name, "moments/normalize/mean")
      self.assertEquals(var.op.name, "moments/normalize/variance")


if __name__ == "__main__":
  tf.test.main()
