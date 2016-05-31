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

# Factor out batchnorm test into separate file because it uses older
# GraphDev version, so we need to globally set env to use older GraphDef
# version

"""Tests for tensorflow.ops.nn."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.ops import gen_nn_ops

exp = math.exp
log = math.log


from tensorflow.contrib.immediate.python.immediate import test_util
import tensorflow.contrib.immediate as immediate

env = immediate.Env({"tf": tf, "gen_nn_ops": gen_nn_ops})
tf = env.tf
gen_nn_ops = env.gen_nn_ops
env.g.graph_def_versions.producer = 8


class BatchNormalizationTest(test_util.TensorFlowTestCase):

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

#   def _testBatchNormGradient(self, param_index, tag, scale_after_normalization,
#                              shift_after_normalization, version,
#                              err_tolerance=1e-11):
#     x_shape = [3, 5, 4, 5]
#     param_shape = [5]
#     np.random.seed(1)  # Make it reproducible.
#     x_val = np.random.random_sample(x_shape).astype(np.float64)
#     m_val = np.random.random_sample(param_shape).astype(np.float64)
#     v_val = np.random.random_sample(param_shape).astype(np.float64)
#     beta_val = np.random.random_sample(param_shape).astype(np.float64)
#     gamma_val = np.random.random_sample(param_shape).astype(np.float64)
#     with self.test_session():
#       x = tf.constant(x_val, name="x")
#       m = tf.constant(m_val, name="m")
#       v = tf.constant(v_val, name="v")
#       beta = tf.constant(beta_val, name="beta")
#       gamma = tf.constant(gamma_val, name="gamma")
#       epsilon = 0.001
#       if version == 1:
#         output = self._tfBatchNormV1(
#             x, m, v, beta, gamma, epsilon, scale_after_normalization)
#       elif version == 2:
#         output = self._tfBatchNormV2(
#             x, m, v, beta, gamma, epsilon, scale_after_normalization,
#             shift_after_normalization)
#       else:
#         print("Invalid version", version)
#         raise ValueError()
#       all_params = [x, m, v, beta, gamma]
#       all_shapes = [x_shape, param_shape, param_shape, param_shape, param_shape]
#       err = tf.test.compute_gradient_error(
#           all_params[param_index], all_shapes[param_index], output, x_shape)
#     print("Batch normalization v%d %s gradient %s scale and %s shift err = " %
#           (version, tag, "with" if scale_after_normalization else "without",
#            "with" if shift_after_normalization else "without"),
#           err)
#     self.assertLess(err, err_tolerance)

#   def _testBatchNormGradientInAllNeedConfigs(
#       self, param_index, tag, err_tolerance=1e-11):
#     for scale_after_normalization in [True, False]:
#       for shift_after_normalization in [True, False]:
#         # shift_after_normalization=False is not supported in version 1.
#         for v in ([1, 2] if shift_after_normalization else [2]):
#           self._testBatchNormGradient(
#               param_index, tag, scale_after_normalization,
#               shift_after_normalization, v, err_tolerance)

#   def testBatchNormInputGradient(self):
#     self._testBatchNormGradientInAllNeedConfigs(0, "x")

#   def testBatchNormMeanGradient(self):
#     self._testBatchNormGradientInAllNeedConfigs(1, "mean")

#   def testBatchNormVarianceGradient(self):
#     self._testBatchNormGradientInAllNeedConfigs(2, "variance",
#                                                 err_tolerance=1e-03)

#   def testBatchNormBetaGradient(self):
#     # Since beta does not exist when scale_after_normalization=False, we only
#     # test for scale_after_normalization=True.
#     for scale_after_normalization in [True, False]:
#       for v in [1, 2]:
#         self._testBatchNormGradient(3, "beta", scale_after_normalization, True,
#                                     v)

#   def testBatchNormGammaGradient(self):
#     # If scale_after_normalization is False, backprop for gamma in v1
#     # will be 0. In version 2 of the API, if scale_after_normalization is False,
#     # gamma is not used at all, and the gradient is None, which displeases the
#     # gradient checker.
#     for scale_after_normalization in [True, False]:
#       self._testBatchNormGradient(4, "gamma", scale_after_normalization, True,
#                                   1)
#     for shift_after_normalization in [True, False]:
#       self._testBatchNormGradient(4, "gamma", True, shift_after_normalization,
#                                   2)

#   def testBatchNormGradImpl(self):
#     x_shape = [7, 5, 4, 6]
#     param_shape = [6]
#     np.random.seed(1)  # Make it reproducible.
#     x_val = np.random.random_sample(x_shape).astype(np.float32)
#     m_val = np.random.random_sample(param_shape).astype(np.float32)
#     v_val = np.random.random_sample(param_shape).astype(np.float32)
#     beta_val = np.random.random_sample(param_shape).astype(np.float32)
#     gamma_val = np.random.random_sample(param_shape).astype(np.float32)
#     backprop_val = np.random.random_sample(x_shape).astype(np.float32)
#     for use_gpu in [False, True]:
#       with self.test_session(use_gpu=use_gpu) as sess:
#         x = tf.constant(x_val, name="x")
#         m = tf.constant(m_val, name="m")
#         v = tf.constant(v_val, name="v")
#         beta = tf.constant(beta_val, name="beta")
#         gamma = tf.constant(gamma_val, name="gamma")
#         backprop = tf.constant(backprop_val, name="backprop")
#         epsilon = 0.001
#         for scale_after_normalization in [True, False]:
#           # _batch_norm_with_global_normalization_grad is deprecated in v9
#           tf.get_default_graph().graph_def_versions.producer = 8
#           dx, dm, dv, db, dg = (
#               gen_nn_ops._batch_norm_with_global_normalization_grad(
#               x, m, v, gamma, backprop, epsilon, scale_after_normalization))
#           on = self._opsBatchNorm(
#               x, m, v, beta, gamma, epsilon, scale_after_normalization, True)
#           odx, odm, odv, odb, odg = tf.gradients(
#               [on], [x, m, v, beta, gamma], [backprop])
#           if scale_after_normalization:
#             all_grads = sess.run([dx, dm, dv, db, dg, odx, odm, odv, odb, odg])
#             to_check = ["dx", "dm", "dv", "db", "dg"]
#           else:
#             all_grads = sess.run([dx, dm, dv, db, odx, odm, odv, odb])
#             to_check = ["dx", "dm", "dv", "db"]
#           for i, _ in enumerate(to_check):
#             self.assertAllClose(
#                 all_grads[i + len(to_check)], all_grads[i], atol=0.000001)

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


if __name__ == "__main__":
  tf.test.main()
