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
"""Functional tests for fused batch norm operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests.xla_test import XLATestCase
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class FusedBatchNormTest(XLATestCase):

  def _reference_training(self, x, scale, offset, epsilon, data_format):
    if data_format != "NHWC":
      raise ValueError("data_format must be NHWC, got %s." % data_format)
    x_square = x * x
    x_square_sum = np.sum(x_square, (0, 1, 2))
    x_sum = np.sum(x, axis=(0, 1, 2))
    element_count = np.size(x) / int(np.shape(x)[0])
    mean = x_sum / element_count
    var = x_square_sum / element_count - mean * mean
    normalized = (x - mean) / np.sqrt(var + epsilon)
    return (normalized * scale + offset), mean, var

  def _reference_grad(self, x, grad_y, scale, mean, var, epsilon, data_format):
    if data_format != "NHWC":
      raise ValueError("data_format must be NHWC, got %s." % data_format)
    grad_x = grad_y * scale * np.sqrt(var + epsilon)
    grad_scale = np.sum(
        grad_y * (x - mean) * np.sqrt(var + epsilon), axis=(0, 1, 2))
    grad_offset = np.sum(grad_y, axis=(0, 1, 2))
    return grad_x, grad_scale, grad_offset

  def testBasic(self):
    x_shape = [2, 2, 6, 2]
    scale_shape = [2]
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    scale_val = np.random.random_sample(scale_shape).astype(np.float32)

    offset_val = np.random.random_sample(scale_shape).astype(np.float32)
    mean_val = np.random.random_sample(scale_shape).astype(np.float32)
    var_val = np.random.random_sample(scale_shape).astype(np.float32)
    data_format = "NHWC"
    with self.test_session() as sess, self.test_scope():
      # To avoid constant folding
      t_val = array_ops.placeholder(np.float32, shape=x_shape, name="x")
      scale = array_ops.placeholder(np.float32, shape=[2], name="scale")
      offset = array_ops.placeholder(np.float32, shape=[2], name="offset")
      epsilon = 0.001
      y, mean, var = nn.fused_batch_norm(
          t_val,
          scale,
          offset,
          mean=None,
          variance=None,
          epsilon=epsilon,
          data_format=data_format,
          is_training=True)
      y_val, mean_val, var_val = sess.run(
          [y, mean, var], {t_val: x_val,
                           scale: scale_val,
                           offset: offset_val})
      y_ref, mean_ref, var_ref = self._reference_training(
          x_val, scale_val, offset_val, epsilon, data_format)
      self.assertAllClose(mean_val, mean_ref, atol=1e-3)
      self.assertAllClose(y_val, y_ref, atol=1e-3)
      self.assertAllClose(var_val, var_ref, atol=1e-3)

  def testGradient(self):
    # TODO(b/64270657): Use gradient_checker here in addition to comparing with
    # this reference implementation.
    x_shape = [2, 2, 6, 2]
    scale_shape = [2]
    grad_val = np.random.random_sample(x_shape).astype(np.float32)
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    scale_val = np.random.random_sample(scale_shape).astype(np.float32)
    mean_val = np.random.random_sample(scale_shape).astype(np.float32)
    var_val = np.random.random_sample(scale_shape).astype(np.float32)
    epsilon = 0.001

    with self.test_session() as sess, self.test_scope():
      grad = array_ops.placeholder(np.float32, shape=x_shape, name="grad")
      x = array_ops.placeholder(np.float32, shape=x_shape, name="x")
      mean = array_ops.placeholder(np.float32, shape=scale_shape, name="mean")
      var = array_ops.placeholder(np.float32, shape=scale_shape, name="var")
      scale = array_ops.placeholder(np.float32, shape=scale_shape, name="scale")
      grad_x, grad_scale, grad_offset, _, _ = gen_nn_ops.fused_batch_norm_grad(
          grad, x, scale, mean, var, data_format="NHWC")
      grad_x_val, grad_scale_val, grad_offset_val = sess.run(
          [grad_x, grad_scale, grad_offset], {
              grad: grad_val,
              x: x_val,
              mean: mean_val,
              var: var_val,
              scale: scale_val
          })

      grad_x_ref, grad_scale_ref, grad_offset_ref = self._reference_grad(
          x_val, grad_val, scale_val, mean_val, var_val, epsilon, "NHWC")

      self.assertAllClose(grad_x_val, grad_x_ref, atol=1e-2)
      self.assertAllClose(grad_scale_val, grad_scale_ref, atol=1e-2)
      self.assertAllClose(grad_offset_val, grad_offset_ref, atol=1e-3)

if __name__ == "__main__":
  test.main()
