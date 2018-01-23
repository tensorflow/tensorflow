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
"""Tests for fused_batch_norm related functionality in tensorflow.ops.nn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad
from tensorflow.python.ops import nn_impl
from tensorflow.python.platform import test


class BatchNormalizationTest(test.TestCase):

  def _batch_norm(self, x, mean, var, offset, scale, epsilon):
    # We compute the batch norm manually in this function because
    # nn_impl.batch_normalization does not support float16 yet.
    # TODO(reedwm): Add float16 support to nn_impl.batch_normalization.
    inv = math_ops.rsqrt(var + epsilon) * scale
    y = math_ops.cast(x, scale.dtype) * inv + (offset - mean * inv)
    return math_ops.cast(y, x.dtype)

  def _inference_ref(self, x, scale, offset, mean, var, epsilon, data_format):
    if data_format not in ['NHWC', 'NCHW']:
      raise ValueError('data_format must be NCHW or NHWC, '
                       'got %s.' % data_format)
    if data_format == 'NCHW':
      x = array_ops.transpose(x, [0, 2, 3, 1])
    y = self._batch_norm(x, mean, var, offset, scale, epsilon)
    if data_format == 'NCHW':
      y = array_ops.transpose(y, [0, 3, 1, 2])
    return y.eval()

  def _test_inference(self,
                      x_shape,
                      x_dtype,
                      scale_shape,
                      scale_dtype,
                      use_gpu=True,
                      data_format='NHWC'):
    np.random.seed(1)
    x_val = np.random.random_sample(x_shape).astype(x_dtype)
    scale_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    offset_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    mean_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    var_val = np.random.random_sample(scale_shape).astype(scale_dtype)

    with self.test_session(use_gpu=use_gpu) as sess:
      x = constant_op.constant(x_val, name='x')
      scale = constant_op.constant(scale_val, name='scale')
      offset = constant_op.constant(offset_val, name='offset')
      mean = constant_op.constant(mean_val, name='mean')
      var = constant_op.constant(var_val, name='variance')
      epsilon = 0.001
      y, _, _ = nn_impl.fused_batch_norm(
          x,
          scale,
          offset,
          mean=mean,
          variance=var,
          epsilon=epsilon,
          data_format=data_format,
          is_training=False)
      y_val = sess.run(y)
      y_ref = self._inference_ref(x, scale, offset, mean, var, epsilon,
                                  data_format)
    # An atol value of 1e-3 is too small for float16's, because some adjacent
    # float16 values that y_val can take are greater than 1e-3 apart, e.g.
    # 2.16602 and 2.16797.
    atol = 2e-3 if x_dtype == np.float16 else 1e-3
    self.assertAllClose(y_ref, y_val, atol=atol)

  def _training_ref(self, x, scale, offset, epsilon, data_format):
    if data_format not in ['NHWC', 'NCHW']:
      raise ValueError('data_format must be NCHW or NHWC, '
                       'got %s.' % data_format)
    if data_format == 'NCHW':
      x = array_ops.transpose(x, [0, 2, 3, 1])
    mean, var = nn_impl.moments(
        math_ops.cast(x, scale.dtype), [0, 1, 2], keep_dims=False)
    y = self._batch_norm(x, mean, var, offset, scale, epsilon)
    if data_format == 'NCHW':
      y = array_ops.transpose(y, [0, 3, 1, 2])
    return y.eval(), mean.eval(), var.eval()

  def _test_training(self,
                     x_shape,
                     x_dtype,
                     scale_shape,
                     scale_dtype,
                     use_gpu=True,
                     data_format='NHWC'):
    np.random.seed(1)
    x_val = np.random.random_sample(x_shape).astype(x_dtype)
    scale_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    offset_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    with self.test_session(use_gpu=use_gpu) as sess:
      x = constant_op.constant(x_val, name='x')
      scale = constant_op.constant(scale_val, name='scale')
      offset = constant_op.constant(offset_val, name='offset')
      epsilon = 0.001
      y, mean, var = nn_impl.fused_batch_norm(
          x,
          scale,
          offset,
          epsilon=epsilon,
          data_format=data_format,
          is_training=True)
      y_val, mean_val, var_val = sess.run([y, mean, var])
      y_ref, mean_ref, var_ref = self._training_ref(x, scale, offset, epsilon,
                                                    data_format)
    y_atol = 2e-3 if x_dtype == np.float16 else 1e-3
    self.assertAllClose(y_ref, y_val, atol=y_atol)
    self.assertAllClose(mean_ref, mean_val, atol=1e-3)
    # This is for Bessel's correction. tf.nn.moments uses n, instead of n-1, as
    # the denominator in the formula to calculate variance, while
    # tf.nn.fused_batch_norm has Bessel's correction built in.
    sample_size = x_val.size / scale_val.size
    var_ref = var_ref * sample_size / (max(sample_size - 1.0, 1.0))
    self.assertAllClose(var_ref, var_val, atol=1e-3)

  def _compute_gradient_error_float16(self, x, x32, x_shape, y, y32, y_shape):
    """Computes the gradient error for float16 inputs and/or outputs.

    This returns the same value as gradient_checker.compute_gradient_error. The
    difference is that gradient_checker.compute_gradient_error does not
    numerically compute the gradients in a numerically stable way for float16
    tensors. To fix this, this function requires float32 versions of x and y to
    numerically compute the gradients, to compare with the float16 symbolically
    computed gradients.

    Args:
      x: The input tensor.
      x32: A float32 version of x.
      x_shape: The shape of x.
      y: The output tensor.
      y32: A float32 version of y. Must be calculated based on x32, not x.
      y_shape: The shape of y.

    Returns:
      The maximum error in between the two Jacobians, as in
      gradient_checker.compute_gradient_error.
    """
    x_init_val = np.random.random_sample(x_shape).astype(np.float16)
    x32_init_val = x_init_val.astype(np.float32)

    # TODO(reedwm): Do not perform the unnecessary computations in
    # compute_gradient, since they double the computation time of this function.
    theoretical_grad, _ = gradient_checker.compute_gradient(
        x, x_shape, y, y_shape, delta=1e-3, x_init_value=x_init_val)
    _, numerical_grad = gradient_checker.compute_gradient(
        x32, x_shape, y32, y_shape, delta=1e-3, x_init_value=x32_init_val)

    # If grad is empty, no error.
    if theoretical_grad.size == 0 and numerical_grad.size == 0:
      return 0
    return np.fabs(theoretical_grad - numerical_grad).max()

  def _test_gradient(self,
                     x_shape,
                     x_dtype,
                     scale_shape,
                     scale_dtype,
                     use_gpu=True,
                     data_format='NHWC',
                     is_training=True):
    np.random.seed(1)
    x_val = np.random.random_sample(x_shape).astype(x_dtype)
    scale_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    offset_val = np.random.random_sample(scale_shape).astype(scale_dtype)

    with self.test_session(use_gpu=use_gpu):
      x = constant_op.constant(x_val, name='x')
      scale = constant_op.constant(scale_val, name='scale')
      offset = constant_op.constant(offset_val, name='offset')
      if is_training:
        pop_mean = None
        pop_var = None
      else:
        pop_mean = np.random.random_sample(scale_shape).astype(scale_dtype)
        pop_var = np.random.random_sample(scale_shape).astype(scale_dtype)
      y, _, _ = nn_impl.fused_batch_norm(
          x,
          scale,
          offset,
          mean=pop_mean,
          variance=pop_var,
          data_format=data_format,
          is_training=is_training)
      if x_dtype != np.float16:
        err_x = gradient_checker.compute_gradient_error(x, x_shape, y, x_shape)
        err_scale = gradient_checker.compute_gradient_error(
            scale, scale_shape, y, x_shape)
        err_offset = gradient_checker.compute_gradient_error(
            offset, scale_shape, y, x_shape)
      else:
        x32 = constant_op.constant(x_val, name='x32', dtype=dtypes.float32)
        y32, _, _ = nn_impl.fused_batch_norm(
            x32,
            scale,
            offset,
            mean=pop_mean,
            variance=pop_var,
            data_format=data_format,
            is_training=is_training)
        err_x = self._compute_gradient_error_float16(x, x32, x_shape, y, y32,
                                                     x_shape)
        err_scale = self._compute_gradient_error_float16(
            scale, scale, scale_shape, y, y32, x_shape)
        err_offset = self._compute_gradient_error_float16(
            offset, offset, scale_shape, y, y32, x_shape)

    x_err_tolerance = 2e-3 if x_dtype == np.float16 else 1e-3
    scale_err_tolerance = 1e-3
    self.assertLess(err_x, x_err_tolerance)
    self.assertLess(err_scale, scale_err_tolerance)
    self.assertLess(err_offset, scale_err_tolerance)

  def _test_grad_grad(self,
                      x_shape,
                      x_dtype,
                      scale_shape,
                      scale_dtype,
                      use_gpu=True,
                      data_format='NHWC',
                      is_training=True,
                      err_tolerance=1e-3):
    np.random.seed(1)
    x_val = np.random.random_sample(x_shape).astype(x_dtype)
    grad_y_val = np.random.random_sample(x_shape).astype(x_dtype)
    scale_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    offset_val = np.random.random_sample(scale_shape).astype(scale_dtype)

    with self.test_session(use_gpu=use_gpu) as sess:
      x = constant_op.constant(x_val, name='x')
      grad_y = constant_op.constant(grad_y_val, name='grad_y')
      scale = constant_op.constant(scale_val, name='scale')
      offset = constant_op.constant(offset_val, name='offset')
      if is_training:
        pop_mean = None
        pop_var = None
      else:
        pop_mean = np.random.random_sample(scale_shape).astype(scale_dtype)
        pop_var = np.random.random_sample(scale_shape).astype(scale_dtype)
      y, _, _ = nn_impl.fused_batch_norm(
          x,
          scale,
          offset,
          mean=pop_mean,
          variance=pop_var,
          data_format=data_format,
          is_training=is_training)
      grad_x, grad_scale, grad_offset = gradients_impl.gradients(
          y, [x, scale, offset], grad_y)

      if is_training:
        epsilon = y.op.get_attr('epsilon')
        data_format = y.op.get_attr('data_format')
        grad_vals = sess.run([grad_x, grad_scale, grad_offset])
        grad_internal = nn_grad._BatchNormGrad(grad_y, x, scale, pop_mean, pop_var, epsilon, data_format)
        grad_internal_vals = sess.run(list(grad_internal))
        for grad_val, grad_internal_val in zip(grad_vals, grad_internal_vals):
          self.assertAllClose(grad_val, grad_internal_val, atol=err_tolerance)

      if x_dtype != np.float16:
        err_grad_grad_y_1 = gradient_checker.compute_gradient_error(
            grad_y, x_shape, grad_x, x_shape)
        err_grad_grad_y_2 = gradient_checker.compute_gradient_error(
            grad_y, x_shape, grad_scale, scale_shape)
        err_grad_grad_y_3 = gradient_checker.compute_gradient_error(
            grad_y, x_shape, grad_offset, scale_shape)
        # In freeze mode, grad_x is not a function of x.
        if is_training:
          err_grad_x_1 = gradient_checker.compute_gradient_error(
              x, x_shape, grad_x, x_shape)
        err_grad_x_2 = gradient_checker.compute_gradient_error(
            x, x_shape, grad_scale, scale_shape)

        err_grad_scale = gradient_checker.compute_gradient_error(
            scale, scale_shape, grad_x, x_shape)
      else:
        x32 = constant_op.constant(x_val, dtype=dtypes.float32, name='x32')
        grad_y32 = constant_op.constant(
            grad_y_val, dtype=dtypes.float32, name='grad_y32')
        y32, _, _ = nn_impl.fused_batch_norm(
            x32,
            scale,
            offset,
            mean=pop_mean,
            variance=pop_var,
            data_format=data_format,
            is_training=is_training)
        grad_x32, grad_scale32, grad_offset32 = gradients_impl.gradients(
            y32, [x32, scale, offset], grad_y32)
        err_grad_grad_y_1 = self._compute_gradient_error_float16(
            grad_y, grad_y32, x_shape, grad_x, grad_x32, x_shape)
        err_grad_grad_y_2 = self._compute_gradient_error_float16(
            grad_y, grad_y32, x_shape, grad_scale, grad_scale32, scale_shape)
        err_grad_grad_y_3 = self._compute_gradient_error_float16(
            grad_y, grad_y32, x_shape, grad_offset, grad_offset32, scale_shape)
        # In freeze mode, grad_x is not a function of x.
        if is_training:
          err_grad_x_1 = self._compute_gradient_error_float16(
              x, x32, x_shape, grad_x, grad_x32, x_shape)
        err_grad_x_2 = self._compute_gradient_error_float16(
            x, x32, x_shape, grad_scale, grad_scale32, scale_shape)

        err_grad_scale = self._compute_gradient_error_float16(
            scale, scale, scale_shape, grad_x, grad_x32, x_shape)

    self.assertLess(err_grad_grad_y_1, err_tolerance)
    self.assertLess(err_grad_grad_y_2, err_tolerance)
    self.assertLess(err_grad_grad_y_3, err_tolerance)
    if is_training:
      self.assertLess(err_grad_x_1, err_tolerance)
    self.assertLess(err_grad_x_2, err_tolerance)
    self.assertLess(err_grad_scale, err_tolerance)

  def testInferenceShape1(self):
    x_shape = [1, 1, 6, 1]
    for dtype in [np.float16, np.float32]:
      if test.is_gpu_available(cuda_only=True):
        self._test_inference(
            x_shape, dtype, [1], np.float32, use_gpu=True, data_format='NHWC')
        self._test_inference(
            x_shape, dtype, [1], np.float32, use_gpu=True, data_format='NCHW')
      self._test_inference(
          x_shape, dtype, [1], np.float32, use_gpu=False, data_format='NHWC')

  def testInferenceShape2(self):
    x_shape = [1, 1, 6, 2]
    if test.is_gpu_available(cuda_only=True):
      for dtype in [np.float16, np.float32]:
        self._test_inference(
            x_shape, dtype, [2], np.float32, use_gpu=True, data_format='NHWC')
        self._test_inference(
            x_shape, dtype, [2], np.float32, use_gpu=False, data_format='NHWC')

  def testInferenceShape3(self):
    x_shape = [1, 2, 1, 6]
    if test.is_gpu_available(cuda_only=True):
      for dtype in [np.float16, np.float32]:
        self._test_inference(
            x_shape, dtype, [2], np.float32, use_gpu=True, data_format='NCHW')

  def testInferenceShape4(self):
    x_shape = [27, 131, 127, 6]
    for dtype in [np.float16, np.float32]:
      if test.is_gpu_available(cuda_only=True):
        self._test_inference(
            x_shape, dtype, [131], np.float32, use_gpu=True, data_format='NCHW')
        self._test_inference(
            x_shape, dtype, [6], np.float32, use_gpu=True, data_format='NHWC')
      self._test_inference(
          x_shape, dtype, [6], np.float32, use_gpu=False, data_format='NHWC')

  def testInferenceShape5(self):
    x_shape = [0, 131, 127, 6]
    for dtype in [np.float16, np.float32]:
      if test.is_gpu_available(cuda_only=True):
        self._test_inference(
            x_shape, dtype, [131], np.float32, use_gpu=True, data_format='NCHW')
        self._test_inference(
            x_shape, dtype, [6], np.float32, use_gpu=True, data_format='NHWC')
      self._test_inference(
          x_shape, dtype, [6], np.float32, use_gpu=False, data_format='NHWC')

  def testTrainingShape1(self):
    x_shape = [1, 1, 6, 1]
    for dtype in [np.float16, np.float32]:
      if test.is_gpu_available(cuda_only=True):
        self._test_training(
            x_shape, dtype, [1], np.float32, use_gpu=True, data_format='NHWC')
        self._test_training(
            x_shape, dtype, [1], np.float32, use_gpu=True, data_format='NCHW')
      self._test_training(
          x_shape, dtype, [1], np.float32, use_gpu=False, data_format='NHWC')

  def testTrainingShape2(self):
    x_shape = [1, 1, 6, 2]
    for dtype in [np.float16, np.float32]:
      if test.is_gpu_available(cuda_only=True):
        self._test_training(
            x_shape, dtype, [2], np.float32, use_gpu=True, data_format='NHWC')
      self._test_training(
          x_shape, dtype, [2], np.float32, use_gpu=False, data_format='NHWC')

  def testTrainingShape3(self):
    x_shape = [1, 2, 1, 6]
    if test.is_gpu_available(cuda_only=True):
      for dtype in [np.float16, np.float32]:
        self._test_training(
            x_shape, dtype, [2], np.float32, use_gpu=True, data_format='NCHW')

  def testTrainingShape4(self):
    x_shape = [27, 131, 127, 6]
    for dtype in [np.float16, np.float32]:
      if test.is_gpu_available(cuda_only=True):
        self._test_training(
            x_shape, dtype, [131], np.float32, use_gpu=True, data_format='NCHW')
        self._test_training(
            x_shape, dtype, [6], np.float32, use_gpu=True, data_format='NHWC')
      self._test_training(
          x_shape, dtype, [6], np.float32, use_gpu=False, data_format='NHWC')

  def testTrainingShape5(self):
    x_shape = [0, 131, 127, 6]
    for dtype in [np.float16, np.float32]:
      if test.is_gpu_available(cuda_only=True):
        self._test_training(
            x_shape, dtype, [131], np.float32, use_gpu=True, data_format='NCHW')
        self._test_training(
            x_shape, dtype, [6], np.float32, use_gpu=True, data_format='NHWC')
      self._test_training(
          x_shape, dtype, [6], np.float32, use_gpu=False, data_format='NHWC')

  def testBatchNormGradShape1(self):
    for is_training in [True, False]:
      x_shape = [1, 1, 6, 1]
      for dtype in [np.float16, np.float32]:
        if test.is_gpu_available(cuda_only=True):
          self._test_gradient(
              x_shape,
              dtype, [1],
              np.float32,
              use_gpu=True,
              data_format='NHWC',
              is_training=is_training)
          self._test_gradient(
              x_shape,
              dtype, [1],
              np.float32,
              use_gpu=True,
              data_format='NCHW',
              is_training=is_training)
        self._test_gradient(
            x_shape,
            dtype, [1],
            np.float32,
            use_gpu=False,
            data_format='NHWC',
            is_training=is_training)

  def testBatchNormGradShape2(self):
    for is_training in [True, False]:
      x_shape = [1, 1, 6, 2]
      for dtype in [np.float16, np.float32]:
        if test.is_gpu_available(cuda_only=True):
          self._test_gradient(
              x_shape,
              dtype, [2],
              np.float32,
              use_gpu=True,
              data_format='NHWC',
              is_training=is_training)
        self._test_gradient(
            x_shape,
            dtype, [2],
            np.float32,
            use_gpu=False,
            data_format='NHWC',
            is_training=is_training)

  def testBatchNormGradShape3(self):
    for is_training in [True, False]:
      x_shape = [1, 2, 1, 6]
      if test.is_gpu_available(cuda_only=True):
        for dtype in [np.float16, np.float32]:
          self._test_gradient(
              x_shape,
              dtype, [2],
              np.float32,
              use_gpu=True,
              data_format='NCHW',
              is_training=is_training)

  def testBatchNormGradShape4(self):
    for is_training in [True, False]:
      x_shape = [5, 7, 11, 4]
      for dtype in [np.float16, np.float32]:
        if test.is_gpu_available(cuda_only=True):
          self._test_gradient(
              x_shape,
              dtype, [7],
              np.float32,
              use_gpu=True,
              data_format='NCHW',
              is_training=is_training)
          self._test_gradient(
              x_shape,
              dtype, [4],
              np.float32,
              use_gpu=True,
              data_format='NHWC',
              is_training=is_training)
        self._test_gradient(
            x_shape,
            dtype, [4],
            np.float32,
            use_gpu=False,
            data_format='NHWC',
            is_training=is_training)

  def testBatchNormGradShape5(self):
    for is_training in [True, False]:
      x_shape = [0, 7, 11, 4]
      for dtype in [np.float16, np.float32]:
        if test.is_gpu_available(cuda_only=True):
          self._test_gradient(
              x_shape,
              dtype, [7],
              np.float32,
              use_gpu=True,
              data_format='NCHW',
              is_training=is_training)
          self._test_gradient(
              x_shape,
              dtype, [4],
              np.float32,
              use_gpu=True,
              data_format='NHWC',
              is_training=is_training)
        self._test_gradient(
            x_shape,
            dtype, [4],
            np.float32,
            use_gpu=False,
            data_format='NHWC',
            is_training=is_training)

  def _testBatchNormGradGrad(self, config):
    shape = config['shape']
    err_tolerance = config['err_tolerance']
    dtype = config['dtype']
    for is_training in [True, False]:
      if test.is_gpu_available(cuda_only=True):
        self._test_grad_grad(
            shape,
            dtype, [shape[3]],
            np.float32,
            use_gpu=True,
            data_format='NHWC',
            is_training=is_training,
            err_tolerance=err_tolerance)
        self._test_grad_grad(
            shape,
            dtype, [shape[1]],
            np.float32,
            use_gpu=True,
            data_format='NCHW',
            is_training=is_training,
            err_tolerance=err_tolerance)
      self._test_grad_grad(
          shape,
          dtype, [shape[3]],
          np.float32,
          use_gpu=False,
          data_format='NHWC',
          is_training=is_training,
          err_tolerance=err_tolerance)

  def testBatchNormGradGradConfig1(self):
    config = {
        'shape': [2, 3, 4, 5],
        'err_tolerance': 1e-2,
        'dtype': np.float32,
    }
    self._testBatchNormGradGrad(config)

  def testBatchNormGradGradConfig2(self):
    config = {
        'shape': [2, 3, 2, 2],
        'err_tolerance': 1e-3,
        'dtype': np.float32,
    }
    self._testBatchNormGradGrad(config)

  def testBatchNormGradGradConfig3(self):
    config = {
        'shape': [2, 3, 4, 5],
        'err_tolerance': 1e-2,
        'dtype': np.float16,
    }
    self._testBatchNormGradGrad(config)

  def testBatchNormGradGradConfig4(self):
    config = {
        'shape': [2, 3, 2, 2],
        'err_tolerance': 2e-3,
        'dtype': np.float16,
    }
    self._testBatchNormGradGrad(config)


if __name__ == '__main__':
  test.main()
