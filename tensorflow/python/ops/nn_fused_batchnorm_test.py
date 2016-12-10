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
import tensorflow as tf


class BatchNormalizationTest(tf.test.TestCase):

  def _inference_ref(self, x, scale, offset, mean, var, epsilon, data_format):
    if data_format not in ['NHWC', 'NCHW']:
      raise ValueError('data_format must be NCHW or NHWC, '
                       'got %s.' % data_format)
    if data_format == 'NCHW':
      x = tf.transpose(x, [0, 2, 3, 1])
    y = tf.nn.batch_normalization(x, mean, var, offset, scale, epsilon)
    if data_format == 'NCHW':
      y = tf.transpose(y, [0, 3, 1, 2])
    return y.eval()

  def _test_inference(self,
                      x_shape,
                      scale_shape,
                      use_gpu=True,
                      data_format='NHWC'):
    np.random.seed(1)
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    scale_val = np.random.random_sample(scale_shape).astype(np.float32)
    offset_val = np.random.random_sample(scale_shape).astype(np.float32)
    mean_val = np.random.random_sample(scale_shape).astype(np.float32)
    var_val = np.random.random_sample(scale_shape).astype(np.float32)

    with self.test_session(use_gpu=use_gpu) as sess:
      x = tf.constant(x_val, name='x')
      scale = tf.constant(scale_val, name='scale')
      offset = tf.constant(offset_val, name='offset')
      mean = tf.constant(mean_val, name='mean')
      var = tf.constant(var_val, name='variance')
      epsilon = 0.001
      y, _, _ = tf.nn.fused_batch_norm(
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
    self.assertAllClose(y_ref, y_val, atol=1e-3)

  def _training_ref(self, x, scale, offset, epsilon, data_format):
    if data_format not in ['NHWC', 'NCHW']:
      raise ValueError('data_format must be NCHW or NHWC, '
                       'got %s.' % data_format)
    if data_format == 'NCHW':
      x = tf.transpose(x, [0, 2, 3, 1])
    mean, var = tf.nn.moments(x, [0, 1, 2], keep_dims=False)
    y = tf.nn.batch_normalization(x, mean, var, offset, scale, epsilon)
    if data_format == 'NCHW':
      y = tf.transpose(y, [0, 3, 1, 2])
    return y.eval(), mean.eval(), var.eval()

  def _test_training(self,
                     x_shape,
                     scale_shape,
                     use_gpu=True,
                     data_format='NHWC'):
    np.random.seed(1)
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    scale_val = np.random.random_sample(scale_shape).astype(np.float32)
    offset_val = np.random.random_sample(scale_shape).astype(np.float32)
    with self.test_session(use_gpu=use_gpu) as sess:
      x = tf.constant(x_val, name='x')
      scale = tf.constant(scale_val, name='scale')
      offset = tf.constant(offset_val, name='offset')
      epsilon = 0.001
      y, mean, var = tf.nn.fused_batch_norm(
          x,
          scale,
          offset,
          epsilon=epsilon,
          data_format=data_format,
          is_training=True)
      y_val, mean_val, var_val = sess.run([y, mean, var])
      y_ref, mean_ref, var_ref = self._training_ref(x, scale, offset, epsilon,
                                                    data_format)
    self.assertAllClose(y_ref, y_val, atol=1e-3)
    self.assertAllClose(mean_ref, mean_val, atol=1e-3)
    # This is for Bessel's correction. tf.nn.moments uses n, instead of n-1, as
    # the denominator in the formula to calculate variance, while
    # tf.nn.fused_batch_norm has Bessel's correction built in.
    sample_size = x_val.size / scale_val.size
    var_ref = var_ref * sample_size / (max(sample_size - 1.0, 1.0))
    self.assertAllClose(var_ref, var_val, atol=1e-3)

  def _test_gradient(self,
                     x_shape,
                     scale_shape,
                     use_gpu=True,
                     data_format='NHWC'):
    np.random.seed(1)
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    scale_val = np.random.random_sample(scale_shape).astype(np.float32)
    offset_val = np.random.random_sample(scale_shape).astype(np.float32)

    with self.test_session(use_gpu=use_gpu):
      x = tf.constant(x_val, name='x')
      scale = tf.constant(scale_val, name='scale')
      offset = tf.constant(offset_val, name='offset')
      y, _, _ = tf.nn.fused_batch_norm(
          x, scale, offset, data_format=data_format)
      err_x = tf.test.compute_gradient_error(x, x_shape, y, x_shape)
      err_scale = tf.test.compute_gradient_error(scale, scale_shape, y, x_shape)
      err_offset = tf.test.compute_gradient_error(offset, scale_shape, y,
                                                  x_shape)
    err_tolerance = 1e-3
    self.assertLess(err_x, err_tolerance)
    self.assertLess(err_scale, err_tolerance)
    self.assertLess(err_offset, err_tolerance)

  def testInference(self):
    x_shape = [1, 1, 6, 1]
    if tf.test.is_gpu_available(cuda_only=True):
      self._test_inference(x_shape, [1], use_gpu=True, data_format='NHWC')
      self._test_inference(x_shape, [1], use_gpu=True, data_format='NCHW')
    self._test_inference(x_shape, [1], use_gpu=False, data_format='NHWC')

    x_shape = [1, 1, 6, 2]
    if tf.test.is_gpu_available(cuda_only=True):
      self._test_inference(x_shape, [2], use_gpu=True, data_format='NHWC')
    self._test_inference(x_shape, [2], use_gpu=False, data_format='NHWC')

    x_shape = [1, 2, 1, 6]
    if tf.test.is_gpu_available(cuda_only=True):
      self._test_inference(x_shape, [2], use_gpu=True, data_format='NCHW')

    x_shape = [27, 131, 127, 6]
    if tf.test.is_gpu_available(cuda_only=True):
      self._test_inference(x_shape, [131], use_gpu=True, data_format='NCHW')
      self._test_inference(x_shape, [6], use_gpu=True, data_format='NHWC')
    self._test_inference(x_shape, [6], use_gpu=False, data_format='NHWC')

  def testTraining(self):
    x_shape = [1, 1, 6, 1]
    if tf.test.is_gpu_available(cuda_only=True):
      self._test_training(x_shape, [1], use_gpu=True, data_format='NHWC')
      self._test_training(x_shape, [1], use_gpu=True, data_format='NCHW')
    self._test_training(x_shape, [1], use_gpu=False, data_format='NHWC')

    x_shape = [1, 1, 6, 2]
    if tf.test.is_gpu_available(cuda_only=True):
      self._test_training(x_shape, [2], use_gpu=True, data_format='NHWC')
    self._test_training(x_shape, [2], use_gpu=False, data_format='NHWC')

    x_shape = [1, 2, 1, 6]
    if tf.test.is_gpu_available(cuda_only=True):
      self._test_training(x_shape, [2], use_gpu=True, data_format='NCHW')

    x_shape = [27, 131, 127, 6]
    if tf.test.is_gpu_available(cuda_only=True):
      self._test_training(x_shape, [131], use_gpu=True, data_format='NCHW')
      self._test_training(x_shape, [6], use_gpu=True, data_format='NHWC')
    self._test_training(x_shape, [6], use_gpu=False, data_format='NHWC')

  def testBatchNormGrad(self):
    x_shape = [1, 1, 6, 1]
    if tf.test.is_gpu_available(cuda_only=True):
      self._test_gradient(x_shape, [1], use_gpu=True, data_format='NHWC')
      self._test_gradient(x_shape, [1], use_gpu=True, data_format='NCHW')
    self._test_gradient(x_shape, [1], use_gpu=False, data_format='NHWC')

    x_shape = [1, 1, 6, 2]
    if tf.test.is_gpu_available(cuda_only=True):
      self._test_gradient(x_shape, [2], use_gpu=True, data_format='NHWC')
    self._test_gradient(x_shape, [2], use_gpu=False, data_format='NHWC')

    x_shape = [1, 2, 1, 6]
    if tf.test.is_gpu_available(cuda_only=True):
      self._test_gradient(x_shape, [2], use_gpu=True, data_format='NCHW')

    x_shape = [7, 9, 13, 6]
    if tf.test.is_gpu_available(cuda_only=True):
      self._test_gradient(x_shape, [9], use_gpu=True, data_format='NCHW')
      self._test_gradient(x_shape, [6], use_gpu=True, data_format='NHWC')
    self._test_gradient(x_shape, [6], use_gpu=False, data_format='NHWC')


if __name__ == '__main__':
  tf.test.main()
