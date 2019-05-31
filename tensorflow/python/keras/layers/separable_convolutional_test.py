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
"""Tests for separable convolutional layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes
class SeparableConv1DTest(keras_parameterized.TestCase):

  def _run_test(self, kwargs):
    num_samples = 2
    stack_size = 3
    length = 7

    with self.cached_session(use_gpu=True):
      testing_utils.layer_test(
          keras.layers.SeparableConv1D,
          kwargs=kwargs,
          input_shape=(num_samples, length, stack_size))

  @parameterized.named_parameters(
      ('padding_valid', {'padding': 'valid'}),
      ('padding_same', {'padding': 'same'}),
      ('padding_same_dilation_2', {'padding': 'same', 'dilation_rate': 2}),
      ('padding_causal', {'padding': 'causal'}),
      ('strides', {'strides': 2}),
      ('dilation_rate', {'dilation_rate': 2}),
      ('depth_multiplier', {'depth_multiplier': 2}),
  )
  def test_separable_conv1d(self, kwargs):
    kwargs['filters'] = 2
    kwargs['kernel_size'] = 3
    self._run_test(kwargs)

  def test_separable_conv1d_regularizers(self):
    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'depthwise_regularizer': 'l2',
        'pointwise_regularizer': 'l2',
        'bias_regularizer': 'l2',
        'activity_regularizer': 'l2',
        'strides': 1
    }
    with self.cached_session(use_gpu=True):
      layer = keras.layers.SeparableConv1D(**kwargs)
      layer.build((None, 5, 2))
      self.assertEqual(len(layer.losses), 3)
      layer(keras.backend.variable(np.ones((1, 5, 2))))
      self.assertEqual(len(layer.losses), 4)

  def test_separable_conv1d_constraints(self):
    d_constraint = lambda x: x
    p_constraint = lambda x: x
    b_constraint = lambda x: x

    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'pointwise_constraint': p_constraint,
        'depthwise_constraint': d_constraint,
        'bias_constraint': b_constraint,
        'strides': 1
    }
    with self.cached_session(use_gpu=True):
      layer = keras.layers.SeparableConv1D(**kwargs)
      layer.build((None, 5, 2))
      self.assertEqual(layer.depthwise_kernel.constraint, d_constraint)
      self.assertEqual(layer.pointwise_kernel.constraint, p_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)


@keras_parameterized.run_all_keras_modes
class SeparableConv2DTest(keras_parameterized.TestCase):

  def _run_test(self, kwargs):
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6

    with self.cached_session(use_gpu=True):
      testing_utils.layer_test(
          keras.layers.SeparableConv2D,
          kwargs=kwargs,
          input_shape=(num_samples, num_row, num_col, stack_size))

  @parameterized.named_parameters(
      ('padding_valid', {'padding': 'valid'}),
      ('padding_same', {'padding': 'same'}),
      ('padding_same_dilation_2', {'padding': 'same', 'dilation_rate': 2}),
      ('strides', {'strides': 2}),
      # Only runs on GPU with CUDA, channels_first is not supported on CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      ('data_format', {'data_format': 'channels_first'}),
      ('dilation_rate', {'dilation_rate': 2}),
      ('depth_multiplier', {'depth_multiplier': 2}),
  )
  def test_separable_conv2d(self, kwargs):
    kwargs['filters'] = 2
    kwargs['kernel_size'] = 3
    if 'data_format' not in kwargs or test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs)

  def test_separable_conv2d_regularizers(self):
    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'depthwise_regularizer': 'l2',
        'pointwise_regularizer': 'l2',
        'bias_regularizer': 'l2',
        'activity_regularizer': 'l2',
        'strides': 1
    }
    with self.cached_session(use_gpu=True):
      layer = keras.layers.SeparableConv2D(**kwargs)
      layer.build((None, 5, 5, 2))
      self.assertEqual(len(layer.losses), 3)
      layer(keras.backend.variable(np.ones((1, 5, 5, 2))))
      self.assertEqual(len(layer.losses), 4)

  def test_separable_conv2d_constraints(self):
    d_constraint = lambda x: x
    p_constraint = lambda x: x
    b_constraint = lambda x: x

    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'pointwise_constraint': p_constraint,
        'depthwise_constraint': d_constraint,
        'bias_constraint': b_constraint,
        'strides': 1
    }
    with self.cached_session(use_gpu=True):
      layer = keras.layers.SeparableConv2D(**kwargs)
      layer.build((None, 5, 5, 2))
      self.assertEqual(layer.depthwise_kernel.constraint, d_constraint)
      self.assertEqual(layer.pointwise_kernel.constraint, p_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)
