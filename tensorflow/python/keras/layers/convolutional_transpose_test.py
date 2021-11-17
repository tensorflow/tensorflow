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
"""Tests for convolutional transpose layers."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes
class Conv2DTransposeTest(keras_parameterized.TestCase):

  def _run_test(self, kwargs):
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6

    with self.cached_session():
      testing_utils.layer_test(
          keras.layers.Conv2DTranspose,
          kwargs=kwargs,
          input_shape=(num_samples, num_row, num_col, stack_size))

  @parameterized.named_parameters(
      ('padding_valid', {'padding': 'valid'}),
      ('padding_same', {'padding': 'same'}),
      ('strides', {'strides': (2, 2)}),
      # Only runs on GPU with CUDA, channels_first is not supported on CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      ('data_format', {'data_format': 'channels_first'}),
      ('strides_output_padding', {'strides': (2, 2), 'output_padding': (1, 1)}),
  )
  def test_conv2d_transpose(self, kwargs):
    kwargs['filters'] = 2
    kwargs['kernel_size'] = (3, 3)
    if 'data_format' not in kwargs or test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs)

  def test_conv2d_transpose_regularizers(self):
    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'kernel_regularizer': 'l2',
        'bias_regularizer': 'l2',
        'activity_regularizer': 'l2',
        'strides': 1
    }
    with self.cached_session():
      layer = keras.layers.Conv2DTranspose(**kwargs)
      layer.build((None, 5, 5, 2))
      self.assertEqual(len(layer.losses), 2)
      layer(keras.backend.variable(np.ones((1, 5, 5, 2))))
      self.assertEqual(len(layer.losses), 3)

  def test_conv2d_transpose_constraints(self):
    k_constraint = lambda x: x
    b_constraint = lambda x: x

    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'kernel_constraint': k_constraint,
        'bias_constraint': b_constraint,
        'strides': 1
    }
    with self.cached_session():
      layer = keras.layers.Conv2DTranspose(**kwargs)
      layer.build((None, 5, 5, 2))
      self.assertEqual(layer.kernel.constraint, k_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)

  def test_conv2d_transpose_dilation(self):
    testing_utils.layer_test(keras.layers.Conv2DTranspose,
                             kwargs={'filters': 2,
                                     'kernel_size': 3,
                                     'padding': 'same',
                                     'data_format': 'channels_last',
                                     'dilation_rate': (2, 2)},
                             input_shape=(2, 5, 6, 3))

    input_data = np.arange(48).reshape((1, 4, 4, 3)).astype(np.float32)
    # pylint: disable=too-many-function-args
    expected_output = np.float32([
        [192, 228, 192, 228],
        [336, 372, 336, 372],
        [192, 228, 192, 228],
        [336, 372, 336, 372]
    ]).reshape((1, 4, 4, 1))
    testing_utils.layer_test(keras.layers.Conv2DTranspose,
                             input_data=input_data,
                             kwargs={'filters': 1,
                                     'kernel_size': 3,
                                     'padding': 'same',
                                     'data_format': 'channels_last',
                                     'dilation_rate': (2, 2),
                                     'kernel_initializer': 'ones'},
                             expected_output=expected_output)


@keras_parameterized.run_all_keras_modes
class Conv3DTransposeTest(keras_parameterized.TestCase):

  def _run_test(self, kwargs):
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6
    depth = 5

    with self.cached_session():
      testing_utils.layer_test(
          keras.layers.Conv3DTranspose,
          kwargs=kwargs,
          input_shape=(num_samples, depth, num_row, num_col, stack_size))

  @parameterized.named_parameters(
      ('padding_valid', {'padding': 'valid'}),
      ('padding_same', {'padding': 'same'}),
      ('strides', {'strides': (2, 2, 2)}),
      # Only runs on GPU with CUDA, channels_first is not supported on CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      ('data_format', {'data_format': 'channels_first'}),
      ('strides_output_padding', {'strides': (2, 2, 2),
                                  'output_padding': (1, 1, 1)}),
  )
  def test_conv3d_transpose(self, kwargs):
    kwargs['filters'] = 2
    kwargs['kernel_size'] = (3, 3, 3)
    if 'data_format' not in kwargs or test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs)

  def test_conv3d_transpose_regularizers(self):
    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'kernel_regularizer': 'l2',
        'bias_regularizer': 'l2',
        'activity_regularizer': 'l2',
        'strides': 1
    }
    with self.cached_session():
      layer = keras.layers.Conv3DTranspose(**kwargs)
      layer.build((None, 5, 5, 5, 2))
      self.assertEqual(len(layer.losses), 2)
      layer(keras.backend.variable(np.ones((1, 5, 5, 5, 2))))
      self.assertEqual(len(layer.losses), 3)

  def test_conv3d_transpose_constraints(self):
    k_constraint = lambda x: x
    b_constraint = lambda x: x

    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'kernel_constraint': k_constraint,
        'bias_constraint': b_constraint,
        'strides': 1
    }
    with self.cached_session():
      layer = keras.layers.Conv3DTranspose(**kwargs)
      layer.build((None, 5, 5, 5, 2))
      self.assertEqual(layer.kernel.constraint, k_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)

  def test_conv3d_transpose_dynamic_shape(self):
    input_data = np.random.random((1, 3, 3, 3, 3)).astype(np.float32)
    with self.cached_session():
      # Won't raise error here.
      testing_utils.layer_test(
          keras.layers.Conv3DTranspose,
          kwargs={
              'data_format': 'channels_last',
              'filters': 3,
              'kernel_size': 3
          },
          input_shape=(None, None, None, None, 3),
          input_data=input_data)
      if test.is_gpu_available(cuda_only=True):
        testing_utils.layer_test(
            keras.layers.Conv3DTranspose,
            kwargs={
                'data_format': 'channels_first',
                'filters': 3,
                'kernel_size': 3
            },
            input_shape=(None, 3, None, None, None),
            input_data=input_data)

if __name__ == '__main__':
  test.main()
