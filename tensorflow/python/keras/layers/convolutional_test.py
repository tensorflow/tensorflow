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
"""Tests for convolutional layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes
class Conv1DTest(keras_parameterized.TestCase):

  def _run_test(self, kwargs, expected_output_shape):
    num_samples = 2
    stack_size = 3
    length = 7

    with self.cached_session(use_gpu=True):
      testing_utils.layer_test(
          keras.layers.Conv1D,
          kwargs=kwargs,
          input_shape=(num_samples, length, stack_size),
          expected_output_shape=expected_output_shape)

  @parameterized.named_parameters(
      ('padding_valid', {
          'padding': 'valid'
      }, (None, 5, 2)),
      ('padding_same', {
          'padding': 'same'
      }, (None, 7, 2)),
      ('padding_same_dilation_2', {
          'padding': 'same',
          'dilation_rate': 2
      }, (None, 7, 2)),
      ('padding_same_dilation_3', {
          'padding': 'same',
          'dilation_rate': 3
      }, (None, 7, 2)),
      ('padding_causal', {
          'padding': 'causal'
      }, (None, 7, 2)),
      ('strides', {
          'strides': 2
      }, (None, 3, 2)),
      ('dilation_rate', {
          'dilation_rate': 2
      }, (None, 3, 2)),
      # Only runs on GPU with CUDA, groups are not supported on CPU.
      # https://github.com/tensorflow/tensorflow/issues/29005
      ('group', {
          'groups': 3,
          'filters': 6
      }, (None, 5, 6), True),
  )
  def test_conv1d(self, kwargs, expected_output_shape, requires_gpu=False):
    kwargs['filters'] = kwargs.get('filters', 2)
    kwargs['kernel_size'] = 3
    if not requires_gpu or test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs, expected_output_shape)

  def test_conv1d_regularizers(self):
    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'kernel_regularizer': 'l2',
        'bias_regularizer': 'l2',
        'activity_regularizer': 'l2',
        'strides': 1
    }
    with self.cached_session(use_gpu=True):
      layer = keras.layers.Conv1D(**kwargs)
      layer.build((None, 5, 2))
      self.assertEqual(len(layer.losses), 2)
      layer(keras.backend.variable(np.ones((1, 5, 2))))
      self.assertEqual(len(layer.losses), 3)

  def test_conv1d_constraints(self):
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
    with self.cached_session(use_gpu=True):
      layer = keras.layers.Conv1D(**kwargs)
      layer.build((None, 5, 2))
      self.assertEqual(layer.kernel.constraint, k_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)

  def test_conv1d_recreate_conv(self):
    with self.cached_session(use_gpu=True):
      layer = keras.layers.Conv1D(filters=1,
                                  kernel_size=3,
                                  strides=1,
                                  dilation_rate=2,
                                  padding='causal')
      inpt1 = np.random.normal(size=[1, 2, 1])
      inpt2 = np.random.normal(size=[1, 1, 1])
      outp1_shape = layer(inpt1).shape
      _ = layer(inpt2).shape
      self.assertEqual(outp1_shape, layer(inpt1).shape)

  def test_conv1d_recreate_conv_unknown_dims(self):
    with self.cached_session(use_gpu=True):
      layer = keras.layers.Conv1D(filters=1,
                                  kernel_size=3,
                                  strides=1,
                                  dilation_rate=2,
                                  padding='causal')

      inpt1 = np.random.normal(size=[1, 9, 1]).astype(np.float32)
      inpt2 = np.random.normal(size=[1, 2, 1]).astype(np.float32)
      outp1_shape = layer(inpt1).shape

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec([1, None, 1])])
      def fn(inpt):
        return layer(inpt)

      fn(inpt2)
      self.assertEqual(outp1_shape, layer(inpt1).shape)


@keras_parameterized.run_all_keras_modes
class Conv2DTest(keras_parameterized.TestCase):

  def _run_test(self, kwargs, expected_output_shape):
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6

    with self.cached_session(use_gpu=True):
      testing_utils.layer_test(
          keras.layers.Conv2D,
          kwargs=kwargs,
          input_shape=(num_samples, num_row, num_col, stack_size),
          expected_output_shape=expected_output_shape)

  @parameterized.named_parameters(
      ('padding_valid', {
          'padding': 'valid'
      }, (None, 5, 4, 2)),
      ('padding_same', {
          'padding': 'same'
      }, (None, 7, 6, 2)),
      ('padding_same_dilation_2', {
          'padding': 'same',
          'dilation_rate': 2
      }, (None, 7, 6, 2)),
      ('strides', {
          'strides': (2, 2)
      }, (None, 3, 2, 2)),
      ('dilation_rate', {
          'dilation_rate': (2, 2)
      }, (None, 3, 2, 2)),
      # Only runs on GPU with CUDA, channels_first is not supported on CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      ('data_format', {
          'data_format': 'channels_first'
      }, None, True),
      # Only runs on GPU with CUDA, groups are not supported on CPU.
      # https://github.com/tensorflow/tensorflow/issues/29005
      ('group', {
          'groups': 3,
          'filters': 6
      }, (None, 5, 4, 6), True),
  )
  def test_conv2d(self, kwargs, expected_output_shape=None, requires_gpu=False):
    kwargs['filters'] = kwargs.get('filters', 2)
    kwargs['kernel_size'] = (3, 3)
    if not requires_gpu or test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs, expected_output_shape)

  def test_conv2d_regularizers(self):
    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'kernel_regularizer': 'l2',
        'bias_regularizer': 'l2',
        'activity_regularizer': 'l2',
        'strides': 1
    }
    with self.cached_session(use_gpu=True):
      layer = keras.layers.Conv2D(**kwargs)
      layer.build((None, 5, 5, 2))
      self.assertEqual(len(layer.losses), 2)
      layer(keras.backend.variable(np.ones((1, 5, 5, 2))))
      self.assertEqual(len(layer.losses), 3)

  def test_conv2d_constraints(self):
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
    with self.cached_session(use_gpu=True):
      layer = keras.layers.Conv2D(**kwargs)
      layer.build((None, 5, 5, 2))
      self.assertEqual(layer.kernel.constraint, k_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)

  def test_conv2d_zero_kernel_size(self):
    kwargs = {'filters': 2, 'kernel_size': 0}
    with self.assertRaises(ValueError):
      keras.layers.Conv2D(**kwargs)


@keras_parameterized.run_all_keras_modes
class Conv3DTest(keras_parameterized.TestCase):

  def _run_test(self, kwargs, expected_output_shape, validate_training=True):
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6
    depth = 5

    with self.cached_session(use_gpu=True):
      testing_utils.layer_test(
          keras.layers.Conv3D,
          kwargs=kwargs,
          input_shape=(num_samples, depth, num_row, num_col, stack_size),
          expected_output_shape=expected_output_shape,
          validate_training=validate_training)

  @parameterized.named_parameters(
      ('padding_valid', {
          'padding': 'valid'
      }, (None, 3, 5, 4, 2)),
      ('padding_same', {
          'padding': 'same'
      }, (None, 5, 7, 6, 2)),
      ('strides', {
          'strides': (2, 2, 2)
      }, (None, 2, 3, 2, 2)),
      ('dilation_rate', {
          'dilation_rate': (2, 2, 2)
      }, (None, 1, 3, 2, 2)),
      # Only runs on GPU with CUDA, channels_first is not supported on CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      ('data_format', {
          'data_format': 'channels_first'
      }, None, True),
      # Only runs on GPU with CUDA, groups are not supported on CPU.
      # https://github.com/tensorflow/tensorflow/issues/29005
      ('group', {
          'groups': 3,
          'filters': 6
      }, (None, 3, 5, 4, 6), True),
  )
  def test_conv3d(self, kwargs, expected_output_shape=None, requires_gpu=False):
    kwargs['filters'] = kwargs.get('filters', 2)
    kwargs['kernel_size'] = (3, 3, 3)
    # train_on_batch currently fails with XLA enabled on GPUs
    test_training = 'groups' not in kwargs or not test_util.is_xla_enabled()
    if not requires_gpu or test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs, expected_output_shape, test_training)

  def test_conv3d_regularizers(self):
    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'kernel_regularizer': 'l2',
        'bias_regularizer': 'l2',
        'activity_regularizer': 'l2',
        'strides': 1
    }
    with self.cached_session(use_gpu=True):
      layer = keras.layers.Conv3D(**kwargs)
      layer.build((None, 5, 5, 5, 2))
      self.assertEqual(len(layer.losses), 2)
      self.assertEqual(len(layer.losses), 2)
      layer(keras.backend.variable(np.ones((1, 5, 5, 5, 2))))
      self.assertEqual(len(layer.losses), 3)

  def test_conv3d_constraints(self):
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
    with self.cached_session(use_gpu=True):
      layer = keras.layers.Conv3D(**kwargs)
      layer.build((None, 5, 5, 5, 2))
      self.assertEqual(layer.kernel.constraint, k_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)

  def test_conv3d_dynamic_shape(self):
    input_data = np.random.random((1, 3, 3, 3, 3)).astype(np.float32)
    with self.cached_session(use_gpu=True):
      # Won't raise error here.
      testing_utils.layer_test(
          keras.layers.Conv3D,
          kwargs={
              'data_format': 'channels_last',
              'filters': 3,
              'kernel_size': 3
          },
          input_shape=(None, None, None, None, 3),
          input_data=input_data)
      if test.is_gpu_available(cuda_only=True):
        testing_utils.layer_test(
            keras.layers.Conv3D,
            kwargs={
                'data_format': 'channels_first',
                'filters': 3,
                'kernel_size': 3
            },
            input_shape=(None, 3, None, None, None),
            input_data=input_data)


class GroupedConvTest(keras_parameterized.TestCase):

  @parameterized.named_parameters(
      ('Conv1D', keras.layers.Conv1D),
      ('Conv2D', keras.layers.Conv2D),
      ('Conv3D', keras.layers.Conv3D),
  )
  def test_group_conv_incorrect_use(self, layer):
    with self.assertRaisesRegexp(ValueError, 'The number of filters'):
      layer(16, 3, groups=3)
    with self.assertRaisesRegexp(ValueError, 'The number of input channels'):
      layer(16, 3, groups=4).build((32, 12, 12, 3))

  @parameterized.named_parameters(
      ('Conv1D', keras.layers.Conv1D, (32, 12, 32)),
      ('Conv2D', keras.layers.Conv2D, (32, 12, 12, 32)),
      ('Conv3D', keras.layers.Conv3D, (32, 12, 12, 12, 32)),
  )
  def test_group_conv(self, layer_cls, input_shape):
    if test.is_gpu_available(cuda_only=True):
      with test_util.use_gpu():
        inputs = random_ops.random_uniform(shape=input_shape)

        layer = layer_cls(16, 3, groups=4, use_bias=False)
        layer.build(input_shape)

        input_slices = array_ops.split(inputs, 4, axis=-1)
        weight_slices = array_ops.split(layer.kernel, 4, axis=-1)
        expected_outputs = array_ops.concat([
            nn.convolution_v2(inputs, weights)
            for inputs, weights in zip(input_slices, weight_slices)
        ],
                                            axis=-1)

        self.assertAllClose(layer(inputs), expected_outputs, rtol=1e-5)

  def test_group_conv_depthwise(self):
    if test.is_gpu_available(cuda_only=True):
      with test_util.use_gpu():
        inputs = random_ops.random_uniform(shape=(3, 27, 27, 32))

        layer = keras.layers.Conv2D(32, 3, groups=32, use_bias=False)
        layer.build((3, 27, 27, 32))

        weights_dw = array_ops.reshape(layer.kernel, [3, 3, 32, 1])
        expected_outputs = nn.depthwise_conv2d(
            inputs, weights_dw, strides=[1, 1, 1, 1], padding='VALID')

        self.assertAllClose(layer(inputs), expected_outputs, rtol=1e-5)


@keras_parameterized.run_all_keras_modes
class Conv1DTransposeTest(keras_parameterized.TestCase):

  def _run_test(self, kwargs, expected_output_shape):
    num_samples = 2
    stack_size = 3
    num_col = 6

    with test_util.use_gpu():
      testing_utils.layer_test(
          keras.layers.Conv1DTranspose,
          kwargs=kwargs,
          input_shape=(num_samples, num_col, stack_size),
          expected_output_shape=expected_output_shape)

  @parameterized.named_parameters(
      ('padding_valid', {'padding': 'valid'}, (None, 8, 2)),
      ('padding_same', {'padding': 'same'}, (None, 6, 2)),
      ('strides', {'strides': 2}, (None, 13, 2)),
      # Only runs on GPU with CUDA, dilation_rate>1 is not supported on CPU.
      ('dilation_rate', {'dilation_rate': 2}, (None, 10, 2)),
      # Only runs on GPU with CUDA, channels_first is not supported on CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      ('data_format', {'data_format': 'channels_first'}),
  )
  def test_conv1d_transpose(self, kwargs, expected_output_shape=None):
    kwargs['filters'] = 2
    kwargs['kernel_size'] = 3
    if (('data_format' not in kwargs and 'dilation_rate' not in kwargs) or
        test.is_gpu_available(cuda_only=True)):
      self._run_test(kwargs, expected_output_shape)


@keras_parameterized.run_all_keras_modes
class Conv3DTransposeTest(keras_parameterized.TestCase):

  def _run_test(self, kwargs, expected_output_shape):
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6
    depth = 5

    with test_util.use_gpu():
      testing_utils.layer_test(
          keras.layers.Conv3DTranspose,
          kwargs=kwargs,
          input_shape=(num_samples, depth, num_row, num_col, stack_size),
          expected_output_shape=expected_output_shape)

  @parameterized.named_parameters(
      ('padding_valid', {'padding': 'valid'}, (None, 7, 9, 8, 2)),
      ('padding_same', {'padding': 'same'}, (None, 5, 7, 6, 2)),
      ('strides', {'strides': (2, 2, 2)}, (None, 11, 15, 13, 2)),
      ('dilation_rate', {'dilation_rate': (2, 2, 2)}, (None, 7, 9, 8, 2)),
      # Only runs on GPU with CUDA, channels_first is not supported on CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      ('data_format', {'data_format': 'channels_first'}),
  )
  def test_conv3d_transpose(self, kwargs, expected_output_shape=None):
    kwargs['filters'] = 2
    kwargs['kernel_size'] = (3, 3, 3)
    if 'data_format' not in kwargs or test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs, expected_output_shape)


@keras_parameterized.run_all_keras_modes
class ConvSequentialTest(keras_parameterized.TestCase):

  def _run_test(self, conv_layer_cls, kwargs, input_shape1, input_shape2,
                expected_output_shape1, expected_output_shape2):
    kwargs['filters'] = 1
    kwargs['kernel_size'] = 3
    kwargs['dilation_rate'] = 2
    with self.cached_session(use_gpu=True):
      layer = conv_layer_cls(**kwargs)
      output1 = layer(np.zeros(input_shape1))
      self.assertEqual(output1.shape, expected_output_shape1)
      output2 = layer(np.zeros(input_shape2))
      self.assertEqual(output2.shape, expected_output_shape2)

  @parameterized.named_parameters(
      ('padding_valid', {'padding': 'valid'},
       (1, 8, 2), (1, 5, 2), (1, 4, 1), (1, 1, 1)),
      ('padding_same', {'padding': 'same'},
       (1, 8, 2), (1, 5, 2), (1, 8, 1), (1, 5, 1)),
      ('padding_causal', {'padding': 'causal'},
       (1, 8, 2), (1, 5, 2), (1, 8, 1), (1, 5, 1)),
  )
  def test_conv1d(self, kwargs, input_shape1, input_shape2,
                  expected_output_shape1, expected_output_shape2):
    self._run_test(keras.layers.Conv1D, kwargs, input_shape1, input_shape2,
                   expected_output_shape1, expected_output_shape2)

  @parameterized.named_parameters(
      ('padding_valid', {'padding': 'valid'},
       (1, 7, 6, 2), (1, 6, 5, 2), (1, 3, 2, 1), (1, 2, 1, 1)),
      ('padding_same', {'padding': 'same'},
       (1, 7, 6, 2), (1, 6, 5, 2), (1, 7, 6, 1), (1, 6, 5, 1)),
  )
  def test_conv2d(self, kwargs, input_shape1, input_shape2,
                  expected_output_shape1, expected_output_shape2):
    self._run_test(keras.layers.Conv2D, kwargs, input_shape1, input_shape2,
                   expected_output_shape1, expected_output_shape2)

  @parameterized.named_parameters(
      ('padding_valid', {'padding': 'valid'},
       (1, 5, 7, 6, 2), (1, 8, 6, 5, 2), (1, 1, 3, 2, 1), (1, 4, 2, 1, 1)),
      ('padding_same', {'padding': 'same'},
       (1, 5, 7, 6, 2), (1, 8, 6, 5, 2), (1, 5, 7, 6, 1), (1, 8, 6, 5, 1)),
  )
  def test_conv3d(self, kwargs, input_shape1, input_shape2,
                  expected_output_shape1, expected_output_shape2):
    self._run_test(keras.layers.Conv3D, kwargs, input_shape1, input_shape2,
                   expected_output_shape1, expected_output_shape2)

  def test_dynamic_shape(self):
    with self.cached_session(use_gpu=True):
      layer = keras.layers.Conv3D(2, 3)
      input_shape = (5, None, None, 2)
      inputs = keras.Input(shape=input_shape)
      x = layer(inputs)
      # Won't raise error here with None values in input shape (b/144282043).
      layer(x)


@keras_parameterized.run_all_keras_modes
class ZeroPaddingTest(keras_parameterized.TestCase):

  def test_zero_padding_1d(self):
    num_samples = 2
    input_dim = 2
    num_steps = 5
    shape = (num_samples, num_steps, input_dim)
    inputs = np.ones(shape)

    with self.cached_session(use_gpu=True):
      # basic test
      testing_utils.layer_test(
          keras.layers.ZeroPadding1D,
          kwargs={'padding': 2},
          input_shape=inputs.shape)
      testing_utils.layer_test(
          keras.layers.ZeroPadding1D,
          kwargs={'padding': (1, 2)},
          input_shape=inputs.shape)

      # correctness test
      layer = keras.layers.ZeroPadding1D(padding=2)
      layer.build(shape)
      output = layer(keras.backend.variable(inputs))
      if context.executing_eagerly():
        np_output = output.numpy()
      else:
        np_output = keras.backend.eval(output)
      for offset in [0, 1, -1, -2]:
        np.testing.assert_allclose(np_output[:, offset, :], 0.)
      np.testing.assert_allclose(np_output[:, 2:-2, :], 1.)

      layer = keras.layers.ZeroPadding1D(padding=(1, 2))
      layer.build(shape)
      output = layer(keras.backend.variable(inputs))
      if context.executing_eagerly():
        np_output = output.numpy()
      else:
        np_output = keras.backend.eval(output)
      for left_offset in [0]:
        np.testing.assert_allclose(np_output[:, left_offset, :], 0.)
      for right_offset in [-1, -2]:
        np.testing.assert_allclose(np_output[:, right_offset, :], 0.)
      np.testing.assert_allclose(np_output[:, 1:-2, :], 1.)
      layer.get_config()

    # test incorrect use
    with self.assertRaises(ValueError):
      keras.layers.ZeroPadding1D(padding=(1, 1, 1))
    with self.assertRaises(ValueError):
      keras.layers.ZeroPadding1D(padding=None)

  def test_zero_padding_2d(self):
    num_samples = 2
    stack_size = 2
    input_num_row = 4
    input_num_col = 5
    for data_format in ['channels_first', 'channels_last']:
      inputs = np.ones((num_samples, input_num_row, input_num_col, stack_size))
      inputs = np.ones((num_samples, stack_size, input_num_row, input_num_col))

      # basic test
      with self.cached_session(use_gpu=True):
        testing_utils.layer_test(
            keras.layers.ZeroPadding2D,
            kwargs={'padding': (2, 2),
                    'data_format': data_format},
            input_shape=inputs.shape)
        testing_utils.layer_test(
            keras.layers.ZeroPadding2D,
            kwargs={'padding': ((1, 2), (3, 4)),
                    'data_format': data_format},
            input_shape=inputs.shape)

      # correctness test
      with self.cached_session(use_gpu=True):
        layer = keras.layers.ZeroPadding2D(
            padding=(2, 2), data_format=data_format)
        layer.build(inputs.shape)
        output = layer(keras.backend.variable(inputs))
        if context.executing_eagerly():
          np_output = output.numpy()
        else:
          np_output = keras.backend.eval(output)
        if data_format == 'channels_last':
          for offset in [0, 1, -1, -2]:
            np.testing.assert_allclose(np_output[:, offset, :, :], 0.)
            np.testing.assert_allclose(np_output[:, :, offset, :], 0.)
          np.testing.assert_allclose(np_output[:, 2:-2, 2:-2, :], 1.)
        elif data_format == 'channels_first':
          for offset in [0, 1, -1, -2]:
            np.testing.assert_allclose(np_output[:, :, offset, :], 0.)
            np.testing.assert_allclose(np_output[:, :, :, offset], 0.)
          np.testing.assert_allclose(np_output[:, 2:-2, 2:-2, :], 1.)

        layer = keras.layers.ZeroPadding2D(
            padding=((1, 2), (3, 4)), data_format=data_format)
        layer.build(inputs.shape)
        output = layer(keras.backend.variable(inputs))
        if context.executing_eagerly():
          np_output = output.numpy()
        else:
          np_output = keras.backend.eval(output)
        if data_format == 'channels_last':
          for top_offset in [0]:
            np.testing.assert_allclose(np_output[:, top_offset, :, :], 0.)
          for bottom_offset in [-1, -2]:
            np.testing.assert_allclose(np_output[:, bottom_offset, :, :], 0.)
          for left_offset in [0, 1, 2]:
            np.testing.assert_allclose(np_output[:, :, left_offset, :], 0.)
          for right_offset in [-1, -2, -3, -4]:
            np.testing.assert_allclose(np_output[:, :, right_offset, :], 0.)
          np.testing.assert_allclose(np_output[:, 1:-2, 3:-4, :], 1.)
        elif data_format == 'channels_first':
          for top_offset in [0]:
            np.testing.assert_allclose(np_output[:, :, top_offset, :], 0.)
          for bottom_offset in [-1, -2]:
            np.testing.assert_allclose(np_output[:, :, bottom_offset, :], 0.)
          for left_offset in [0, 1, 2]:
            np.testing.assert_allclose(np_output[:, :, :, left_offset], 0.)
          for right_offset in [-1, -2, -3, -4]:
            np.testing.assert_allclose(np_output[:, :, :, right_offset], 0.)
          np.testing.assert_allclose(np_output[:, :, 1:-2, 3:-4], 1.)

      # test incorrect use
      with self.assertRaises(ValueError):
        keras.layers.ZeroPadding2D(padding=(1, 1, 1))
      with self.assertRaises(ValueError):
        keras.layers.ZeroPadding2D(padding=None)

  def test_zero_padding_3d(self):
    num_samples = 2
    stack_size = 2
    input_len_dim1 = 4
    input_len_dim2 = 5
    input_len_dim3 = 3

    inputs = np.ones((num_samples, input_len_dim1, input_len_dim2,
                      input_len_dim3, stack_size))

    with self.cached_session(use_gpu=True):
      # basic test
      testing_utils.layer_test(
          keras.layers.ZeroPadding3D,
          kwargs={'padding': (2, 2, 2)},
          input_shape=inputs.shape)

      # correctness test
      layer = keras.layers.ZeroPadding3D(padding=(2, 2, 2))
      layer.build(inputs.shape)
      output = layer(keras.backend.variable(inputs))
      if context.executing_eagerly():
        np_output = output.numpy()
      else:
        np_output = keras.backend.eval(output)
      for offset in [0, 1, -1, -2]:
        np.testing.assert_allclose(np_output[:, offset, :, :, :], 0.)
        np.testing.assert_allclose(np_output[:, :, offset, :, :], 0.)
        np.testing.assert_allclose(np_output[:, :, :, offset, :], 0.)
      np.testing.assert_allclose(np_output[:, 2:-2, 2:-2, 2:-2, :], 1.)

    # test incorrect use
    with self.assertRaises(ValueError):
      keras.layers.ZeroPadding3D(padding=(1, 1))
    with self.assertRaises(ValueError):
      keras.layers.ZeroPadding3D(padding=None)


@test_util.for_all_test_methods(test_util.disable_xla,
                                'align_corners=False not supported by XLA')
@keras_parameterized.run_all_keras_modes
class UpSamplingTest(keras_parameterized.TestCase):

  def test_upsampling_1d(self):
    with self.cached_session(use_gpu=True):
      testing_utils.layer_test(
          keras.layers.UpSampling1D, kwargs={'size': 2}, input_shape=(3, 5, 4))

  def test_upsampling_2d(self):
    num_samples = 2
    stack_size = 2
    input_num_row = 11
    input_num_col = 12

    for data_format in ['channels_first', 'channels_last']:
      if data_format == 'channels_first':
        inputs = np.random.rand(num_samples, stack_size, input_num_row,
                                input_num_col)
      else:
        inputs = np.random.rand(num_samples, input_num_row, input_num_col,
                                stack_size)

      # basic test
      with self.cached_session(use_gpu=True):
        testing_utils.layer_test(
            keras.layers.UpSampling2D,
            kwargs={'size': (2, 2),
                    'data_format': data_format},
            input_shape=inputs.shape)

        for length_row in [2]:
          for length_col in [2, 3]:
            layer = keras.layers.UpSampling2D(
                size=(length_row, length_col), data_format=data_format)
            layer.build(inputs.shape)
            output = layer(keras.backend.variable(inputs))
            if context.executing_eagerly():
              np_output = output.numpy()
            else:
              np_output = keras.backend.eval(output)
            if data_format == 'channels_first':
              assert np_output.shape[2] == length_row * input_num_row
              assert np_output.shape[3] == length_col * input_num_col
            else:  # tf
              assert np_output.shape[1] == length_row * input_num_row
              assert np_output.shape[2] == length_col * input_num_col

            # compare with numpy
            if data_format == 'channels_first':
              expected_out = np.repeat(inputs, length_row, axis=2)
              expected_out = np.repeat(expected_out, length_col, axis=3)
            else:  # tf
              expected_out = np.repeat(inputs, length_row, axis=1)
              expected_out = np.repeat(expected_out, length_col, axis=2)

            np.testing.assert_allclose(np_output, expected_out)

  def test_upsampling_2d_bilinear(self):
    num_samples = 2
    stack_size = 2
    input_num_row = 11
    input_num_col = 12
    for data_format in ['channels_first', 'channels_last']:
      if data_format == 'channels_first':
        inputs = np.random.rand(num_samples, stack_size, input_num_row,
                                input_num_col)
      else:
        inputs = np.random.rand(num_samples, input_num_row, input_num_col,
                                stack_size)

      testing_utils.layer_test(keras.layers.UpSampling2D,
                               kwargs={'size': (2, 2),
                                       'data_format': data_format,
                                       'interpolation': 'bilinear'},
                               input_shape=inputs.shape)

      if not context.executing_eagerly():
        for length_row in [2]:
          for length_col in [2, 3]:
            layer = keras.layers.UpSampling2D(
                size=(length_row, length_col),
                data_format=data_format)
            layer.build(inputs.shape)
            outputs = layer(keras.backend.variable(inputs))
            np_output = keras.backend.eval(outputs)
            if data_format == 'channels_first':
              self.assertEqual(np_output.shape[2], length_row * input_num_row)
              self.assertEqual(np_output.shape[3], length_col * input_num_col)
            else:
              self.assertEqual(np_output.shape[1], length_row * input_num_row)
              self.assertEqual(np_output.shape[2], length_col * input_num_col)

  def test_upsampling_3d(self):
    num_samples = 2
    stack_size = 2
    input_len_dim1 = 10
    input_len_dim2 = 11
    input_len_dim3 = 12

    for data_format in ['channels_first', 'channels_last']:
      if data_format == 'channels_first':
        inputs = np.random.rand(num_samples, stack_size, input_len_dim1,
                                input_len_dim2, input_len_dim3)
      else:
        inputs = np.random.rand(num_samples, input_len_dim1, input_len_dim2,
                                input_len_dim3, stack_size)

      # basic test
      with self.cached_session(use_gpu=True):
        testing_utils.layer_test(
            keras.layers.UpSampling3D,
            kwargs={'size': (2, 2, 2),
                    'data_format': data_format},
            input_shape=inputs.shape)

        for length_dim1 in [2, 3]:
          for length_dim2 in [2]:
            for length_dim3 in [3]:
              layer = keras.layers.UpSampling3D(
                  size=(length_dim1, length_dim2, length_dim3),
                  data_format=data_format)
              layer.build(inputs.shape)
              output = layer(keras.backend.variable(inputs))
              if context.executing_eagerly():
                np_output = output.numpy()
              else:
                np_output = keras.backend.eval(output)
              if data_format == 'channels_first':
                assert np_output.shape[2] == length_dim1 * input_len_dim1
                assert np_output.shape[3] == length_dim2 * input_len_dim2
                assert np_output.shape[4] == length_dim3 * input_len_dim3
              else:  # tf
                assert np_output.shape[1] == length_dim1 * input_len_dim1
                assert np_output.shape[2] == length_dim2 * input_len_dim2
                assert np_output.shape[3] == length_dim3 * input_len_dim3

              # compare with numpy
              if data_format == 'channels_first':
                expected_out = np.repeat(inputs, length_dim1, axis=2)
                expected_out = np.repeat(expected_out, length_dim2, axis=3)
                expected_out = np.repeat(expected_out, length_dim3, axis=4)
              else:  # tf
                expected_out = np.repeat(inputs, length_dim1, axis=1)
                expected_out = np.repeat(expected_out, length_dim2, axis=2)
                expected_out = np.repeat(expected_out, length_dim3, axis=3)

              np.testing.assert_allclose(np_output, expected_out)


@keras_parameterized.run_all_keras_modes
class CroppingTest(keras_parameterized.TestCase):

  def test_cropping_1d(self):
    num_samples = 2
    time_length = 4
    input_len_dim1 = 2
    inputs = np.random.rand(num_samples, time_length, input_len_dim1)

    with self.cached_session(use_gpu=True):
      testing_utils.layer_test(
          keras.layers.Cropping1D,
          kwargs={'cropping': (2, 2)},
          input_shape=inputs.shape)

    # test incorrect use
    with self.assertRaises(ValueError):
      keras.layers.Cropping1D(cropping=(1, 1, 1))
    with self.assertRaises(ValueError):
      keras.layers.Cropping1D(cropping=None)

  def test_cropping_2d(self):
    num_samples = 2
    stack_size = 2
    input_len_dim1 = 9
    input_len_dim2 = 9
    cropping = ((2, 2), (3, 3))

    for data_format in ['channels_first', 'channels_last']:
      if data_format == 'channels_first':
        inputs = np.random.rand(num_samples, stack_size, input_len_dim1,
                                input_len_dim2)
      else:
        inputs = np.random.rand(num_samples, input_len_dim1, input_len_dim2,
                                stack_size)
      with self.cached_session(use_gpu=True):
        # basic test
        testing_utils.layer_test(
            keras.layers.Cropping2D,
            kwargs={'cropping': cropping,
                    'data_format': data_format},
            input_shape=inputs.shape)
        # correctness test
        layer = keras.layers.Cropping2D(
            cropping=cropping, data_format=data_format)
        layer.build(inputs.shape)
        output = layer(keras.backend.variable(inputs))
        if context.executing_eagerly():
          np_output = output.numpy()
        else:
          np_output = keras.backend.eval(output)
        # compare with numpy
        if data_format == 'channels_first':
          expected_out = inputs[:, :, cropping[0][0]:-cropping[0][1], cropping[
              1][0]:-cropping[1][1]]
        else:
          expected_out = inputs[:, cropping[0][0]:-cropping[0][1], cropping[1][
              0]:-cropping[1][1], :]
        np.testing.assert_allclose(np_output, expected_out)

    for data_format in ['channels_first', 'channels_last']:
      if data_format == 'channels_first':
        inputs = np.random.rand(num_samples, stack_size, input_len_dim1,
                                input_len_dim2)
      else:
        inputs = np.random.rand(num_samples, input_len_dim1, input_len_dim2,
                                stack_size)
      # another correctness test (no cropping)
      with self.cached_session(use_gpu=True):
        cropping = ((0, 0), (0, 0))
        layer = keras.layers.Cropping2D(
            cropping=cropping, data_format=data_format)
        layer.build(inputs.shape)
        output = layer(keras.backend.variable(inputs))
        if context.executing_eagerly():
          np_output = output.numpy()
        else:
          np_output = keras.backend.eval(output)
        # compare with input
        np.testing.assert_allclose(np_output, inputs)

    # test incorrect use
    with self.assertRaises(ValueError):
      keras.layers.Cropping2D(cropping=(1, 1, 1))
    with self.assertRaises(ValueError):
      keras.layers.Cropping2D(cropping=None)

  def test_cropping_3d(self):
    num_samples = 2
    stack_size = 2
    input_len_dim1 = 8
    input_len_dim2 = 8
    input_len_dim3 = 8
    croppings = [((2, 2), (1, 1), (2, 3)), 3, (0, 1, 1)]

    for cropping in croppings:
      for data_format in ['channels_last', 'channels_first']:
        if data_format == 'channels_first':
          inputs = np.random.rand(num_samples, stack_size, input_len_dim1,
                                  input_len_dim2, input_len_dim3)
        else:
          inputs = np.random.rand(num_samples, input_len_dim1, input_len_dim2,
                                  input_len_dim3, stack_size)
        # basic test
        with self.cached_session(use_gpu=True):
          testing_utils.layer_test(
              keras.layers.Cropping3D,
              kwargs={'cropping': cropping,
                      'data_format': data_format},
              input_shape=inputs.shape)

        if len(croppings) == 3 and len(croppings[0]) == 2:
          # correctness test
          with self.cached_session(use_gpu=True):
            layer = keras.layers.Cropping3D(
                cropping=cropping, data_format=data_format)
            layer.build(inputs.shape)
            output = layer(keras.backend.variable(inputs))
            if context.executing_eagerly():
              np_output = output.numpy()
            else:
              np_output = keras.backend.eval(output)
            # compare with numpy
            if data_format == 'channels_first':
              expected_out = inputs[:, :,
                                    cropping[0][0]:-cropping[0][1],
                                    cropping[1][0]:-cropping[1][1],
                                    cropping[2][0]:-cropping[2][1]]
            else:
              expected_out = inputs[:,
                                    cropping[0][0]:-cropping[0][1],
                                    cropping[1][0]:-cropping[1][1],
                                    cropping[2][0]:-cropping[2][1], :]
            np.testing.assert_allclose(np_output, expected_out)

    # test incorrect use
    with self.assertRaises(ValueError):
      keras.layers.Cropping3D(cropping=(1, 1))
    with self.assertRaises(ValueError):
      keras.layers.Cropping3D(cropping=None)


@keras_parameterized.run_all_keras_modes
class DepthwiseConv2DTest(keras_parameterized.TestCase):

  def _run_test(self, kwargs, expected_output_shape=None):
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6

    with self.cached_session(use_gpu=True):
      testing_utils.layer_test(
          keras.layers.DepthwiseConv2D,
          kwargs=kwargs,
          input_shape=(num_samples, num_row, num_col, stack_size),
          expected_output_shape=expected_output_shape)

  @parameterized.named_parameters(
      ('padding_valid', {'padding': 'valid'}),
      ('padding_same', {'padding': 'same'}),
      ('strides', {'strides': (2, 2)}),
      # Only runs on GPU with CUDA, channels_first is not supported on CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      ('data_format', {'data_format': 'channels_first'}),
      ('depth_multiplier_1', {'depth_multiplier': 1}),
      ('depth_multiplier_2', {'depth_multiplier': 2}),
      ('dilation_rate', {'dilation_rate': (2, 2)}, (None, 3, 2, 3)),
  )
  def test_depthwise_conv2d(self, kwargs, expected_output_shape=None):
    kwargs['kernel_size'] = (3, 3)
    if 'data_format' not in kwargs or test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs, expected_output_shape)

  def test_depthwise_conv2d_full(self):
    kwargs = {
        'kernel_size': 3,
        'padding': 'valid',
        'data_format': 'channels_last',
        'dilation_rate': (1, 1),
        'activation': None,
        'depthwise_regularizer': 'l2',
        'bias_regularizer': 'l2',
        'activity_regularizer': 'l2',
        'depthwise_constraint': 'unit_norm',
        'use_bias': True,
        'strides': (2, 2),
        'depth_multiplier': 1,
    }
    self._run_test(kwargs)

if __name__ == '__main__':
  test.main()
