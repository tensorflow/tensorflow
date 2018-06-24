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

import copy

import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test


class Convolution1DTest(test.TestCase):

  def _run_test(self, kwargs, arg, values):
    num_samples = 2
    stack_size = 3
    length = 7

    test_kwargs = copy.copy(kwargs)
    for value in values:
      test_kwargs[arg] = value
      with self.test_session(use_gpu=True):
        testing_utils.layer_test(
            keras.layers.Conv1D,
            kwargs=test_kwargs,
            input_shape=(num_samples, length, stack_size))

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_conv1d(self):
    kwargs = {
        'filters': 2,
        'kernel_size': 3,
    }

    self._run_test(kwargs, 'padding', ['valid', 'same'])
    self._run_test(kwargs, 'strides', [2])
    self._run_test(kwargs, 'dilation_rate', [2])

    kwargs = {
        'filters': 2,
        'kernel_size': 3,
        'padding': 'same',
    }
    self._run_test(kwargs, 'dilation_rate', [2])
    self._run_test(kwargs, 'dilation_rate', [3])

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
    with self.test_session(use_gpu=True):
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
    with self.test_session(use_gpu=True):
      layer = keras.layers.Conv1D(**kwargs)
      layer.build((None, 5, 2))
      self.assertEqual(layer.kernel.constraint, k_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)


class Conv2DTest(test.TestCase):

  def _run_test(self, kwargs, arg, values):
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6

    test_kwargs = copy.copy(kwargs)
    for value in values:
      test_kwargs[arg] = value
      with self.test_session(use_gpu=True):
        testing_utils.layer_test(
            keras.layers.SeparableConv2D,
            kwargs=test_kwargs,
            input_shape=(num_samples, num_row, num_col, stack_size))

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_conv2d(self):
    kwargs = {
        'filters': 2,
        'kernel_size': (3, 3),
    }

    self._run_test(kwargs, 'padding', ['valid', 'same'])
    self._run_test(kwargs, 'strides', [(2, 2)])
    if test.is_gpu_available(cuda_only=True):
      # Only runs on GPU with CUDA, channels_first is not supported on CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      self._run_test(kwargs, 'data_format', ['channels_first'])
    self._run_test(kwargs, 'dilation_rate', [(2, 2)])

    kwargs = {
        'filters': 2,
        'kernel_size': 3,
        'padding': 'same',
    }
    self._run_test(kwargs, 'dilation_rate', [2])

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
    with self.test_session(use_gpu=True):
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
    with self.test_session(use_gpu=True):
      layer = keras.layers.Conv2D(**kwargs)
      layer.build((None, 5, 5, 2))
      self.assertEqual(layer.kernel.constraint, k_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)


class Conv2DTransposeTest(test.TestCase):

  def _run_test(self, kwargs, arg, values):
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6

    test_kwargs = copy.copy(kwargs)
    for value in values:
      test_kwargs[arg] = value
      with self.test_session(use_gpu=True):
        testing_utils.layer_test(
            keras.layers.Conv2DTranspose,
            kwargs=test_kwargs,
            input_shape=(num_samples, num_row, num_col, stack_size))

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_conv2dtranspose(self):
    kwargs = {
        'filters': 2,
        'kernel_size': (3, 3),
    }

    self._run_test(kwargs, 'padding', ['valid', 'same'])
    self._run_test(kwargs, 'strides', [(2, 2)])
    if test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs, 'data_format', ['channels_first'])

  def test_conv2dtranspose_regularizers(self):
    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'kernel_regularizer': 'l2',
        'bias_regularizer': 'l2',
        'activity_regularizer': 'l2',
        'strides': 1
    }
    with self.test_session(use_gpu=True):
      layer = keras.layers.Conv2DTranspose(**kwargs)
      layer.build((None, 5, 5, 2))
      self.assertEqual(len(layer.losses), 2)
      layer(keras.backend.variable(np.ones((1, 5, 5, 2))))
      self.assertEqual(len(layer.losses), 3)

  def test_conv2dtranspose_constraints(self):
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
    with self.test_session(use_gpu=True):
      layer = keras.layers.Conv2DTranspose(**kwargs)
      layer.build((None, 5, 5, 2))
      self.assertEqual(layer.kernel.constraint, k_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)


class Conv3DTransposeTest(test.TestCase):

  def _run_test(self, kwargs, arg, values):
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6
    depth = 5

    test_kwargs = copy.copy(kwargs)
    for value in values:
      test_kwargs[arg] = value
      with self.test_session(use_gpu=True):
        testing_utils.layer_test(
            keras.layers.Conv3DTranspose,
            kwargs=test_kwargs,
            input_shape=(num_samples, depth, num_row, num_col, stack_size))

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_conv3dtranspose(self):
    kwargs = {
        'filters': 2,
        'kernel_size': (3, 3, 3),
    }

    self._run_test(kwargs, 'padding', ['valid', 'same'])
    self._run_test(kwargs, 'strides', [(2, 2, 2)])
    if test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs, 'data_format', ['channels_first'])

  def test_conv3dtranspose_regularizers(self):
    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'kernel_regularizer': 'l2',
        'bias_regularizer': 'l2',
        'activity_regularizer': 'l2',
        'strides': 1
    }
    with self.test_session(use_gpu=True):
      layer = keras.layers.Conv3DTranspose(**kwargs)
      layer.build((None, 5, 5, 5, 2))
      self.assertEqual(len(layer.losses), 2)
      layer(keras.backend.variable(np.ones((1, 5, 5, 5, 2))))
      self.assertEqual(len(layer.losses), 3)

  def test_conv3dtranspose_constraints(self):
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
    with self.test_session(use_gpu=True):
      layer = keras.layers.Conv3DTranspose(**kwargs)
      layer.build((None, 5, 5, 5, 2))
      self.assertEqual(layer.kernel.constraint, k_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)


class SeparableConv1DTest(test.TestCase):

  def _run_test(self, kwargs, arg, values):
    num_samples = 2
    stack_size = 3
    length = 7

    test_kwargs = copy.copy(kwargs)
    for value in values:
      test_kwargs[arg] = value
      with self.test_session(use_gpu=True):
        testing_utils.layer_test(
            keras.layers.SeparableConv1D,
            kwargs=test_kwargs,
            input_shape=(num_samples, length, stack_size))

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_separable_conv1d(self):
    kwargs = {
        'filters': 2,
        'kernel_size': 3,
    }

    self._run_test(kwargs, 'padding', ['valid', 'same'])
    self._run_test(kwargs, 'strides', [2])
    self._run_test(kwargs, 'dilation_rate', [2])
    self._run_test(kwargs, 'depth_multiplier', [2])

    kwargs = {
        'filters': 2,
        'kernel_size': 3,
        'padding': 'same',
    }
    self._run_test(kwargs, 'dilation_rate', [2])

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
    with self.test_session(use_gpu=True):
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
    with self.test_session(use_gpu=True):
      layer = keras.layers.SeparableConv1D(**kwargs)
      layer.build((None, 5, 2))
      self.assertEqual(layer.depthwise_kernel.constraint, d_constraint)
      self.assertEqual(layer.pointwise_kernel.constraint, p_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)


class SeparableConv2DTest(test.TestCase):

  def _run_test(self, kwargs, arg, values):
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6

    test_kwargs = copy.copy(kwargs)
    for value in values:
      test_kwargs[arg] = value
      with self.test_session(use_gpu=True):
        testing_utils.layer_test(
            keras.layers.SeparableConv2D,
            kwargs=test_kwargs,
            input_shape=(num_samples, num_row, num_col, stack_size))

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_separable_conv2d(self):
    kwargs = {
        'filters': 2,
        'kernel_size': 3,
    }

    self._run_test(kwargs, 'padding', ['valid', 'same'])
    self._run_test(kwargs, 'strides', [2])
    if test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs, 'data_format', ['channels_first'])
    self._run_test(kwargs, 'dilation_rate', [2])
    self._run_test(kwargs, 'depth_multiplier', [2])

    kwargs = {
        'filters': 2,
        'kernel_size': 3,
        'padding': 'same',
    }
    self._run_test(kwargs, 'dilation_rate', [2])

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
    with self.test_session(use_gpu=True):
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
    with self.test_session(use_gpu=True):
      layer = keras.layers.SeparableConv2D(**kwargs)
      layer.build((None, 5, 5, 2))
      self.assertEqual(layer.depthwise_kernel.constraint, d_constraint)
      self.assertEqual(layer.pointwise_kernel.constraint, p_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)


class Conv3DTest(test.TestCase):

  def _run_test(self, kwargs, arg, values):
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6
    depth = 5

    test_kwargs = copy.copy(kwargs)
    for value in values:
      test_kwargs[arg] = value
      with self.test_session(use_gpu=True):
        testing_utils.layer_test(
            keras.layers.Conv3D,
            kwargs=test_kwargs,
            input_shape=(num_samples, depth, num_row, num_col, stack_size))

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_conv3d(self):
    kwargs = {
        'filters': 2,
        'kernel_size': (3, 3, 3),
    }

    self._run_test(kwargs, 'padding', ['valid', 'same'])
    self._run_test(kwargs, 'strides', [(2, 2, 2)])
    self._run_test(kwargs, 'dilation_rate', [(2, 2, 2)])
    if test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs, 'data_format', ['channels_first'])

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
    with self.test_session(use_gpu=True):
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
    with self.test_session(use_gpu=True):
      layer = keras.layers.Conv3D(**kwargs)
      layer.build((None, 5, 5, 5, 2))
      self.assertEqual(layer.kernel.constraint, k_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)


class ZeroPaddingTest(test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_zero_padding_1d(self):
    num_samples = 2
    input_dim = 2
    num_steps = 5
    shape = (num_samples, num_steps, input_dim)
    inputs = np.ones(shape)

    # basic test
    with self.test_session(use_gpu=True):
      testing_utils.layer_test(
          keras.layers.ZeroPadding1D,
          kwargs={'padding': 2},
          input_shape=inputs.shape)
      testing_utils.layer_test(
          keras.layers.ZeroPadding1D,
          kwargs={'padding': (1, 2)},
          input_shape=inputs.shape)

    # correctness test
    with self.test_session(use_gpu=True):
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

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_zero_padding_2d(self):
    num_samples = 2
    stack_size = 2
    input_num_row = 4
    input_num_col = 5
    for data_format in ['channels_first', 'channels_last']:
      inputs = np.ones((num_samples, input_num_row, input_num_col, stack_size))
      inputs = np.ones((num_samples, stack_size, input_num_row, input_num_col))

      # basic test
      with self.test_session(use_gpu=True):
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
      with self.test_session(use_gpu=True):
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

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_zero_padding_3d(self):
    num_samples = 2
    stack_size = 2
    input_len_dim1 = 4
    input_len_dim2 = 5
    input_len_dim3 = 3

    inputs = np.ones((num_samples, input_len_dim1, input_len_dim2,
                      input_len_dim3, stack_size))

    # basic test
    with self.test_session(use_gpu=True):
      testing_utils.layer_test(
          keras.layers.ZeroPadding3D,
          kwargs={'padding': (2, 2, 2)},
          input_shape=inputs.shape)

    # correctness test
    with self.test_session(use_gpu=True):
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


class UpSamplingTest(test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_upsampling_1d(self):
    with self.test_session(use_gpu=True):
      testing_utils.layer_test(
          keras.layers.UpSampling1D, kwargs={'size': 2}, input_shape=(3, 5, 4))

  @tf_test_util.run_in_graph_and_eager_modes()
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
      with self.test_session(use_gpu=True):
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

  @tf_test_util.run_in_graph_and_eager_modes()
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
      with self.test_session(use_gpu=True):
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


class CroppingTest(test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_cropping_1d(self):
    num_samples = 2
    time_length = 4
    input_len_dim1 = 2
    inputs = np.random.rand(num_samples, time_length, input_len_dim1)

    with self.test_session(use_gpu=True):
      testing_utils.layer_test(
          keras.layers.Cropping1D,
          kwargs={'cropping': (2, 2)},
          input_shape=inputs.shape)

    # test incorrect use
    with self.assertRaises(ValueError):
      keras.layers.Cropping1D(cropping=(1, 1, 1))
    with self.assertRaises(ValueError):
      keras.layers.Cropping1D(cropping=None)

  @tf_test_util.run_in_graph_and_eager_modes()
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
      # basic test
      with self.test_session(use_gpu=True):
        testing_utils.layer_test(
            keras.layers.Cropping2D,
            kwargs={'cropping': cropping,
                    'data_format': data_format},
            input_shape=inputs.shape)
      # correctness test
      with self.test_session(use_gpu=True):
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
      with self.test_session(use_gpu=True):
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

  @tf_test_util.run_in_graph_and_eager_modes()
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
        with self.test_session(use_gpu=True):
          testing_utils.layer_test(
              keras.layers.Cropping3D,
              kwargs={'cropping': cropping,
                      'data_format': data_format},
              input_shape=inputs.shape)

        if len(croppings) == 3 and len(croppings[0]) == 2:
          # correctness test
          with self.test_session(use_gpu=True):
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


class DepthwiseConv2DTest(test.TestCase):

  def _run_test(self, kwargs, arg, values):
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6

    test_kwargs = copy.copy(kwargs)
    for value in values:
      test_kwargs[arg] = value
      with self.test_session(use_gpu=True):
        testing_utils.layer_test(
            keras.layers.DepthwiseConv2D,
            kwargs=test_kwargs,
            input_shape=(num_samples, num_row, num_col, stack_size))

  def test_depthwise_conv2d(self):
    kwargs = {'kernel_size': (3, 3)}

    self._run_test(kwargs, 'padding', ['valid', 'same'])
    self._run_test(kwargs, 'strides', [(2, 2)])
    if test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs, 'data_format', ['channels_first'])
    self._run_test(kwargs, 'depth_multiplier', [1, 2])

    kwargs = {'kernel_size': 3,
              'padding': 'valid',
              'data_format': 'channels_first',
              'activation': None,
              'depthwise_regularizer': 'l2',
              'bias_regularizer': 'l2',
              'activity_regularizer': 'l2',
              'depthwise_constraint': 'unit_norm',
              'strides': (2, 2),
             }
    self._run_test(kwargs, 'depth_multiplier', [1])

if __name__ == '__main__':
  test.main()
