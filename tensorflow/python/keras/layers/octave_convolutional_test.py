"""Tests for octave convolutional layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes
class OctaveConv1DTest(keras_parameterized.TestCase):

  def _run_test(self, kwargs, expected_output_shape):
    num_samples = 2
    stack_size = 3
    length = 14

    with self.cached_session(use_gpu=True):
      testing_utils.layer_test(
          keras.layers.octave_convolutional.OctaveConv1D,
          kwargs=kwargs,
          input_shape=(num_samples, length, stack_size),
          expected_output_shape=expected_output_shape)

  @parameterized.named_parameters(
      ('padding_same', {'padding': 'same'}, (None, 7, 2)),
      ('padding_same_dilation_2', {'padding': 'same', 'dilation_rate': 2},
       (None, 7, 2)),
      ('padding_same_dilation_3', {'padding': 'same', 'dilation_rate': 3},
       (None, 7, 2)),
      ('padding_causal', {'padding': 'causal'}, (None, 7, 2)),
      ('low_freq_ratio', {'low_freq_ratio': 0.25}, (0.0, 0.5, 1.0)),
      ('strides', {'strides': 2}, (None, 3, 2)),
      ('dilation_rate', {'dilation_rate': 2}, (None, 3, 2)),
  )
  def test_octave_conv1D(self, kwargs, expected_output_shape):
    kwargs['filters'] = 2
    kwargs['kernel_size'] = 3
    self._run_test(kwargs, expected_output_shape)

  def test_octave_conv1D_regularizers(self):
    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'same',
        'low_freq_ratio': 0.5,
        'kernel_regularizer': 'l2',
        'bias_regularizer': 'l2',
        'activity_regularizer': 'l2',
        'strides': 1
    }
    with self.cached_session(use_gpu=True):
      layer = keras.layers.octave_convolutional.OctaveConv1D(**kwargs)
      layer.build((None, 5, 2))
      self.assertEqual(len(layer.losses), 2)
      layer(keras.backend.variable(np.ones((1, 5, 2))))
      self.assertEqual(len(layer.losses), 3)

  def test_octave_conv1D_constraints(self):
    k_constraint = lambda x: x
    b_constraint = lambda x: x

    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'same',
        'low_freq_ratio': 0.5,
        'kernel_constraint': k_constraint,
        'bias_constraint': b_constraint,
        'strides': 1
    }
    with self.cached_session(use_gpu=True):
      layer = keras.layers.octave_convolutional.OctaveConv1D(**kwargs)
      layer.build((None, 5, 2))
      self.assertEqual(layer.kernel.constraint, k_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)

  def test_octave_conv1D_recreate_conv(self):
    with self.cached_session(use_gpu=True):
      layer = keras.layers.octave_convolutional.OctaveConv1D(filters=1,
                                  kernel_size=3,
                                  strides=1,
                                  dilation_rate=2,
                                  padding='causal')
      inpt1 = np.random.normal(size=[1, 2, 1])
      inpt2 = np.random.normal(size=[1, 1, 1])
      outp1_shape = layer(inpt1).shape
      _ = layer(inpt2).shape
      self.assertEqual(outp1_shape, layer(inpt1).shape)


@keras_parameterized.run_all_keras_modes
class OctaveConv2DTest(keras_parameterized.TestCase):

  def _run_test(self, kwargs, expected_output_shape):
    num_samples = 2
    stack_size = 6
    num_row = 14
    num_col = 12

    with self.cached_session(use_gpu=True):
      testing_utils.layer_test(
          keras.layers.octave_convolutional.OctaveConv2D,
          kwargs=kwargs,
          input_shape=(num_samples, num_row, num_col, stack_size),
          expected_output_shape=expected_output_shape)

  @parameterized.named_parameters(
      ('padding_same', {'padding': 'same'}, (None, 7, 6, 2)),
      ('padding_same_dilation_2', {'padding': 'same', 'dilation_rate': 2},
       (None, 7, 6, 2)),
      ('strides', {'strides': (2, 2)}, (None, 3, 2, 2)),
      ('dilation_rate', {'dilation_rate': (2, 2)}, (None, 3, 2, 2)),
      ('low_freq_ratio', {'low_freq_ratio': 0.25}, (0.0, 0.5, 1.0)),
      # Only runs on GPU with CUDA, channels_first is not supported on CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      ('data_format', {'data_format': 'channels_first'}),
  )
  def test_octave_conv2D(self, kwargs, expected_output_shape=None):
    kwargs['filters'] = 2
    kwargs['kernel_size'] = (3, 3)
    if 'data_format' not in kwargs or test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs, expected_output_shape)

  def test_octave_conv2D_regularizers(self):
    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'same',
        'low_freq_ratio': 0.5,
        'kernel_regularizer': 'l2',
        'bias_regularizer': 'l2',
        'activity_regularizer': 'l2',
        'strides': 1
    }
    with self.cached_session(use_gpu=True):
      layer = keras.layers.octave_convolutional.OctaveConv2D(**kwargs)
      layer.build((None, 5, 5, 2))
      self.assertEqual(len(layer.losses), 2)
      layer(keras.backend.variable(np.ones((1, 5, 5, 2))))
      self.assertEqual(len(layer.losses), 3)

  def test_octave_conv2D_constraints(self):
    k_constraint = lambda x: x
    b_constraint = lambda x: x

    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'same',
        'low_freq_ratio': 0.5,
        'kernel_constraint': k_constraint,
        'bias_constraint': b_constraint,
        'strides': 1
    }
    with self.cached_session(use_gpu=True):
      layer = keras.layers.octave_convolutional.OctaveConv2D(**kwargs)
      layer.build((None, 5, 5, 2))
      self.assertEqual(layer.kernel.constraint, k_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)

  def test_octave_conv2D_zero_kernel_size(self):
    kwargs = {'filters': 2, 'kernel_size': 0}
    with self.assertRaises(ValueError):
      keras.layers.octave_convolutional.OctaveConv2D(**kwargs)


@keras_parameterized.run_all_keras_modes
class OctaveConv3DTest(keras_parameterized.TestCase):

  def _run_test(self, kwargs, expected_output_shape):
    num_samples = 2
    stack_size = 3
    num_row = 14
    num_col = 12
    depth = 10

    with self.cached_session(use_gpu=True):
      testing_utils.layer_test(
          keras.layers.octave_convolutional.OctaveConv3D,
          kwargs=kwargs,
          input_shape=(num_samples, depth, num_row, num_col, stack_size),
          expected_output_shape=expected_output_shape)

  @parameterized.named_parameters(
      ('padding_same', {'padding': 'same'}, (None, 5, 7, 6, 2)),
      ('strides', {'strides': (2, 2, 2)}, (None, 2, 3, 2, 2)),
      ('dilation_rate', {'dilation_rate': (2, 2, 2)}, (None, 1, 3, 2, 2)),
      ('low_freq_ratio', {'low_freq_ratio': 0.25}, (0.0, 0.5, 1.0)),
      # Only runs on GPU with CUDA, channels_first is not supported on CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      ('data_format', {'data_format': 'channels_first'}),
  )
  def test_octave_conv3D(self, kwargs, expected_output_shape=None):
    kwargs['filters'] = 2
    kwargs['kernel_size'] = (3, 3, 3)
    if 'data_format' not in kwargs or test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs, expected_output_shape)

  def test_octave_conv3D_regularizers(self):
    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'same',
        'low_freq_ratio': 0.5,
        'kernel_regularizer': 'l2',
        'bias_regularizer': 'l2',
        'activity_regularizer': 'l2',
        'strides': 1
    }
    with self.cached_session(use_gpu=True):
      layer = keras.layers.octave_convolutional.OctaveConv3D(**kwargs)
      layer.build((None, 5, 5, 5, 2))
      self.assertEqual(len(layer.losses), 2)
      self.assertEqual(len(layer.losses), 2)
      layer(keras.backend.variable(np.ones((1, 5, 5, 5, 2))))
      self.assertEqual(len(layer.losses), 3)

  def test_octave_conv3D_constraints(self):
    k_constraint = lambda x: x
    b_constraint = lambda x: x

    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'same',
        'low_freq_ratio': 0.5,
        'kernel_constraint': k_constraint,
        'bias_constraint': b_constraint,
        'strides': 1
    }
    with self.cached_session(use_gpu=True):
      layer = keras.layers.octave_convolutional.OctaveConv3D(**kwargs)
      layer.build((None, 5, 5, 5, 2))
      self.assertEqual(layer.kernel.constraint, k_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)

  def test_octave_conv3D_dynamic_shape(self):
    input_data = np.random.random((1, 3, 3, 3, 3)).astype(np.float32)
    with self.cached_session(use_gpu=True):
      # Won't raise error here.
      testing_utils.layer_test(
          keras.layers.octave_convolutional.OctaveConv3D,
          kwargs={
              'data_format': 'channels_last',
              'filters': 3,
              'kernel_size': 3,
              'low_freq_ratio': 0.5
          },
          input_shape=(None, None, None, None, 3),
          input_data=input_data)
      if test.is_gpu_available(cuda_only=True):
        testing_utils.layer_test(
            keras.layers.octave_convolutional.OctaveConv3D,
            kwargs={
                'data_format': 'channels_first',
                'filters': 3,
                'kernel_size': 3,
                'low_freq_ratio': 0.5
            },
            input_shape=(None, 3, None, None, None),
            input_data=input_data)


@keras_parameterized.run_all_keras_modes
class OctaveConv2DTransposeTest(keras_parameterized.TestCase):

  def _run_test(self, kwargs, expected_output_shape):
    num_samples = 2
    stack_size = 3
    num_row = 14
    num_col = 12

    with test_util.use_gpu():
      testing_utils.layer_test(
          keras.layers.octave_convolutional.OctaveConv2DTranspose,
          kwargs=kwargs,
          input_shape=(num_samples, num_row, num_col, stack_size),
          expected_output_shape=expected_output_shape)

  @parameterized.named_parameters(
      ('padding_same', {'padding': 'same'}, (None, 5, 7, 6, 2)),
      ('strides', {'strides': (2, 2, 2)}, (None, 11, 15, 13, 2)),
      ('dilation_rate', {'dilation_rate': (2, 2, 2)}, (None, 7, 9, 8, 2)),
      ('low_freq_ratio', {'low_freq_ratio': 0.25}, (0.0, 0.5, 1.0)),
      # Only runs on GPU with CUDA, channels_first is not supported on CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      ('data_format', {'data_format': 'channels_first'}),
  )
  def test_octave_conv2D_transpose(self, kwargs, expected_output_shape=None):
    kwargs['filters'] = 2
    kwargs['kernel_size'] = (3, 3)
    if 'data_format' not in kwargs or test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs, expected_output_shape)


@keras_parameterized.run_all_keras_modes
class OctaveConv3DTransposeTest(keras_parameterized.TestCase):

  def _run_test(self, kwargs, expected_output_shape):
    num_samples = 2
    stack_size = 3
    num_row = 14
    num_col = 12
    depth = 10

    with test_util.use_gpu():
      testing_utils.layer_test(
          keras.layers.octave_convolutional.OctaveConv3DTranspose,
          kwargs=kwargs,
          input_shape=(num_samples, depth, num_row, num_col, stack_size),
          expected_output_shape=expected_output_shape)

  @parameterized.named_parameters(
      ('padding_same', {'padding': 'same'}, (None, 5, 7, 6, 2)),
      ('strides', {'strides': (2, 2, 2)}, (None, 11, 15, 13, 2)),
      ('dilation_rate', {'dilation_rate': (2, 2, 2)}, (None, 7, 9, 8, 2)),
      ('low_freq_ratio', {'low_freq_ratio': 0.25}, (0.0, 0.5, 1.0)),
      # Only runs on GPU with CUDA, channels_first is not supported on CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      ('data_format', {'data_format': 'channels_first'}),
  )
  def test_octave_conv3D_transpose(self, kwargs, expected_output_shape=None):
    kwargs['filters'] = 2
    kwargs['kernel_size'] = (3, 3, 3)
    if 'data_format' not in kwargs or test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs, expected_output_shape)

if __name__ == '__main__':
  test.main()
