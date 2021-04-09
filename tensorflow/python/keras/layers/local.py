# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-classes-have-attributes
"""Locally-connected layers."""

import numpy as np

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.LocallyConnected1D')
class LocallyConnected1D(Layer):
  """Locally-connected layer for 1D inputs.

  The `LocallyConnected1D` layer works similarly to
  the `Conv1D` layer, except that weights are unshared,
  that is, a different set of filters is applied at each different patch
  of the input.

  Note: layer attributes cannot be modified after the layer has been called
  once (except the `trainable` attribute).

  Example:
  ```python
      # apply a unshared weight convolution 1d of length 3 to a sequence with
      # 10 timesteps, with 64 output filters
      model = Sequential()
      model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
      # now model.output_shape == (None, 8, 64)
      # add a new conv1d on top
      model.add(LocallyConnected1D(32, 3))
      # now model.output_shape == (None, 6, 32)
  ```

  Args:
      filters: Integer, the dimensionality of the output space (i.e. the number
        of output filters in the convolution).
      kernel_size: An integer or tuple/list of a single integer, specifying the
        length of the 1D convolution window.
      strides: An integer or tuple/list of a single integer, specifying the
        stride length of the convolution.
      padding: Currently only supports `"valid"` (case-insensitive). `"same"`
        may be supported in the future. `"valid"` means no padding.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch, length,
        channels)` while `channels_first` corresponds to inputs with shape
        `(batch, channels, length)`. It defaults to the `image_data_format`
        value found in your Keras config file at `~/.keras/keras.json`. If you
        never set it, then it will be "channels_last".
      activation: Activation function to use. If you don't specify anything, no
        activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
      implementation: implementation mode, either `1`, `2`, or `3`. `1` loops
        over input spatial locations to perform the forward pass. It is
        memory-efficient but performs a lot of (small) ops.  `2` stores layer
        weights in a dense but sparsely-populated 2D matrix and implements the
        forward pass as a single matrix-multiply. It uses a lot of RAM but
        performs few (large) ops.  `3` stores layer weights in a sparse tensor
        and implements the forward pass as a single sparse matrix-multiply.
          How to choose:
          `1`: large, dense models,
          `2`: small models,
          `3`: large, sparse models,  where "large" stands for large
            input/output activations (i.e. many `filters`, `input_filters`,
            large `input_size`, `output_size`), and "sparse" stands for few
            connections between inputs and outputs, i.e. small ratio `filters *
            input_filters * kernel_size / (input_size * strides)`, where inputs
            to and outputs of the layer are assumed to have shapes `(input_size,
            input_filters)`, `(output_size, filters)` respectively.  It is
            recommended to benchmark each in the setting of interest to pick the
            most efficient one (in terms of speed and memory usage). Correct
            choice of implementation can lead to dramatic speed improvements
            (e.g. 50X), potentially at the expense of RAM.  Also, only
            `padding="valid"` is supported by `implementation=1`.
  Input shape:
      3D tensor with shape: `(batch_size, steps, input_dim)`
  Output shape:
      3D tensor with shape: `(batch_size, new_steps, filters)` `steps` value
        might have changed due to padding or strides.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               implementation=1,
               **kwargs):
    super(LocallyConnected1D, self).__init__(**kwargs)
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(kernel_size, 1, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, 1, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    if self.padding != 'valid' and implementation == 1:
      raise ValueError('Invalid border mode for LocallyConnected1D '
                       '(only "valid" is supported if implementation is 1): ' +
                       padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.implementation = implementation
    self.input_spec = InputSpec(ndim=3)

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    if self.data_format == 'channels_first':
      input_dim, input_length = input_shape[1], input_shape[2]
    else:
      input_dim, input_length = input_shape[2], input_shape[1]

    if input_dim is None:
      raise ValueError(
          'Axis 2 of input should be fully-defined. '
          'Found shape:', input_shape)
    self.output_length = conv_utils.conv_output_length(input_length,
                                                       self.kernel_size[0],
                                                       self.padding,
                                                       self.strides[0])

    if self.implementation == 1:
      self.kernel_shape = (self.output_length, self.kernel_size[0] * input_dim,
                           self.filters)

      self.kernel = self.add_weight(
          shape=self.kernel_shape,
          initializer=self.kernel_initializer,
          name='kernel',
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint)

    elif self.implementation == 2:
      if self.data_format == 'channels_first':
        self.kernel_shape = (input_dim, input_length, self.filters,
                             self.output_length)
      else:
        self.kernel_shape = (input_length, input_dim, self.output_length,
                             self.filters)

      self.kernel = self.add_weight(
          shape=self.kernel_shape,
          initializer=self.kernel_initializer,
          name='kernel',
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint)

      self.kernel_mask = get_locallyconnected_mask(
          input_shape=(input_length,),
          kernel_shape=self.kernel_size,
          strides=self.strides,
          padding=self.padding,
          data_format=self.data_format,
      )

    elif self.implementation == 3:
      self.kernel_shape = (self.output_length * self.filters,
                           input_length * input_dim)

      self.kernel_idxs = sorted(
          conv_utils.conv_kernel_idxs(
              input_shape=(input_length,),
              kernel_shape=self.kernel_size,
              strides=self.strides,
              padding=self.padding,
              filters_in=input_dim,
              filters_out=self.filters,
              data_format=self.data_format))

      self.kernel = self.add_weight(
          shape=(len(self.kernel_idxs),),
          initializer=self.kernel_initializer,
          name='kernel',
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint)

    else:
      raise ValueError('Unrecognized implementation mode: %d.' %
                       self.implementation)

    if self.use_bias:
      self.bias = self.add_weight(
          shape=(self.output_length, self.filters),
          initializer=self.bias_initializer,
          name='bias',
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None

    if self.data_format == 'channels_first':
      self.input_spec = InputSpec(ndim=3, axes={1: input_dim})
    else:
      self.input_spec = InputSpec(ndim=3, axes={-1: input_dim})
    self.built = True

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if self.data_format == 'channels_first':
      input_length = input_shape[2]
    else:
      input_length = input_shape[1]

    length = conv_utils.conv_output_length(input_length, self.kernel_size[0],
                                           self.padding, self.strides[0])

    if self.data_format == 'channels_first':
      return (input_shape[0], self.filters, length)
    elif self.data_format == 'channels_last':
      return (input_shape[0], length, self.filters)

  def call(self, inputs):
    if self.implementation == 1:
      output = backend.local_conv(
          inputs, self.kernel, self.kernel_size, self.strides,
          (self.output_length,), self.data_format)

    elif self.implementation == 2:
      output = local_conv_matmul(inputs, self.kernel, self.kernel_mask,
                                 self.compute_output_shape(inputs.shape))

    elif self.implementation == 3:
      output = local_conv_sparse_matmul(inputs, self.kernel, self.kernel_idxs,
                                        self.kernel_shape,
                                        self.compute_output_shape(inputs.shape))

    else:
      raise ValueError('Unrecognized implementation mode: %d.' %
                       self.implementation)

    if self.use_bias:
      output = backend.bias_add(output, self.bias, data_format=self.data_format)

    output = self.activation(output)
    return output

  def get_config(self):
    config = {
        'filters':
            self.filters,
        'kernel_size':
            self.kernel_size,
        'strides':
            self.strides,
        'padding':
            self.padding,
        'data_format':
            self.data_format,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'implementation':
            self.implementation
    }
    base_config = super(LocallyConnected1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.LocallyConnected2D')
class LocallyConnected2D(Layer):
  """Locally-connected layer for 2D inputs.

  The `LocallyConnected2D` layer works similarly
  to the `Conv2D` layer, except that weights are unshared,
  that is, a different set of filters is applied at each
  different patch of the input.

  Note: layer attributes cannot be modified after the layer has been called
  once (except the `trainable` attribute).

  Examples:
  ```python
      # apply a 3x3 unshared weights convolution with 64 output filters on a
      32x32 image
      # with `data_format="channels_last"`:
      model = Sequential()
      model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
      # now model.output_shape == (None, 30, 30, 64)
      # notice that this layer will consume (30*30)*(3*3*3*64) + (30*30)*64
      parameters

      # add a 3x3 unshared weights convolution on top, with 32 output filters:
      model.add(LocallyConnected2D(32, (3, 3)))
      # now model.output_shape == (None, 28, 28, 32)
  ```

  Args:
      filters: Integer, the dimensionality of the output space (i.e. the number
        of output filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the width
        and height of the 2D convolution window. Can be a single integer to
        specify the same value for all spatial dimensions.
      strides: An integer or tuple/list of 2 integers, specifying the strides of
        the convolution along the width and height. Can be a single integer to
        specify the same value for all spatial dimensions.
      padding: Currently only support `"valid"` (case-insensitive). `"same"`
        will be supported in future. `"valid"` means no padding.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch, height, width,
        channels)` while `channels_first` corresponds to inputs with shape
        `(batch, channels, height, width)`. It defaults to the
        `image_data_format` value found in your Keras config file at
        `~/.keras/keras.json`. If you never set it, then it will be
        "channels_last".
      activation: Activation function to use. If you don't specify anything, no
        activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation").
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.
      implementation: implementation mode, either `1`, `2`, or `3`. `1` loops
        over input spatial locations to perform the forward pass. It is
        memory-efficient but performs a lot of (small) ops.  `2` stores layer
        weights in a dense but sparsely-populated 2D matrix and implements the
        forward pass as a single matrix-multiply. It uses a lot of RAM but
        performs few (large) ops.  `3` stores layer weights in a sparse tensor
        and implements the forward pass as a single sparse matrix-multiply.
          How to choose:
          `1`: large, dense models,
          `2`: small models,
          `3`: large, sparse models,  where "large" stands for large
            input/output activations (i.e. many `filters`, `input_filters`,
            large `np.prod(input_size)`, `np.prod(output_size)`), and "sparse"
            stands for few connections between inputs and outputs, i.e. small
            ratio `filters * input_filters * np.prod(kernel_size) /
            (np.prod(input_size) * np.prod(strides))`, where inputs to and
            outputs of the layer are assumed to have shapes `input_size +
            (input_filters,)`, `output_size + (filters,)` respectively.  It is
            recommended to benchmark each in the setting of interest to pick the
            most efficient one (in terms of speed and memory usage). Correct
            choice of implementation can lead to dramatic speed improvements
            (e.g. 50X), potentially at the expense of RAM.  Also, only
            `padding="valid"` is supported by `implementation=1`.
  Input shape:
      4D tensor with shape: `(samples, channels, rows, cols)` if
        data_format='channels_first'
      or 4D tensor with shape: `(samples, rows, cols, channels)` if
        data_format='channels_last'.
  Output shape:
      4D tensor with shape: `(samples, filters, new_rows, new_cols)` if
        data_format='channels_first'
      or 4D tensor with shape: `(samples, new_rows, new_cols, filters)` if
        data_format='channels_last'. `rows` and `cols` values might have changed
        due to padding.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               implementation=1,
               **kwargs):
    super(LocallyConnected2D, self).__init__(**kwargs)
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    if self.padding != 'valid' and implementation == 1:
      raise ValueError('Invalid border mode for LocallyConnected2D '
                       '(only "valid" is supported if implementation is 1): ' +
                       padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.implementation = implementation
    self.input_spec = InputSpec(ndim=4)

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    if self.data_format == 'channels_last':
      input_row, input_col = input_shape[1:-1]
      input_filter = input_shape[3]
    else:
      input_row, input_col = input_shape[2:]
      input_filter = input_shape[1]
    if input_row is None or input_col is None:
      raise ValueError('The spatial dimensions of the inputs to '
                       ' a LocallyConnected2D layer '
                       'should be fully-defined, but layer received '
                       'the inputs shape ' + str(input_shape))
    output_row = conv_utils.conv_output_length(input_row, self.kernel_size[0],
                                               self.padding, self.strides[0])
    output_col = conv_utils.conv_output_length(input_col, self.kernel_size[1],
                                               self.padding, self.strides[1])
    self.output_row = output_row
    self.output_col = output_col

    if self.implementation == 1:
      self.kernel_shape = (output_row * output_col, self.kernel_size[0] *
                           self.kernel_size[1] * input_filter, self.filters)

      self.kernel = self.add_weight(
          shape=self.kernel_shape,
          initializer=self.kernel_initializer,
          name='kernel',
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint)

    elif self.implementation == 2:
      if self.data_format == 'channels_first':
        self.kernel_shape = (input_filter, input_row, input_col, self.filters,
                             self.output_row, self.output_col)
      else:
        self.kernel_shape = (input_row, input_col, input_filter,
                             self.output_row, self.output_col, self.filters)

      self.kernel = self.add_weight(
          shape=self.kernel_shape,
          initializer=self.kernel_initializer,
          name='kernel',
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint)

      self.kernel_mask = get_locallyconnected_mask(
          input_shape=(input_row, input_col),
          kernel_shape=self.kernel_size,
          strides=self.strides,
          padding=self.padding,
          data_format=self.data_format,
      )

    elif self.implementation == 3:
      self.kernel_shape = (self.output_row * self.output_col * self.filters,
                           input_row * input_col * input_filter)

      self.kernel_idxs = sorted(
          conv_utils.conv_kernel_idxs(
              input_shape=(input_row, input_col),
              kernel_shape=self.kernel_size,
              strides=self.strides,
              padding=self.padding,
              filters_in=input_filter,
              filters_out=self.filters,
              data_format=self.data_format))

      self.kernel = self.add_weight(
          shape=(len(self.kernel_idxs),),
          initializer=self.kernel_initializer,
          name='kernel',
          regularizer=self.kernel_regularizer,
          constraint=self.kernel_constraint)

    else:
      raise ValueError('Unrecognized implementation mode: %d.' %
                       self.implementation)

    if self.use_bias:
      self.bias = self.add_weight(
          shape=(output_row, output_col, self.filters),
          initializer=self.bias_initializer,
          name='bias',
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    if self.data_format == 'channels_first':
      self.input_spec = InputSpec(ndim=4, axes={1: input_filter})
    else:
      self.input_spec = InputSpec(ndim=4, axes={-1: input_filter})
    self.built = True

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if self.data_format == 'channels_first':
      rows = input_shape[2]
      cols = input_shape[3]
    elif self.data_format == 'channels_last':
      rows = input_shape[1]
      cols = input_shape[2]

    rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                         self.padding, self.strides[0])
    cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                         self.padding, self.strides[1])

    if self.data_format == 'channels_first':
      return (input_shape[0], self.filters, rows, cols)
    elif self.data_format == 'channels_last':
      return (input_shape[0], rows, cols, self.filters)

  def call(self, inputs):
    if self.implementation == 1:
      output = backend.local_conv(
          inputs, self.kernel, self.kernel_size, self.strides,
          (self.output_row, self.output_col),
          self.data_format)

    elif self.implementation == 2:
      output = local_conv_matmul(inputs, self.kernel, self.kernel_mask,
                                 self.compute_output_shape(inputs.shape))

    elif self.implementation == 3:
      output = local_conv_sparse_matmul(inputs, self.kernel, self.kernel_idxs,
                                        self.kernel_shape,
                                        self.compute_output_shape(inputs.shape))

    else:
      raise ValueError('Unrecognized implementation mode: %d.' %
                       self.implementation)

    if self.use_bias:
      output = backend.bias_add(output, self.bias, data_format=self.data_format)

    output = self.activation(output)
    return output

  def get_config(self):
    config = {
        'filters':
            self.filters,
        'kernel_size':
            self.kernel_size,
        'strides':
            self.strides,
        'padding':
            self.padding,
        'data_format':
            self.data_format,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'implementation':
            self.implementation
    }
    base_config = super(LocallyConnected2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def get_locallyconnected_mask(input_shape, kernel_shape, strides, padding,
                              data_format):
  """Return a mask representing connectivity of a locally-connected operation.

  This method returns a masking numpy array of 0s and 1s (of type `np.float32`)
  that, when element-wise multiplied with a fully-connected weight tensor, masks
  out the weights between disconnected input-output pairs and thus implements
  local connectivity through a sparse fully-connected weight tensor.

  Assume an unshared convolution with given parameters is applied to an input
  having N spatial dimensions with `input_shape = (d_in1, ..., d_inN)`
  to produce an output with spatial shape `(d_out1, ..., d_outN)` (determined
  by layer parameters such as `strides`).

  This method returns a mask which can be broadcast-multiplied (element-wise)
  with a 2*(N+1)-D weight matrix (equivalent to a fully-connected layer between
  (N+1)-D activations (N spatial + 1 channel dimensions for input and output)
  to make it perform an unshared convolution with given `kernel_shape`,
  `strides`, `padding` and `data_format`.

  Args:
    input_shape: tuple of size N: `(d_in1, ..., d_inN)` spatial shape of the
      input.
    kernel_shape: tuple of size N, spatial shape of the convolutional kernel /
      receptive field.
    strides: tuple of size N, strides along each spatial dimension.
    padding: type of padding, string `"same"` or `"valid"`.
    data_format: a string, `"channels_first"` or `"channels_last"`.

  Returns:
    a `np.float32`-type `np.ndarray` of shape
    `(1, d_in1, ..., d_inN, 1, d_out1, ..., d_outN)`
    if `data_format == `"channels_first"`, or
    `(d_in1, ..., d_inN, 1, d_out1, ..., d_outN, 1)`
    if `data_format == "channels_last"`.

  Raises:
    ValueError: if `data_format` is neither `"channels_first"` nor
                `"channels_last"`.
  """
  mask = conv_utils.conv_kernel_mask(
      input_shape=input_shape,
      kernel_shape=kernel_shape,
      strides=strides,
      padding=padding)

  ndims = int(mask.ndim / 2)

  if data_format == 'channels_first':
    mask = np.expand_dims(mask, 0)
    mask = np.expand_dims(mask, -ndims - 1)

  elif data_format == 'channels_last':
    mask = np.expand_dims(mask, ndims)
    mask = np.expand_dims(mask, -1)

  else:
    raise ValueError('Unrecognized data_format: ' + str(data_format))

  return mask


def local_conv_matmul(inputs, kernel, kernel_mask, output_shape):
  """Apply N-D convolution with un-shared weights using a single matmul call.

  This method outputs `inputs . (kernel * kernel_mask)`
  (with `.` standing for matrix-multiply and `*` for element-wise multiply)
  and requires a precomputed `kernel_mask` to zero-out weights in `kernel` and
  hence perform the same operation as a convolution with un-shared
  (the remaining entries in `kernel`) weights. It also does the necessary
  reshapes to make `inputs` and `kernel` 2-D and `output` (N+2)-D.

  Args:
      inputs: (N+2)-D tensor with shape `(batch_size, channels_in, d_in1, ...,
        d_inN)` or `(batch_size, d_in1, ..., d_inN, channels_in)`.
      kernel: the unshared weights for N-D convolution,
          an (N+2)-D tensor of shape: `(d_in1, ..., d_inN, channels_in, d_out2,
            ..., d_outN, channels_out)` or `(channels_in, d_in1, ..., d_inN,
            channels_out, d_out2, ..., d_outN)`, with the ordering of channels
            and spatial dimensions matching that of the input. Each entry is the
            weight between a particular input and output location, similarly to
            a fully-connected weight matrix.
      kernel_mask: a float 0/1 mask tensor of shape: `(d_in1, ..., d_inN, 1,
        d_out2, ..., d_outN, 1)` or `(1, d_in1, ..., d_inN, 1, d_out2, ...,
        d_outN)`, with the ordering of singleton and spatial dimensions matching
        that of the input. Mask represents the connectivity pattern of the layer
        and is
           precomputed elsewhere based on layer parameters: stride, padding, and
             the receptive field shape.
      output_shape: a tuple of (N+2) elements representing the output shape:
        `(batch_size, channels_out, d_out1, ..., d_outN)` or `(batch_size,
        d_out1, ..., d_outN, channels_out)`, with the ordering of channels and
        spatial dimensions matching that of the input.

  Returns:
      Output (N+2)-D tensor with shape `output_shape`.
  """
  inputs_flat = backend.reshape(inputs, (backend.shape(inputs)[0], -1))

  kernel = kernel_mask * kernel
  kernel = make_2d(kernel, split_dim=backend.ndim(kernel) // 2)

  output_flat = math_ops.sparse_matmul(inputs_flat, kernel, b_is_sparse=True)
  output = backend.reshape(output_flat, [
      backend.shape(output_flat)[0],
  ] + output_shape.as_list()[1:])
  return output


def local_conv_sparse_matmul(inputs, kernel, kernel_idxs, kernel_shape,
                             output_shape):
  """Apply N-D convolution with un-shared weights using a single sparse matmul.

  This method outputs `inputs . tf.sparse.SparseTensor(indices=kernel_idxs,
  values=kernel, dense_shape=kernel_shape)`, with `.` standing for
  matrix-multiply. It also reshapes `inputs` to 2-D and `output` to (N+2)-D.

  Args:
      inputs: (N+2)-D tensor with shape `(batch_size, channels_in, d_in1, ...,
        d_inN)` or `(batch_size, d_in1, ..., d_inN, channels_in)`.
      kernel: a 1-D tensor with shape `(len(kernel_idxs),)` containing all the
        weights of the layer.
      kernel_idxs:  a list of integer tuples representing indices in a sparse
        matrix performing the un-shared convolution as a matrix-multiply.
      kernel_shape: a tuple `(input_size, output_size)`, where `input_size =
        channels_in * d_in1 * ... * d_inN` and `output_size = channels_out *
        d_out1 * ... * d_outN`.
      output_shape: a tuple of (N+2) elements representing the output shape:
        `(batch_size, channels_out, d_out1, ..., d_outN)` or `(batch_size,
        d_out1, ..., d_outN, channels_out)`, with the ordering of channels and
        spatial dimensions matching that of the input.

  Returns:
      Output (N+2)-D dense tensor with shape `output_shape`.
  """
  inputs_flat = backend.reshape(inputs, (backend.shape(inputs)[0], -1))
  output_flat = gen_sparse_ops.SparseTensorDenseMatMul(
      a_indices=kernel_idxs,
      a_values=kernel,
      a_shape=kernel_shape,
      b=inputs_flat,
      adjoint_b=True)
  output_flat_transpose = backend.transpose(output_flat)

  output_reshaped = backend.reshape(output_flat_transpose, [
      backend.shape(output_flat_transpose)[0],
  ] + output_shape.as_list()[1:])
  return output_reshaped


def make_2d(tensor, split_dim):
  """Reshapes an N-dimensional tensor into a 2D tensor.

  Dimensions before (excluding) and after (including) `split_dim` are grouped
  together.

  Args:
    tensor: a tensor of shape `(d0, ..., d(N-1))`.
    split_dim: an integer from 1 to N-1, index of the dimension to group
      dimensions before (excluding) and after (including).

  Returns:
    Tensor of shape
    `(d0 * ... * d(split_dim-1), d(split_dim) * ... * d(N-1))`.
  """
  shape = array_ops.shape(tensor)
  in_dims = shape[:split_dim]
  out_dims = shape[split_dim:]

  in_size = math_ops.reduce_prod(in_dims)
  out_size = math_ops.reduce_prod(out_dims)

  return array_ops.reshape(tensor, (in_size, out_size))
