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
"""Locally-connected layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import InputSpec
from tensorflow.python.keras.engine import Layer
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.layers.LocallyConnected1D')
class LocallyConnected1D(Layer):
  """Locally-connected layer for 1D inputs.

  The `LocallyConnected1D` layer works similarly to
  the `Conv1D` layer, except that weights are unshared,
  that is, a different set of filters is applied at each different patch
  of the input.

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

  Arguments:
      filters: Integer, the dimensionality of the output space
          (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of a single integer,
          specifying the length of the 1D convolution window.
      strides: An integer or tuple/list of a single integer,
          specifying the stride length of the convolution.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: Currently only supports `"valid"` (case-insensitive).
          `"same"` may be supported in the future.
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, length, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, length)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.

  Input shape:
      3D tensor with shape: `(batch_size, steps, input_dim)`

  Output shape:
      3D tensor with shape: `(batch_size, new_steps, filters)`
      `steps` value might have changed due to padding or strides.
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
               **kwargs):
    super(LocallyConnected1D, self).__init__(**kwargs)
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(kernel_size, 1, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, 1, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    if self.padding != 'valid':
      raise ValueError('Invalid border mode for LocallyConnected1D '
                       '(only "valid" is supported): ' + padding)
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
    self.input_spec = InputSpec(ndim=3)

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    if self.data_format == 'channels_first':
      input_dim, input_length = input_shape[1], input_shape[2]
    else:
      input_dim, input_length = input_shape[2], input_shape[1]

    if input_dim is None:
      raise ValueError('Axis 2 of input should be fully-defined. '
                       'Found shape:', input_shape)
    self.output_length = conv_utils.conv_output_length(
        input_length, self.kernel_size[0], self.padding, self.strides[0])
    self.kernel_shape = (self.output_length, self.kernel_size[0] * input_dim,
                         self.filters)
    self.kernel = self.add_weight(
        shape=self.kernel_shape,
        initializer=self.kernel_initializer,
        name='kernel',
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
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
    output = K.local_conv(inputs, self.kernel, self.kernel_size, self.strides,
                          (self.output_length,), self.data_format)

    if self.use_bias:
      output = K.bias_add(output, self.bias, data_format=self.data_format)

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
            constraints.serialize(self.bias_constraint)
    }
    base_config = super(LocallyConnected1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.LocallyConnected2D')
class LocallyConnected2D(Layer):
  """Locally-connected layer for 2D inputs.

  The `LocallyConnected2D` layer works similarly
  to the `Conv2D` layer, except that weights are unshared,
  that is, a different set of filters is applied at each
  different patch of the input.

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

  Arguments:
      filters: Integer, the dimensionality of the output space
          (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
          width and height of the 2D convolution window.
          Can be a single integer to specify the same value for
          all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the width and height.
          Can be a single integer to specify the same value for
          all spatial dimensions.
      padding: Currently only support `"valid"` (case-insensitive).
          `"same"` will be supported in future.
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
          the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.

  Input shape:
      4D tensor with shape:
      `(samples, channels, rows, cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, rows, cols, channels)` if data_format='channels_last'.

  Output shape:
      4D tensor with shape:
      `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
      `rows` and `cols` values might have changed due to padding.
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
               **kwargs):
    super(LocallyConnected2D, self).__init__(**kwargs)
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    if self.padding != 'valid':
      raise ValueError('Invalid border mode for LocallyConnected2D '
                       '(only "valid" is supported): ' + padding)
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
    self.kernel_shape = (
        output_row * output_col,
        self.kernel_size[0] * self.kernel_size[1] * input_filter, self.filters)
    self.kernel = self.add_weight(
        shape=self.kernel_shape,
        initializer=self.kernel_initializer,
        name='kernel',
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
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
    output = K.local_conv(inputs, self.kernel, self.kernel_size, self.strides,
                          (self.output_row, self.output_col), self.data_format)

    if self.use_bias:
      output = K.bias_add(output, self.bias, data_format=self.data_format)

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
            constraints.serialize(self.bias_constraint)
    }
    base_config = super(LocallyConnected2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
