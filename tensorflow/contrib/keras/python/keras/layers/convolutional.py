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
"""Keras convolution layers and image transformation layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.keras.python.keras import activations
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras import constraints
from tensorflow.contrib.keras.python.keras import initializers
from tensorflow.contrib.keras.python.keras import regularizers
from tensorflow.contrib.keras.python.keras.engine import InputSpec
from tensorflow.contrib.keras.python.keras.engine import Layer
# imports for backwards namespace compatibility
# pylint: disable=unused-import
from tensorflow.contrib.keras.python.keras.layers.pooling import AveragePooling1D
from tensorflow.contrib.keras.python.keras.layers.pooling import AveragePooling2D
from tensorflow.contrib.keras.python.keras.layers.pooling import AveragePooling3D
from tensorflow.contrib.keras.python.keras.layers.pooling import MaxPooling1D
from tensorflow.contrib.keras.python.keras.layers.pooling import MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers.pooling import MaxPooling3D
# pylint: enable=unused-import
from tensorflow.contrib.keras.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape


class _Conv(Layer):
  """Abstract nD convolution layer (private, used as implementation base).

  This layer creates a convolution kernel that is convolved
  with the layer input to produce a tensor of outputs.
  If `use_bias` is True, a bias vector is created and added to the outputs.
  Finally, if `activation` is not `None`,
  it is applied to the outputs as well.

  Arguments:
      rank: An integer, the rank of the convolution,
          e.g. "2" for 2D convolution.
      filters: Integer, the dimensionality of the output space
          (i.e. the number output of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
          dimensions of the convolution window.
      strides: An integer or tuple/list of n integers,
          specifying the strides of the convolution.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, ..., channels)` while `channels_first` corresponds to
          inputs with shape `(batch, channels, ...)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      dilation_rate: An integer or tuple/list of n integers, specifying
          the dilation rate to use for dilated convolution.
          Currently, specifying any `dilation_rate` value != 1 is
          incompatible with specifying any `strides` value != 1.
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
  """

  def __init__(self,
               rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
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
    super(_Conv, self).__init__(**kwargs)
    self.rank = rank
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank,
                                                  'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank,
                                                    'dilation_rate')
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.input_spec = InputSpec(ndim=self.rank + 2)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis]
    kernel_shape = self.kernel_size + (input_dim, self.filters)

    self.kernel = self.add_weight(
        kernel_shape,
        initializer=self.kernel_initializer,
        name='kernel',
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    if self.use_bias:
      self.bias = self.add_weight(
          (self.filters,),
          initializer=self.bias_initializer,
          name='bias',
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    # Set input spec.
    self.input_spec = InputSpec(
        ndim=self.rank + 2, axes={channel_axis: input_dim})
    self.built = True

  def call(self, inputs):
    if self.rank == 1:
      outputs = K.conv1d(
          inputs,
          self.kernel,
          strides=self.strides[0],
          padding=self.padding,
          data_format=self.data_format,
          dilation_rate=self.dilation_rate[0])
    if self.rank == 2:
      outputs = K.conv2d(
          inputs,
          self.kernel,
          strides=self.strides,
          padding=self.padding,
          data_format=self.data_format,
          dilation_rate=self.dilation_rate)
    if self.rank == 3:
      outputs = K.conv3d(
          inputs,
          self.kernel,
          strides=self.strides,
          padding=self.padding,
          data_format=self.data_format,
          dilation_rate=self.dilation_rate)

    if self.use_bias:
      outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                      [self.filters])
    else:
      space = input_shape[2:]
      new_space = []
      for i in range(len(space)):
        new_dim = conv_utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                      new_space)

  def get_config(self):
    config = {
        'rank':
            self.rank,
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
        'dilation_rate':
            self.dilation_rate,
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
    base_config = super(_Conv, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Conv1D(_Conv):
  """1D convolution layer (e.g.

  temporal convolution).

  This layer creates a convolution kernel that is convolved
  with the layer input over a single spatial (or temporal) dimension
  to produce a tensor of outputs.
  If `use_bias` is True, a bias vector is created and added to the outputs.
  Finally, if `activation` is not `None`,
  it is applied to the outputs as well.

  When using this layer as the first layer in a model,
  provide an `input_shape` argument
  (tuple of integers or `None`, e.g.
  `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
  or `(None, 128)` for variable-length sequences of 128-dimensional vectors.

  Arguments:
      filters: Integer, the dimensionality of the output space
          (i.e. the number output of filters in the convolution).
      kernel_size: An integer or tuple/list of a single integer,
          specifying the length of the 1D convolution window.
      strides: An integer or tuple/list of a single integer,
          specifying the stride length of the convolution.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
          `"causal"` results in causal (dilated) convolutions, e.g. output[t]
          does not depend on input[t+1:]. Useful when modeling temporal data
          where the model should not violate the temporal order.
          See [WaveNet: A Generative Model for Raw Audio, section
            2.1](https://arxiv.org/abs/1609.03499).
      dilation_rate: an integer or tuple/list of a single integer, specifying
          the dilation rate to use for dilated convolution.
          Currently, specifying any `dilation_rate` value != 1 is
          incompatible with specifying any `strides` value != 1.
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
               dilation_rate=1,
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
    super(Conv1D, self).__init__(
        rank=1,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format='channels_last',
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)
    self.input_spec = InputSpec(ndim=3)

  def get_config(self):
    config = super(Conv1D, self).get_config()
    config.pop('rank')
    config.pop('data_format')
    return config


class Conv2D(_Conv):
  """2D convolution layer (e.g. spatial convolution over images).

  This layer creates a convolution kernel that is convolved
  with the layer input to produce a tensor of
  outputs. If `use_bias` is True,
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers, does not include the sample axis),
  e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
  in `data_format="channels_last"`.

  Arguments:
      filters: Integer, the dimensionality of the output space
          (i.e. the number output of filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
          width and height of the 2D convolution window.
          Can be a single integer to specify the same value for
          all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the width and height.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
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
      dilation_rate: an integer or tuple/list of 2 integers, specifying
          the dilation rate to use for dilated convolution.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Currently, specifying any `dilation_rate` value != 1 is
          incompatible with specifying any stride value != 1.
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
               dilation_rate=(1, 1),
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
    super(Conv2D, self).__init__(
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)
    self.input_spec = InputSpec(ndim=4)

  def get_config(self):
    config = super(Conv2D, self).get_config()
    config.pop('rank')
    return config


class Conv3D(_Conv):
  """3D convolution layer (e.g.

  spatial convolution over volumes).

  This layer creates a convolution kernel that is convolved
  with the layer input to produce a tensor of
  outputs. If `use_bias` is True,
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers, does not include the sample axis),
  e.g. `input_shape=(128, 128, 128, 3)` for 128x128x128 volumes
  with a single channel,
  in `data_format="channels_last"`.

  Arguments:
      filters: Integer, the dimensionality of the output space
          (i.e. the number output of filters in the convolution).
      kernel_size: An integer or tuple/list of 3 integers, specifying the
          width and height of the 3D convolution window.
          Can be a single integer to specify the same value for
          all spatial dimensions.
      strides: An integer or tuple/list of 3 integers,
          specifying the strides of the convolution along each spatial
            dimension.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
          while `channels_first` corresponds to inputs with shape
          `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      dilation_rate: an integer or tuple/list of 3 integers, specifying
          the dilation rate to use for dilated convolution.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Currently, specifying any `dilation_rate` value != 1 is
          incompatible with specifying any stride value != 1.
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
      5D tensor with shape:
      `(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if
        data_format='channels_first'
      or 5D tensor with shape:
      `(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if
        data_format='channels_last'.

  Output shape:
      5D tensor with shape:
      `(samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if
        data_format='channels_first'
      or 5D tensor with shape:
      `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)` if
        data_format='channels_last'.
      `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have
        changed due to padding.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1, 1),
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
    super(Conv3D, self).__init__(
        rank=3,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)
    self.input_spec = InputSpec(ndim=5)

  def get_config(self):
    config = super(Conv3D, self).get_config()
    config.pop('rank')
    return config


class Conv2DTranspose(Conv2D):
  """Transposed convolution layer (sometimes called Deconvolution).

  The need for transposed convolutions generally arises
  from the desire to use a transformation going in the opposite direction
  of a normal convolution, i.e., from something that has the shape of the
  output of some convolution to something that has the shape of its input
  while maintaining a connectivity pattern that is compatible with
  said convolution.

  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers, does not include the sample axis),
  e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
  in `data_format="channels_last"`.

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
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
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
      dilation_rate: an integer or tuple/list of 2 integers, specifying
          the dilation rate to use for dilated convolution.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Currently, specifying any `dilation_rate` value != 1 is
          incompatible with specifying any stride value != 1.
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
      `(batch, channels, rows, cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(batch, rows, cols, channels)` if data_format='channels_last'.

  Output shape:
      4D tensor with shape:
      `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
      `rows` and `cols` values might have changed due to padding.

  References:
      - [A guide to convolution arithmetic for deep
        learning](https://arxiv.org/abs/1603.07285v1)
      - [Deconvolutional
        Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
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
    super(Conv2DTranspose, self).__init__(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)
    self.input_spec = InputSpec(ndim=4)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if len(input_shape) != 4:
      raise ValueError(
          'Inputs should have rank ' + str(4) + '; Received input shape:',
          str(input_shape))
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis]
    kernel_shape = self.kernel_size + (self.filters, input_dim)

    self.kernel = self.add_weight(
        kernel_shape,
        initializer=self.kernel_initializer,
        name='kernel',
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    if self.use_bias:
      self.bias = self.add_weight(
          (self.filters,),
          initializer=self.bias_initializer,
          name='bias',
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    # Set input spec.
    self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
    self.built = True

  def call(self, inputs):
    input_shape = K.shape(inputs)
    batch_size = input_shape[0]
    if self.data_format == 'channels_first':
      h_axis, w_axis = 2, 3
    else:
      h_axis, w_axis = 1, 2

    height, width = input_shape[h_axis], input_shape[w_axis]
    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides

    # Infer the dynamic output shape:
    out_height = conv_utils.deconv_length(height, stride_h, kernel_h,
                                          self.padding)
    out_width = conv_utils.deconv_length(width, stride_w, kernel_w,
                                         self.padding)
    if self.data_format == 'channels_first':
      output_shape = (batch_size, self.filters, out_height, out_width)
    else:
      output_shape = (batch_size, out_height, out_width, self.filters)

    outputs = K.conv2d_transpose(
        inputs,
        self.kernel,
        output_shape,
        self.strides,
        padding=self.padding,
        data_format=self.data_format)

    if self.bias:
      outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = list(input_shape)
    if self.data_format == 'channels_first':
      c_axis, h_axis, w_axis = 1, 2, 3
    else:
      c_axis, h_axis, w_axis = 3, 1, 2

    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides

    output_shape[c_axis] = self.filters
    output_shape[h_axis] = conv_utils.deconv_length(
        output_shape[h_axis], stride_h, kernel_h, self.padding)
    output_shape[w_axis] = conv_utils.deconv_length(
        output_shape[w_axis], stride_w, kernel_w, self.padding)
    return tensor_shape.TensorShape(output_shape)

  def get_config(self):
    config = super(Conv2DTranspose, self).get_config()
    config.pop('dilation_rate')
    return config


class SeparableConv2D(Conv2D):
  """Depthwise separable 2D convolution.

  Separable convolutions consist in first performing
  a depthwise spatial convolution
  (which acts on each input channel separately)
  followed by a pointwise convolution which mixes together the resulting
  output channels. The `depth_multiplier` argument controls how many
  output channels are generated per input channel in the depthwise step.

  Intuitively, separable convolutions can be understood as
  a way to factorize a convolution kernel into two smaller kernels,
  or as an extreme version of an Inception block.

  Arguments:
      filters: Integer, the dimensionality of the output space
          (i.e. the number output of filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
          width and height of the 2D convolution window.
          Can be a single integer to specify the same value for
          all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
          specifying the strides of the convolution along the width and height.
          Can be a single integer to specify the same value for
          all spatial dimensions.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
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
      depth_multiplier: The number of depthwise convolution output channels
          for each input channel.
          The total number of depthwise convolution output
          channels will be equal to `filterss_in * depth_multiplier`.
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      depthwise_initializer: Initializer for the depthwise kernel matrix.
      pointwise_initializer: Initializer for the pointwise kernel matrix.
      bias_initializer: Initializer for the bias vector.
      depthwise_regularizer: Regularizer function applied to
          the depthwise kernel matrix.
      pointwise_regularizer: Regularizer function applied to
          the depthwise kernel matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
          the output of the layer (its "activation")..
      depthwise_constraint: Constraint function applied to
          the depthwise kernel matrix.
      pointwise_constraint: Constraint function applied to
          the pointwise kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.

  Input shape:
      4D tensor with shape:
      `(batch, channels, rows, cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(batch, rows, cols, channels)` if data_format='channels_last'.

  Output shape:
      4D tensor with shape:
      `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
      `rows` and `cols` values might have changed due to padding.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               depth_multiplier=1,
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               pointwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               pointwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               pointwise_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(SeparableConv2D, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=activation,
        use_bias=use_bias,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        bias_constraint=bias_constraint,
        **kwargs)
    self.depth_multiplier = depth_multiplier
    self.depthwise_initializer = initializers.get(depthwise_initializer)
    self.pointwise_initializer = initializers.get(pointwise_initializer)
    self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
    self.pointwise_regularizer = regularizers.get(pointwise_regularizer)
    self.depthwise_constraint = constraints.get(depthwise_constraint)
    self.pointwise_constraint = constraints.get(pointwise_constraint)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if len(input_shape) < 4:
      raise ValueError('Inputs to `SeparableConv2D` should have rank 4. '
                       'Received input shape:', str(input_shape))
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = 3
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs to '
                       '`SeparableConv2D` '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    depthwise_kernel_shape = (self.kernel_size[0], self.kernel_size[1],
                              input_dim, self.depth_multiplier)
    pointwise_kernel_shape = (1, 1, self.depth_multiplier * input_dim,
                              self.filters)

    self.depthwise_kernel = self.add_weight(
        depthwise_kernel_shape,
        initializer=self.depthwise_initializer,
        name='depthwise_kernel',
        regularizer=self.depthwise_regularizer,
        constraint=self.depthwise_constraint)
    self.pointwise_kernel = self.add_weight(
        pointwise_kernel_shape,
        initializer=self.pointwise_initializer,
        name='pointwise_kernel',
        regularizer=self.pointwise_regularizer,
        constraint=self.pointwise_constraint)

    if self.use_bias:
      self.bias = self.add_weight(
          (self.filters,),
          initializer=self.bias_initializer,
          name='bias',
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    # Set input spec.
    self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
    self.built = True

  def call(self, inputs):
    outputs = K.separable_conv2d(
        inputs,
        self.depthwise_kernel,
        self.pointwise_kernel,
        data_format=self.data_format,
        strides=self.strides,
        padding=self.padding)

    if self.bias:
      outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      rows = input_shape[2]
      cols = input_shape[3]
    else:
      rows = input_shape[1]
      cols = input_shape[2]

    rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                         self.padding, self.strides[0])
    cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                         self.padding, self.strides[1])
    if self.data_format == 'channels_first':
      return tensor_shape.TensorShape(
          [input_shape[0], self.filters, rows, cols])
    else:
      return tensor_shape.TensorShape(
          [input_shape[0], rows, cols, self.filters])

  def get_config(self):
    config = super(SeparableConv2D, self).get_config()
    config.pop('kernel_initializer')
    config.pop('kernel_regularizer')
    config.pop('kernel_constraint')
    config['depth_multiplier'] = self.depth_multiplier
    config['depthwise_initializer'] = initializers.serialize(
        self.depthwise_initializer)
    config['pointwise_initializer'] = initializers.serialize(
        self.pointwise_initializer)
    config['depthwise_regularizer'] = regularizers.serialize(
        self.depthwise_regularizer)
    config['pointwise_regularizer'] = regularizers.serialize(
        self.pointwise_regularizer)
    config['depthwise_constraint'] = constraints.serialize(
        self.depthwise_constraint)
    config['pointwise_constraint'] = constraints.serialize(
        self.pointwise_constraint)
    return config


class UpSampling1D(Layer):
  """Upsampling layer for 1D inputs.

  Repeats each temporal step `size` times along the time axis.

  Arguments:
      size: integer. Upsampling factor.

  Input shape:
      3D tensor with shape: `(batch, steps, features)`.

  Output shape:
      3D tensor with shape: `(batch, upsampled_steps, features)`.
  """

  def __init__(self, size=2, **kwargs):
    super(UpSampling1D, self).__init__(**kwargs)
    self.size = int(size)
    self.input_spec = InputSpec(ndim=3)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    size = self.size * input_shape[1] if input_shape[1] is not None else None
    return tensor_shape.TensorShape([input_shape[0], size, input_shape[2]])

  def call(self, inputs):
    output = K.repeat_elements(inputs, self.size, axis=1)
    return output

  def get_config(self):
    config = {'size': self.size}
    base_config = super(UpSampling1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class UpSampling2D(Layer):
  """Upsampling layer for 2D inputs.

  Repeats the rows and columns of the data
  by size[0] and size[1] respectively.

  Arguments:
      size: int, or tuple of 2 integers.
          The upsampling factors for rows and columns.
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

  Input shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch, rows, cols, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch, channels, rows, cols)`

  Output shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch, upsampled_rows, upsampled_cols, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch, channels, upsampled_rows, upsampled_cols)`
  """

  def __init__(self, size=(2, 2), data_format=None, **kwargs):
    super(UpSampling2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.size = conv_utils.normalize_tuple(size, 2, 'size')
    self.input_spec = InputSpec(ndim=4)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      height = self.size[0] * input_shape[2] if input_shape[
          2] is not None else None
      width = self.size[1] * input_shape[3] if input_shape[
          3] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], height, width])
    else:
      height = self.size[0] * input_shape[1] if input_shape[
          1] is not None else None
      width = self.size[1] * input_shape[2] if input_shape[
          2] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], height, width, input_shape[3]])

  def call(self, inputs):
    return K.resize_images(inputs, self.size[0], self.size[1], self.data_format)

  def get_config(self):
    config = {'size': self.size, 'data_format': self.data_format}
    base_config = super(UpSampling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class UpSampling3D(Layer):
  """Upsampling layer for 3D inputs.

  Repeats the 1st, 2nd and 3rd dimensions
  of the data by size[0], size[1] and size[2] respectively.

  Arguments:
      size: int, or tuple of 3 integers.
          The upsampling factors for dim1, dim2 and dim3.
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
          while `channels_first` corresponds to inputs with shape
          `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".

  Input shape:
      5D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch, dim1, dim2, dim3, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch, channels, dim1, dim2, dim3)`

  Output shape:
      5D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)`
  """

  def __init__(self, size=(2, 2, 2), data_format=None, **kwargs):
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.size = conv_utils.normalize_tuple(size, 3, 'size')
    self.input_spec = InputSpec(ndim=5)
    super(UpSampling3D, self).__init__(**kwargs)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      dim1 = self.size[0] * input_shape[2] if input_shape[
          2] is not None else None
      dim2 = self.size[1] * input_shape[3] if input_shape[
          3] is not None else None
      dim3 = self.size[2] * input_shape[4] if input_shape[
          4] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], dim1, dim2, dim3])
    else:
      dim1 = self.size[0] * input_shape[1] if input_shape[
          1] is not None else None
      dim2 = self.size[1] * input_shape[2] if input_shape[
          2] is not None else None
      dim3 = self.size[2] * input_shape[3] if input_shape[
          3] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], dim1, dim2, dim3, input_shape[4]])

  def call(self, inputs):
    return K.resize_volumes(inputs, self.size[0], self.size[1], self.size[2],
                            self.data_format)

  def get_config(self):
    config = {'size': self.size, 'data_format': self.data_format}
    base_config = super(UpSampling3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class ZeroPadding1D(Layer):
  """Zero-padding layer for 1D input (e.g. temporal sequence).

  Arguments:
      padding: int, or tuple of int (length 2), or dictionary.
          - If int:
          How many zeros to add at the beginning and end of
          the padding dimension (axis 1).
          - If tuple of int (length 2):
          How many zeros to add at the beginning and at the end of
          the padding dimension (`(left_pad, right_pad)`).

  Input shape:
      3D tensor with shape `(batch, axis_to_pad, features)`

  Output shape:
      3D tensor with shape `(batch, padded_axis, features)`
  """

  def __init__(self, padding=1, **kwargs):
    super(ZeroPadding1D, self).__init__(**kwargs)
    self.padding = conv_utils.normalize_tuple(padding, 2, 'padding')
    self.input_spec = InputSpec(ndim=3)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    length = input_shape[1] + self.padding[0] + self.padding[1] if input_shape[
        1] is not None else None
    return tensor_shape.TensorShape([input_shape[0], length, input_shape[2]])

  def call(self, inputs):
    return K.temporal_padding(inputs, padding=self.padding)

  def get_config(self):
    config = {'padding': self.padding}
    base_config = super(ZeroPadding1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class ZeroPadding2D(Layer):
  """Zero-padding layer for 2D input (e.g. picture).

  This layer can add rows and columns or zeros
  at the top, bottom, left and right side of an image tensor.

  Arguments:
      padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
          - If int: the same symmetric padding
              is applied to width and height.
          - If tuple of 2 ints:
              interpreted as two different
              symmetric padding values for height and width:
              `(symmetric_height_pad, symmetric_width_pad)`.
          - If tuple of 2 tuples of 2 ints:
              interpreted as
              `((top_pad, bottom_pad), (left_pad, right_pad))`
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

  Input shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch, rows, cols, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch, channels, rows, cols)`

  Output shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch, padded_rows, padded_cols, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch, channels, padded_rows, padded_cols)`
  """

  def __init__(self, padding=(1, 1), data_format=None, **kwargs):
    super(ZeroPadding2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    if isinstance(padding, int):
      self.padding = ((padding, padding), (padding, padding))
    elif hasattr(padding, '__len__'):
      if len(padding) != 2:
        raise ValueError('`padding` should have two elements. '
                         'Found: ' + str(padding))
      height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                  '1st entry of padding')
      width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                 '2nd entry of padding')
      self.padding = (height_padding, width_padding)
    else:
      raise ValueError('`padding` should be either an int, '
                       'a tuple of 2 ints '
                       '(symmetric_height_pad, symmetric_width_pad), '
                       'or a tuple of 2 tuples of 2 ints '
                       '((top_pad, bottom_pad), (left_pad, right_pad)). '
                       'Found: ' + str(padding))
    self.input_spec = InputSpec(ndim=4)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      rows = input_shape[2] + self.padding[0][0] + self.padding[0][
          1] if input_shape[2] is not None else None
      cols = input_shape[3] + self.padding[1][0] + self.padding[1][
          1] if input_shape[3] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], rows, cols])
    else:
      rows = input_shape[1] + self.padding[0][0] + self.padding[0][
          1] if input_shape[1] is not None else None
      cols = input_shape[2] + self.padding[1][0] + self.padding[1][
          1] if input_shape[2] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], rows, cols, input_shape[3]])

  def call(self, inputs):
    return K.spatial_2d_padding(
        inputs, padding=self.padding, data_format=self.data_format)

  def get_config(self):
    config = {'padding': self.padding, 'data_format': self.data_format}
    base_config = super(ZeroPadding2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class ZeroPadding3D(Layer):
  """Zero-padding layer for 3D data (spatial or spatio-temporal).

  Arguments:
      padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
          - If int: the same symmetric padding
              is applied to width and height.
          - If tuple of 2 ints:
              interpreted as two different
              symmetric padding values for height and width:
              `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
          - If tuple of 2 tuples of 2 ints:
              interpreted as
              `((left_dim1_pad, right_dim1_pad), (left_dim2_pad,
                right_dim2_pad), (left_dim3_pad, right_dim3_pad))`
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
          while `channels_first` corresponds to inputs with shape
          `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".

  Input shape:
      5D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad,
            depth)`
      - If `data_format` is `"channels_first"`:
          `(batch, depth, first_axis_to_pad, second_axis_to_pad,
            third_axis_to_pad)`

  Output shape:
      5D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch, first_padded_axis, second_padded_axis, third_axis_to_pad,
            depth)`
      - If `data_format` is `"channels_first"`:
          `(batch, depth, first_padded_axis, second_padded_axis,
            third_axis_to_pad)`
  """

  def __init__(self, padding=(1, 1, 1), data_format=None, **kwargs):
    super(ZeroPadding3D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    if isinstance(padding, int):
      self.padding = ((padding, padding), (padding, padding), (padding,
                                                               padding))
    elif hasattr(padding, '__len__'):
      if len(padding) != 3:
        raise ValueError('`padding` should have 3 elements. '
                         'Found: ' + str(padding))
      dim1_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                '1st entry of padding')
      dim2_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                '2nd entry of padding')
      dim3_padding = conv_utils.normalize_tuple(padding[2], 2,
                                                '3rd entry of padding')
      self.padding = (dim1_padding, dim2_padding, dim3_padding)
    else:
      raise ValueError(
          '`padding` should be either an int, '
          'a tuple of 3 ints '
          '(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad), '
          'or a tuple of 3 tuples of 2 ints '
          '((left_dim1_pad, right_dim1_pad),'
          ' (left_dim2_pad, right_dim2_pad),'
          ' (left_dim3_pad, right_dim2_pad)). '
          'Found: ' + str(padding))
    self.input_spec = InputSpec(ndim=5)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      dim1 = input_shape[2] + 2 * self.padding[0][0] if input_shape[
          2] is not None else None
      dim2 = input_shape[3] + 2 * self.padding[1][0] if input_shape[
          3] is not None else None
      dim3 = input_shape[4] + 2 * self.padding[2][0] if input_shape[
          4] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], dim1, dim2, dim3])
    else:
      dim1 = input_shape[1] + 2 * self.padding[0][1] if input_shape[
          1] is not None else None
      dim2 = input_shape[2] + 2 * self.padding[1][1] if input_shape[
          2] is not None else None
      dim3 = input_shape[3] + 2 * self.padding[2][1] if input_shape[
          3] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], dim1, dim2, dim3, input_shape[4]])

  def call(self, inputs):
    return K.spatial_3d_padding(
        inputs, padding=self.padding, data_format=self.data_format)

  def get_config(self):
    config = {'padding': self.padding, 'data_format': self.data_format}
    base_config = super(ZeroPadding3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Cropping1D(Layer):
  """Cropping layer for 1D input (e.g. temporal sequence).

  It crops along the time dimension (axis 1).

  Arguments:
      cropping: int or tuple of int (length 2)
          How many units should be trimmed off at the beginning and end of
          the cropping dimension (axis 1).
          If a single int is provided,
          the same value will be used for both.

  Input shape:
      3D tensor with shape `(batch, axis_to_crop, features)`

  Output shape:
      3D tensor with shape `(batch, cropped_axis, features)`
  """

  def __init__(self, cropping=(1, 1), **kwargs):
    super(Cropping1D, self).__init__(**kwargs)
    self.cropping = conv_utils.normalize_tuple(cropping, 2, 'cropping')
    self.input_spec = InputSpec(ndim=3)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if input_shape[1] is not None:
      length = input_shape[1] - self.cropping[0] - self.cropping[1]
    else:
      length = None
    return tensor_shape.TensorShape([input_shape[0], length, input_shape[2]])

  def call(self, inputs):
    if self.cropping[1] == 0:
      return inputs[:, self.cropping[0]:, :]
    else:
      return inputs[:, self.cropping[0]:-self.cropping[1], :]

  def get_config(self):
    config = {'cropping': self.cropping}
    base_config = super(Cropping1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Cropping2D(Layer):
  """Cropping layer for 2D input (e.g. picture).

  It crops along spatial dimensions, i.e. width and height.

  Arguments:
      cropping: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
          - If int: the same symmetric cropping
              is applied to width and height.
          - If tuple of 2 ints:
              interpreted as two different
              symmetric cropping values for height and width:
              `(symmetric_height_crop, symmetric_width_crop)`.
          - If tuple of 2 tuples of 2 ints:
              interpreted as
              `((top_crop, bottom_crop), (left_crop, right_crop))`
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

  Input shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch, rows, cols, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch, channels, rows, cols)`

  Output shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch, cropped_rows, cropped_cols, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch, channels, cropped_rows, cropped_cols)`

  Examples:

  ```python
      # Crop the input 2D images or feature maps
      model = Sequential()
      model.add(Cropping2D(cropping=((2, 2), (4, 4)),
                           input_shape=(28, 28, 3)))
      # now model.output_shape == (None, 24, 20, 3)
      model.add(Conv2D(64, (3, 3), padding='same))
      model.add(Cropping2D(cropping=((2, 2), (2, 2))))
      # now model.output_shape == (None, 20, 16. 64)
  ```
  """

  def __init__(self, cropping=((0, 0), (0, 0)), data_format=None, **kwargs):
    super(Cropping2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    if isinstance(cropping, int):
      self.cropping = ((cropping, cropping), (cropping, cropping))
    elif hasattr(cropping, '__len__'):
      if len(cropping) != 2:
        raise ValueError('`cropping` should have two elements. '
                         'Found: ' + str(cropping))
      height_cropping = conv_utils.normalize_tuple(cropping[0], 2,
                                                   '1st entry of cropping')
      width_cropping = conv_utils.normalize_tuple(cropping[1], 2,
                                                  '2nd entry of cropping')
      self.cropping = (height_cropping, width_cropping)
    else:
      raise ValueError('`cropping` should be either an int, '
                       'a tuple of 2 ints '
                       '(symmetric_height_crop, symmetric_width_crop), '
                       'or a tuple of 2 tuples of 2 ints '
                       '((top_crop, bottom_crop), (left_crop, right_crop)). '
                       'Found: ' + str(cropping))
    self.input_spec = InputSpec(ndim=4)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    # pylint: disable=invalid-unary-operand-type
    if self.data_format == 'channels_first':
      return tensor_shape.TensorShape([
          input_shape[0], input_shape[1],
          input_shape[2] - self.cropping[0][0] - self.cropping[0][1]
          if input_shape[2] else None,
          input_shape[3] - self.cropping[1][0] - self.cropping[1][1]
          if input_shape[3] else None
      ])
    else:
      return tensor_shape.TensorShape([
          input_shape[0],
          input_shape[1] - self.cropping[0][0] - self.cropping[0][1]
          if input_shape[1] else None,
          input_shape[2] - self.cropping[1][0] - self.cropping[1][1]
          if input_shape[2] else None, input_shape[3]
      ])
    # pylint: enable=invalid-unary-operand-type

  def call(self, inputs):
    # pylint: disable=invalid-unary-operand-type
    if self.data_format == 'channels_first':
      if self.cropping[0][1] == self.cropping[1][1] == 0:
        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:]
      elif self.cropping[0][1] == 0:
        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:
                      -self.cropping[1][1]]
      elif self.cropping[1][1] == 0:
        return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
                      self.cropping[1][0]:]
      return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
                    self.cropping[1][0]:-self.cropping[1][1]]
    else:
      if self.cropping[0][1] == self.cropping[1][1] == 0:
        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:, :]
      elif self.cropping[0][1] == 0:
        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:
                      -self.cropping[1][1], :]
      elif self.cropping[1][1] == 0:
        return inputs[:, self.cropping[0][0]:-self.cropping[0][1],
                      self.cropping[1][0]:, :]
      return inputs[:, self.cropping[0][0]:-self.cropping[0][1], self.cropping[
          1][0]:-self.cropping[1][1], :]  # pylint: disable=invalid-unary-operand-type
    # pylint: enable=invalid-unary-operand-type

  def get_config(self):
    config = {'cropping': self.cropping, 'data_format': self.data_format}
    base_config = super(Cropping2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Cropping3D(Layer):
  """Cropping layer for 3D data (e.g.

  spatial or spatio-temporal).

  Arguments:
      cropping: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
          - If int: the same symmetric cropping
              is applied to width and height.
          - If tuple of 2 ints:
              interpreted as two different
              symmetric cropping values for height and width:
              `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`.
          - If tuple of 2 tuples of 2 ints:
              interpreted as
              `((left_dim1_crop, right_dim1_crop), (left_dim2_crop,
                right_dim2_crop), (left_dim3_crop, right_dim3_crop))`
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
          while `channels_first` corresponds to inputs with shape
          `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".

  Input shape:
      5D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop,
            depth)`
      - If `data_format` is `"channels_first"`:
          `(batch, depth, first_axis_to_crop, second_axis_to_crop,
            third_axis_to_crop)`

  Output shape:
      5D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch, first_cropped_axis, second_cropped_axis, third_cropped_axis,
            depth)`
      - If `data_format` is `"channels_first"`:
          `(batch, depth, first_cropped_axis, second_cropped_axis,
            third_cropped_axis)`
  """

  def __init__(self,
               cropping=((1, 1), (1, 1), (1, 1)),
               data_format=None,
               **kwargs):
    super(Cropping3D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    if isinstance(cropping, int):
      self.cropping = ((cropping, cropping), (cropping, cropping), (cropping,
                                                                    cropping))
    elif hasattr(cropping, '__len__'):
      if len(cropping) != 3:
        raise ValueError('`cropping` should have 3 elements. '
                         'Found: ' + str(cropping))
      dim1_cropping = conv_utils.normalize_tuple(cropping[0], 2,
                                                 '1st entry of cropping')
      dim2_cropping = conv_utils.normalize_tuple(cropping[1], 2,
                                                 '2nd entry of cropping')
      dim3_cropping = conv_utils.normalize_tuple(cropping[2], 2,
                                                 '3rd entry of cropping')
      self.cropping = (dim1_cropping, dim2_cropping, dim3_cropping)
    else:
      raise ValueError(
          '`cropping` should be either an int, '
          'a tuple of 3 ints '
          '(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop), '
          'or a tuple of 3 tuples of 2 ints '
          '((left_dim1_crop, right_dim1_crop),'
          ' (left_dim2_crop, right_dim2_crop),'
          ' (left_dim3_crop, right_dim2_crop)). '
          'Found: ' + str(cropping))
    self.input_spec = InputSpec(ndim=5)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    # pylint: disable=invalid-unary-operand-type
    if self.data_format == 'channels_first':
      dim1 = input_shape[2] - self.cropping[0][0] - self.cropping[0][
          1] if input_shape[2] is not None else None
      dim2 = input_shape[3] - self.cropping[1][0] - self.cropping[1][
          1] if input_shape[3] is not None else None
      dim3 = input_shape[4] - self.cropping[2][0] - self.cropping[2][
          1] if input_shape[4] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], dim1, dim2, dim3])
    else:
      dim1 = input_shape[1] - self.cropping[0][0] - self.cropping[0][
          1] if input_shape[1] is not None else None
      dim2 = input_shape[2] - self.cropping[1][0] - self.cropping[1][
          1] if input_shape[2] is not None else None
      dim3 = input_shape[3] - self.cropping[2][0] - self.cropping[2][
          1] if input_shape[3] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], dim1, dim2, dim3, input_shape[4]])
    # pylint: enable=invalid-unary-operand-type

  def call(self, inputs):
    # pylint: disable=invalid-unary-operand-type
    if self.data_format == 'channels_first':
      if self.cropping[0][1] == self.cropping[1][1] == self.cropping[2][1] == 0:
        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:,
                      self.cropping[2][0]:]
      elif self.cropping[0][1] == self.cropping[1][1] == 0:
        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:,
                      self.cropping[2][0]:-self.cropping[2][1]]
      elif self.cropping[1][1] == self.cropping[2][1] == 0:
        return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
                      self.cropping[1][0]:, self.cropping[2][0]:]
      elif self.cropping[0][1] == self.cropping[2][1] == 0:
        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:
                      -self.cropping[1][1], self.cropping[2][0]:]
      elif self.cropping[0][1] == 0:
        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][
            0]:-self.cropping[1][1], self.cropping[2][0]:-self.cropping[2][1]]
      elif self.cropping[1][1] == 0:
        return inputs[:, :, self.cropping[0][0]:-self.cropping[0][
            1], self.cropping[1][0]:, self.cropping[2][0]:-self.cropping[2][1]]
      elif self.cropping[2][1] == 0:
        return inputs[:, :, self.cropping[0][0]:-self.cropping[0][
            1], self.cropping[1][0]:-self.cropping[1][1], self.cropping[2][0]:]
      return inputs[:, :, self.cropping[0][0]:-self.cropping[0][
          1], self.cropping[1][0]:-self.cropping[1][1], self.cropping[2][0]:
                    -self.cropping[2][1]]
    else:
      if self.cropping[0][1] == self.cropping[1][1] == self.cropping[2][1] == 0:
        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:,
                      self.cropping[2][0]:, :]
      elif self.cropping[0][1] == self.cropping[1][1] == 0:
        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:,
                      self.cropping[2][0]:-self.cropping[2][1], :]
      elif self.cropping[1][1] == self.cropping[2][1] == 0:
        return inputs[:, self.cropping[0][0]:-self.cropping[0][1],
                      self.cropping[1][0]:, self.cropping[2][0]:, :]
      elif self.cropping[0][1] == self.cropping[2][1] == 0:
        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:
                      -self.cropping[1][1], self.cropping[2][0]:, :]
      elif self.cropping[0][1] == 0:
        return inputs[:, self.cropping[0][0]:, self.cropping[1][
            0]:-self.cropping[1][1], self.cropping[2][0]:-self.cropping[2][
                1], :]
      elif self.cropping[1][1] == 0:
        return inputs[:, self.cropping[0][0]:-self.cropping[0][
            1], self.cropping[1][0]:, self.cropping[2][0]:-self.cropping[2][
                1], :]
      elif self.cropping[2][1] == 0:
        return inputs[:, self.cropping[0][0]:-self.cropping[0][
            1], self.cropping[1][0]:-self.cropping[1][1], self.cropping[2][
                0]:, :]
      return inputs[:, self.cropping[0][0]:-self.cropping[0][1], self.cropping[
          1][0]:-self.cropping[1][1], self.cropping[2][0]:-self.cropping[2][  # pylint: disable=invalid-unary-operand-type
              1], :]
    # pylint: enable=invalid-unary-operand-type

  def get_config(self):
    config = {'cropping': self.cropping, 'data_format': self.data_format}
    base_config = super(Cropping3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# Aliases

Convolution1D = Conv1D
Convolution2D = Conv2D
Convolution3D = Conv3D
SeparableConvolution2D = SeparableConv2D
Convolution2DTranspose = Conv2DTranspose
Deconvolution2D = Deconv2D = Conv2DTranspose
