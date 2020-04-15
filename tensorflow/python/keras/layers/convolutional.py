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

from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
# imports for backwards namespace compatibility
# pylint: disable=unused-import
from tensorflow.python.keras.layers.pooling import AveragePooling1D
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.layers.pooling import AveragePooling3D
from tensorflow.python.keras.layers.pooling import MaxPooling1D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.pooling import MaxPooling3D
# pylint: enable=unused-import
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.tf_export import keras_export
# pylint: disable=g-classes-have-attributes


class Conv(Layer):
  """Abstract N-D convolution layer (private, used as implementation base).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  Note: layer attributes cannot be modified after the layer has been called
  once (except the `trainable` attribute).

  Arguments:
    rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of n integers, specifying the
      length of the convolution window.
    strides: An integer or tuple/list of n integers,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch_size, channels, ...)`.
    dilation_rate: An integer or tuple/list of n integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default
      initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` the weights of this layer will be marked as
      trainable (and listed in `layer.trainable_weights`).
    name: A string, the name of the layer.
  """

  def __init__(self, rank,
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
               trainable=True,
               name=None,
               **kwargs):
    super(Conv, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)
    self.rank = rank
    if filters is not None and not isinstance(filters, int):
      filters = int(filters)
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(
        kernel_size, rank, 'kernel_size')
    if not all(self.kernel_size):
      raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                       'Received: %s' % (kernel_size,))
    self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    if (self.padding == 'causal' and not isinstance(self,
                                                    (Conv1D, SeparableConv1D))):
      raise ValueError('Causal padding is only supported for `Conv1D`'
                       'and ``SeparableConv1D`.')
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(
        dilation_rate, rank, 'dilation_rate')
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.input_spec = InputSpec(ndim=self.rank + 2)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_channel = self._get_input_channel(input_shape)
    kernel_shape = self.kernel_size + (input_channel, self.filters)

    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    channel_axis = self._get_channel_axis()
    self.input_spec = InputSpec(ndim=self.rank + 2,
                                axes={channel_axis: input_channel})

    self._build_conv_op_input_shape = input_shape
    self._build_input_channel = input_channel
    self._padding_op = self._get_padding_op()
    self._conv_op_data_format = conv_utils.convert_data_format(
        self.data_format, self.rank + 2)
    self._convolution_op = nn_ops.Convolution(
        input_shape,
        filter_shape=self.kernel.shape,
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=self._padding_op,
        data_format=self._conv_op_data_format)
    self.built = True

  def call(self, inputs):
    if self._recreate_conv_op(inputs):
      self._convolution_op = nn_ops.Convolution(
          inputs.get_shape(),
          filter_shape=self.kernel.shape,
          dilation_rate=self.dilation_rate,
          strides=self.strides,
          padding=self._padding_op,
          data_format=self._conv_op_data_format)
      self._build_conv_op_input_shape = inputs.get_shape()

    # Apply causal padding to inputs for Conv1D.
    if self.padding == 'causal' and self.__class__.__name__ == 'Conv1D':
      inputs = array_ops.pad(inputs, self._compute_causal_padding())

    outputs = self._convolution_op(inputs, self.kernel)

    if self.use_bias:
      if self.data_format == 'channels_first':
        if self.rank == 1:
          # nn.bias_add does not accept a 1D input tensor.
          bias = array_ops.reshape(self.bias, (1, self.filters, 1))
          outputs += bias
        else:
          outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
      else:
        outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def _spatial_output_shape(self, spatial_input_shape):
    return [
        conv_utils.conv_output_length(
            length,
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        for i, length in enumerate(spatial_input_shape)
    ]

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      return tensor_shape.TensorShape(
          [input_shape[0]] + self._spatial_output_shape(input_shape[1:-1]) +
          [self.filters])
    else:
      return tensor_shape.TensorShape(
          [input_shape[0], self.filters] +
          self._spatial_output_shape(input_shape[2:]))

  def get_config(self):
    config = {
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'strides': self.strides,
        'padding': self.padding,
        'data_format': self.data_format,
        'dilation_rate': self.dilation_rate,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(Conv, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _compute_causal_padding(self):
    """Calculates padding for 'causal' option for 1-d conv layers."""
    left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
    if self.data_format == 'channels_last':
      causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
    else:
      causal_padding = [[0, 0], [0, 0], [left_pad, 0]]
    return causal_padding

  def _get_channel_axis(self):
    if self.data_format == 'channels_first':
      return 1
    else:
      return -1

  def _get_input_channel(self, input_shape):
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    return int(input_shape[channel_axis])

  def _get_padding_op(self):
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    if not isinstance(op_padding, (list, tuple)):
      op_padding = op_padding.upper()
    return op_padding

  def _recreate_conv_op(self, inputs):
    """Recreate conv_op if necessary.

    Check if the input_shape in call() is different from that in build().
    For the values that are not None, if they are different, recreate
    the _convolution_op to avoid the stateful behavior.

    Args:
      inputs: The input data to call() method.

    Returns:
      `True` or `False` to indicate whether to recreate the conv_op.
    """
    call_input_shape = inputs.get_shape()
    for axis in range(1, len(call_input_shape)):
      if (call_input_shape[axis] is not None
          and self._build_conv_op_input_shape[axis] is not None
          and call_input_shape[axis] != self._build_conv_op_input_shape[axis]):
        return True
    return False


@keras_export('keras.layers.Conv1D', 'keras.layers.Convolution1D')
class Conv1D(Conv):
  """1D convolution layer (e.g. temporal convolution).

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

  Examples:

  >>> # The inputs are 128-length vectors with 10 timesteps, and the batch size
  >>> # is 4.
  >>> input_shape = (4, 10, 128)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv1D(
  ... 32, 3, activation='relu',input_shape=input_shape)(x)
  >>> print(y.shape)
  (4, 8, 32)

  Arguments:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of a single integer,
      specifying the length of the 1D convolution window.
    strides: An integer or tuple/list of a single integer,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
      `"causal"` results in causal (dilated) convolutions, e.g. `output[t]`
      does not depend on `input[t+1:]`. Useful when modeling temporal data
      where the model should not violate the temporal order.
      See [WaveNet: A Generative Model for Raw Audio, section
        2.1](https://arxiv.org/abs/1609.03499).
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
    dilation_rate: an integer or tuple/list of a single integer, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix (
      see `keras.initializers`).
    bias_initializer: Initializer for the bias vector (
      see `keras.initializers`).
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation") (
      see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).

  Input shape:
    3D tensor with shape: `(batch_size, steps, input_dim)`

  Output shape:
    3D tensor with shape: `(batch_size, new_steps, filters)`
      `steps` value might have changed due to padding or strides.

  Returns:
    A tensor of rank 3 representing
    `activation(conv1d(inputs, kernel) + bias)`.

  Raises:
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
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
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)


@keras_export('keras.layers.Conv2D', 'keras.layers.Convolution2D')
class Conv2D(Conv):
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

  Examples:

  >>> # The inputs are 28x28 RGB images with `channels_last` and the batch
  >>> # size is 4.
  >>> input_shape = (4, 28, 28, 3)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv2D(
  ... 2, 3, activation='relu', input_shape=input_shape)(x)
  >>> print(y.shape)
  (4, 26, 26, 2)

  >>> # With `dilation_rate` as 2.
  >>> input_shape = (4, 28, 28, 3)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv2D(
  ... 2, 3, activation='relu', dilation_rate=2, input_shape=input_shape)(x)
  >>> print(y.shape)
  (4, 24, 24, 2)

  >>> # With `padding` as "same".
  >>> input_shape = (4, 28, 28, 3)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv2D(
  ... 2, 3, activation='relu', padding="same", input_shape=input_shape)(x)
  >>> print(y.shape)
  (4, 28, 28, 2)


  Arguments:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
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
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix (
      see `keras.initializers`).
    bias_initializer: Initializer for the bias vector (
      see `keras.initializers`).
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation") (
      see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).

  Input shape:
    4D tensor with shape:
    `(batch_size, channels, rows, cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(batch_size, rows, cols, channels)` if data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `(batch_size, filters, new_rows, new_cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(batch_size, new_rows, new_cols, filters)` if data_format='channels_last'.
    `rows` and `cols` values might have changed due to padding.

  Returns:
    A tensor of rank 4 representing
    `activation(conv2d(inputs, kernel) + bias)`.

  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.
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
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)


@keras_export('keras.layers.Conv3D', 'keras.layers.Convolution3D')
class Conv3D(Conv):
  """3D convolution layer (e.g. spatial convolution over volumes).

  This layer creates a convolution kernel that is convolved
  with the layer input to produce a tensor of
  outputs. If `use_bias` is True,
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers, does not include the sample axis),
  e.g. `input_shape=(128, 128, 128, 1)` for 128x128x128 volumes
  with a single channel,
  in `data_format="channels_last"`.

  Examples:

  >>> # The inputs are 28x28x28 volumes with a single channel, and the
  >>> # batch size is 4
  >>> input_shape =(4, 28, 28, 28, 1)
  >>> x = tf.random.normal(input_shape)
  >>> y = tf.keras.layers.Conv3D(
  ... 2, 3, activation='relu', input_shape=input_shape)(x)
  >>> print(y.shape)
  (4, 26, 26, 26, 2)

  Arguments:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 3 integers, specifying the
      depth, height and width of the 3D convolution window.
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
      `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      while `channels_first` corresponds to inputs with shape
      `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
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
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix (
      see `keras.initializers`).
    bias_initializer: Initializer for the bias vector (
      see `keras.initializers`).
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix (
      see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation") (
      see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).

  Input shape:
    5D tensor with shape:
    `(batch_size, channels, conv_dim1, conv_dim2, conv_dim3)` if
      data_format='channels_first'
    or 5D tensor with shape:
    `(batch_size, conv_dim1, conv_dim2, conv_dim3, channels)` if
      data_format='channels_last'.

  Output shape:
    5D tensor with shape:
    `(batch_size, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if
      data_format='channels_first'
    or 5D tensor with shape:
    `(batch_size, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)` if
      data_format='channels_last'.
    `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have
      changed due to padding.

  Returns:
    A tensor of rank 5 representing
    `activation(conv3d(inputs, kernel) + bias)`.

  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.
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
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)


@keras_export('keras.layers.Conv1DTranspose',
              'keras.layers.Convolution1DTranspose')
class Conv1DTranspose(Conv1D):
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
  e.g. `input_shape=(128, 3)` for data with 128 time steps and 3 channels.

  Arguments:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer length of the 1D convolution window.
    strides: An integer specifying the stride of the convolution along the
      time dimension. Specifying a stride value != 1 is incompatible with
      specifying a `dilation_rate` value != 1. Defaults to 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    output_padding: An integer specifying the amount of padding along
      the time dimension of the output tensor.
      The amount of output padding must be lower than the stride.
      If set to `None` (default), the output shape is inferred.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch_size, channels, length)`.
    dilation_rate: an integer, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying a `dilation_rate` value != 1 is
      incompatible with specifying a stride value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix (
      see `keras.initializers`).
    bias_initializer: Initializer for the bias vector (
      see `keras.initializers`).
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation") (see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).

  Input shape:
    3D tensor with shape:
    `(batch_size, steps, channels)`

  Output shape:
    3D tensor with shape:
    `(batch_size, new_steps, filters)`
    If `output_padding` is specified:
    ```
    new_timesteps = ((timesteps - 1) * strides + kernel_size -
    2 * padding + output_padding)
    ```

  Returns:
    A tensor of rank 3 representing
    `activation(conv1dtranspose(inputs, kernel) + bias)`.

  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.

  References:
    - [A guide to convolution arithmetic for deep learning](
      https://arxiv.org/abs/1603.07285v1)
    - [Deconvolutional Networks](
      https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               output_padding=None,
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
    super(Conv1DTranspose, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

    self.output_padding = output_padding
    if self.output_padding is not None:
      self.output_padding = conv_utils.normalize_tuple(
          self.output_padding, 1, 'output_padding')
      for stride, out_pad in zip(self.strides, self.output_padding):
        if out_pad >= stride:
          raise ValueError('Stride ' + str(self.strides) + ' must be '
                           'greater than output padding ' +
                           str(self.output_padding))

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if len(input_shape) != 3:
      raise ValueError('Inputs should have rank 3. Received input shape: ' +
                       str(input_shape))
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
    kernel_shape = self.kernel_size + (self.filters, input_dim)

    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    inputs_shape = array_ops.shape(inputs)
    batch_size = inputs_shape[0]
    if self.data_format == 'channels_first':
      t_axis = 2
    else:
      t_axis = 1

    length = inputs_shape[t_axis]
    if self.output_padding is None:
      output_padding = None
    else:
      output_padding = self.output_padding[0]

    # Infer the dynamic output shape:
    out_length = conv_utils.deconv_output_length(
        length, self.kernel_size[0], padding=self.padding,
        output_padding=output_padding, stride=self.strides[0],
        dilation=self.dilation_rate[0])
    if self.data_format == 'channels_first':
      output_shape = (batch_size, self.filters, out_length)
    else:
      output_shape = (batch_size, out_length, self.filters)
    data_format = conv_utils.convert_data_format(self.data_format, ndim=3)

    output_shape_tensor = array_ops.stack(output_shape)
    outputs = nn_ops.conv1d_transpose(
        inputs,
        self.kernel,
        output_shape_tensor,
        strides=self.strides,
        padding=self.padding.upper(),
        data_format=data_format,
        dilations=self.dilation_rate)

    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(inputs.shape)
      outputs.set_shape(out_shape)

    if self.use_bias:
      outputs = nn.bias_add(
          outputs,
          self.bias,
          data_format=data_format)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = list(input_shape)
    if self.data_format == 'channels_first':
      c_axis, t_axis = 1, 2
    else:
      c_axis, t_axis = 2, 1

    if self.output_padding is None:
      output_padding = None
    else:
      output_padding = self.output_padding[0]
    output_shape[c_axis] = self.filters
    output_shape[t_axis] = conv_utils.deconv_output_length(
        output_shape[t_axis],
        self.kernel_size[0],
        padding=self.padding,
        output_padding=output_padding,
        stride=self.strides[0],
        dilation=self.dilation_rate[0])
    return tensor_shape.TensorShape(output_shape)

  def get_config(self):
    config = super(Conv1DTranspose, self).get_config()
    config['output_padding'] = self.output_padding
    return config


@keras_export('keras.layers.Conv2DTranspose',
              'keras.layers.Convolution2DTranspose')
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
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    output_padding: An integer or tuple/list of 2 integers,
      specifying the amount of padding along the height and width
      of the output tensor.
      Can be a single integer to specify the same value for all
      spatial dimensions.
      The amount of output padding along a given dimension must be
      lower than the stride along that same dimension.
      If set to `None` (default), the output shape is inferred.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
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
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix (
      see `keras.initializers`).
    bias_initializer: Initializer for the bias vector (
      see `keras.initializers`).
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation") (see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).

  Input shape:
    4D tensor with shape:
    `(batch_size, channels, rows, cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(batch_size, rows, cols, channels)` if data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `(batch_size, filters, new_rows, new_cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(batch_size, new_rows, new_cols, filters)` if data_format='channels_last'.
    `rows` and `cols` values might have changed due to padding.
    If `output_padding` is specified:
    ```
    new_rows = ((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] +
    output_padding[0])
    new_cols = ((cols - 1) * strides[1] + kernel_size[1] - 2 * padding[1] +
    output_padding[1])
    ```

  Returns:
    A tensor of rank 4 representing
    `activation(conv2dtranspose(inputs, kernel) + bias)`.

  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.

  References:
    - [A guide to convolution arithmetic for deep
      learning](https://arxiv.org/abs/1603.07285v1)
    - [Deconvolutional
      Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               output_padding=None,
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
    super(Conv2DTranspose, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

    self.output_padding = output_padding
    if self.output_padding is not None:
      self.output_padding = conv_utils.normalize_tuple(
          self.output_padding, 2, 'output_padding')
      for stride, out_pad in zip(self.strides, self.output_padding):
        if out_pad >= stride:
          raise ValueError('Stride ' + str(self.strides) + ' must be '
                           'greater than output padding ' +
                           str(self.output_padding))

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if len(input_shape) != 4:
      raise ValueError('Inputs should have rank 4. Received input shape: ' +
                       str(input_shape))
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
    kernel_shape = self.kernel_size + (self.filters, input_dim)

    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    inputs_shape = array_ops.shape(inputs)
    batch_size = inputs_shape[0]
    if self.data_format == 'channels_first':
      h_axis, w_axis = 2, 3
    else:
      h_axis, w_axis = 1, 2

    # Use the constant height and weight when possible.
    # TODO(scottzhu): Extract this into a utility function that can be applied
    # to all convolutional layers, which currently lost the static shape
    # information due to tf.shape().
    height, width = None, None
    if inputs.shape.rank is not None:
      dims = inputs.shape.as_list()
      height = dims[h_axis]
      width = dims[w_axis]
    height = height if height is not None else inputs_shape[h_axis]
    width = width if width is not None else inputs_shape[w_axis]

    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides

    if self.output_padding is None:
      out_pad_h = out_pad_w = None
    else:
      out_pad_h, out_pad_w = self.output_padding

    # Infer the dynamic output shape:
    out_height = conv_utils.deconv_output_length(height,
                                                 kernel_h,
                                                 padding=self.padding,
                                                 output_padding=out_pad_h,
                                                 stride=stride_h,
                                                 dilation=self.dilation_rate[0])
    out_width = conv_utils.deconv_output_length(width,
                                                kernel_w,
                                                padding=self.padding,
                                                output_padding=out_pad_w,
                                                stride=stride_w,
                                                dilation=self.dilation_rate[1])
    if self.data_format == 'channels_first':
      output_shape = (batch_size, self.filters, out_height, out_width)
    else:
      output_shape = (batch_size, out_height, out_width, self.filters)

    output_shape_tensor = array_ops.stack(output_shape)
    outputs = backend.conv2d_transpose(
        inputs,
        self.kernel,
        output_shape_tensor,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format,
        dilation_rate=self.dilation_rate)

    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(inputs.shape)
      outputs.set_shape(out_shape)

    if self.use_bias:
      outputs = nn.bias_add(
          outputs,
          self.bias,
          data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = list(input_shape)
    if self.data_format == 'channels_first':
      c_axis, h_axis, w_axis = 1, 2, 3
    else:
      c_axis, h_axis, w_axis = 3, 1, 2

    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides

    if self.output_padding is None:
      out_pad_h = out_pad_w = None
    else:
      out_pad_h, out_pad_w = self.output_padding

    output_shape[c_axis] = self.filters
    output_shape[h_axis] = conv_utils.deconv_output_length(
        output_shape[h_axis],
        kernel_h,
        padding=self.padding,
        output_padding=out_pad_h,
        stride=stride_h,
        dilation=self.dilation_rate[0])
    output_shape[w_axis] = conv_utils.deconv_output_length(
        output_shape[w_axis],
        kernel_w,
        padding=self.padding,
        output_padding=out_pad_w,
        stride=stride_w,
        dilation=self.dilation_rate[1])
    return tensor_shape.TensorShape(output_shape)

  def get_config(self):
    config = super(Conv2DTranspose, self).get_config()
    config['output_padding'] = self.output_padding
    return config


@keras_export('keras.layers.Conv3DTranspose',
              'keras.layers.Convolution3DTranspose')
class Conv3DTranspose(Conv3D):
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
  e.g. `input_shape=(128, 128, 128, 3)` for a 128x128x128 volume with 3 channels
  if `data_format="channels_last"`.

  Arguments:
    filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 3 integers, specifying the
        depth, height and width of the 3D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
        specifying the strides of the convolution along the depth, height
          and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    output_padding: An integer or tuple/list of 3 integers,
      specifying the amount of padding along the depth, height, and
      width.
      Can be a single integer to specify the same value for all
      spatial dimensions.
      The amount of output padding along a given dimension must be
      lower than the stride along that same dimension.
      If set to `None` (default), the output shape is inferred.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, depth, height, width)`.
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
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix (
      see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation") (
      see `keras.regularizers`).
    kernel_constraint: Constraint function applied to the kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).

  Input shape:
    5D tensor with shape:
    `(batch_size, channels, depth, rows, cols)` if data_format='channels_first'
    or 5D tensor with shape:
    `(batch_size, depth, rows, cols, channels)` if data_format='channels_last'.

  Output shape:
    5D tensor with shape:
    `(batch_size, filters, new_depth, new_rows, new_cols)` if
      data_format='channels_first'
    or 5D tensor with shape:
    `(batch_size, new_depth, new_rows, new_cols, filters)` if
      data_format='channels_last'.
    `depth` and `rows` and `cols` values might have changed due to padding.
    If `output_padding` is specified::
    ```
    new_depth = ((depth - 1) * strides[0] + kernel_size[0] - 2 * padding[0] +
    output_padding[0])
    new_rows = ((rows - 1) * strides[1] + kernel_size[1] - 2 * padding[1] +
    output_padding[1])
    new_cols = ((cols - 1) * strides[2] + kernel_size[2] - 2 * padding[2] +
    output_padding[2])
    ```

  Returns:
    A tensor of rank 5 representing
    `activation(conv3dtranspose(inputs, kernel) + bias)`.

  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.

  References:
    - [A guide to convolution arithmetic for deep
      learning](https://arxiv.org/abs/1603.07285v1)
    - [Deconvolutional
      Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1, 1),
               padding='valid',
               output_padding=None,
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
    super(Conv3DTranspose, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

    self.output_padding = output_padding
    if self.output_padding is not None:
      self.output_padding = conv_utils.normalize_tuple(
          self.output_padding, 3, 'output_padding')
      for stride, out_pad in zip(self.strides, self.output_padding):
        if out_pad >= stride:
          raise ValueError('Stride ' + str(self.strides) + ' must be '
                           'greater than output padding ' +
                           str(self.output_padding))

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if len(input_shape) != 5:
      raise ValueError('Inputs should have rank 5, received input shape:',
                       str(input_shape))
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined, found None: ' + str(input_shape))
    input_dim = int(input_shape[channel_axis])
    kernel_shape = self.kernel_size + (self.filters, input_dim)
    self.input_spec = InputSpec(ndim=5, axes={channel_axis: input_dim})

    self.kernel = self.add_weight(
        'kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    inputs_shape = array_ops.shape(inputs)
    batch_size = inputs_shape[0]
    if self.data_format == 'channels_first':
      d_axis, h_axis, w_axis = 2, 3, 4
    else:
      d_axis, h_axis, w_axis = 1, 2, 3

    depth = inputs_shape[d_axis]
    height = inputs_shape[h_axis]
    width = inputs_shape[w_axis]

    kernel_d, kernel_h, kernel_w = self.kernel_size
    stride_d, stride_h, stride_w = self.strides

    if self.output_padding is None:
      out_pad_d = out_pad_h = out_pad_w = None
    else:
      out_pad_d, out_pad_h, out_pad_w = self.output_padding

    # Infer the dynamic output shape:
    out_depth = conv_utils.deconv_output_length(depth,
                                                kernel_d,
                                                padding=self.padding,
                                                output_padding=out_pad_d,
                                                stride=stride_d)
    out_height = conv_utils.deconv_output_length(height,
                                                 kernel_h,
                                                 padding=self.padding,
                                                 output_padding=out_pad_h,
                                                 stride=stride_h)
    out_width = conv_utils.deconv_output_length(width,
                                                kernel_w,
                                                padding=self.padding,
                                                output_padding=out_pad_w,
                                                stride=stride_w)
    if self.data_format == 'channels_first':
      output_shape = (batch_size, self.filters, out_depth, out_height,
                      out_width)
      strides = (1, 1, stride_d, stride_h, stride_w)
    else:
      output_shape = (batch_size, out_depth, out_height, out_width,
                      self.filters)
      strides = (1, stride_d, stride_h, stride_w, 1)

    output_shape_tensor = array_ops.stack(output_shape)
    outputs = nn.conv3d_transpose(
        inputs,
        self.kernel,
        output_shape_tensor,
        strides,
        data_format=conv_utils.convert_data_format(self.data_format, ndim=5),
        padding=self.padding.upper())

    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(inputs.shape)
      outputs.set_shape(out_shape)

    if self.use_bias:
      outputs = nn.bias_add(
          outputs,
          self.bias,
          data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = list(input_shape)
    if self.data_format == 'channels_first':
      c_axis, d_axis, h_axis, w_axis = 1, 2, 3, 4
    else:
      c_axis, d_axis, h_axis, w_axis = 4, 1, 2, 3

    kernel_d, kernel_h, kernel_w = self.kernel_size
    stride_d, stride_h, stride_w = self.strides

    if self.output_padding is None:
      out_pad_d = out_pad_h = out_pad_w = None
    else:
      out_pad_d, out_pad_h, out_pad_w = self.output_padding

    output_shape[c_axis] = self.filters
    output_shape[d_axis] = conv_utils.deconv_output_length(
        output_shape[d_axis],
        kernel_d,
        padding=self.padding,
        output_padding=out_pad_d,
        stride=stride_d)
    output_shape[h_axis] = conv_utils.deconv_output_length(
        output_shape[h_axis],
        kernel_h,
        padding=self.padding,
        output_padding=out_pad_h,
        stride=stride_h)
    output_shape[w_axis] = conv_utils.deconv_output_length(
        output_shape[w_axis],
        kernel_w,
        padding=self.padding,
        output_padding=out_pad_w,
        stride=stride_w)
    return tensor_shape.TensorShape(output_shape)

  def get_config(self):
    config = super(Conv3DTranspose, self).get_config()
    config.pop('dilation_rate')
    config['output_padding'] = self.output_padding
    return config


class SeparableConv(Conv):
  """Abstract base layer for separable nD convolution.

  This layer performs a depthwise convolution that acts separately on
  channels, followed by a pointwise convolution that mixes channels.
  If `use_bias` is True and a bias initializer is provided,
  it adds a bias vector to the output.
  It then optionally applies an activation function to produce the final output.

  Arguments:
    rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: A tuple or list of integers specifying the spatial
      dimensions of the filters. Can be a single integer to specify the same
      value for all spatial dimensions.
    strides: A tuple or list of integers specifying the strides
      of the convolution. Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any `stride` value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch_size, channels, ...)`.
    dilation_rate: An integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to `num_filters_in * depth_multiplier`.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias.
    depthwise_initializer: An initializer for the depthwise convolution kernel.
    pointwise_initializer: An initializer for the pointwise convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default
      initializer will be used.
    depthwise_regularizer: Optional regularizer for the depthwise
      convolution kernel.
    pointwise_regularizer: Optional regularizer for the pointwise
      convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    depthwise_constraint: Optional projection function to be applied to the
      depthwise kernel after being updated by an `Optimizer` (e.g. used for
      norm constraints or value constraints for layer weights). The function
      must take as input the unprojected variable and must return the
      projected variable (which must have the same shape). Constraints are
      not safe to use when doing asynchronous distributed training.
    pointwise_constraint: Optional projection function to be applied to the
      pointwise kernel after being updated by an `Optimizer`.
    bias_constraint: Optional projection function to be applied to the
      bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` the weights of this layer will be marked as
      trainable (and listed in `layer.trainable_weights`).
    name: A string, the name of the layer.
  """

  def __init__(self,
               rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
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
               trainable=True,
               name=None,
               **kwargs):
    super(SeparableConv, self).__init__(
        rank=rank,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activations.get(activation),
        use_bias=use_bias,
        bias_initializer=initializers.get(bias_initializer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name,
        **kwargs)
    self.depth_multiplier = depth_multiplier
    self.depthwise_initializer = initializers.get(depthwise_initializer)
    self.pointwise_initializer = initializers.get(pointwise_initializer)
    self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
    self.pointwise_regularizer = regularizers.get(pointwise_regularizer)
    self.depthwise_constraint = constraints.get(depthwise_constraint)
    self.pointwise_constraint = constraints.get(pointwise_constraint)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    self.input_spec = InputSpec(ndim=self.rank + 2,
                                axes={channel_axis: input_dim})
    depthwise_kernel_shape = self.kernel_size + (input_dim,
                                                 self.depth_multiplier)
    pointwise_kernel_shape = (
        1,) * self.rank + (self.depth_multiplier * input_dim, self.filters)

    self.depthwise_kernel = self.add_weight(
        name='depthwise_kernel',
        shape=depthwise_kernel_shape,
        initializer=self.depthwise_initializer,
        regularizer=self.depthwise_regularizer,
        constraint=self.depthwise_constraint,
        trainable=True,
        dtype=self.dtype)
    self.pointwise_kernel = self.add_weight(
        name='pointwise_kernel',
        shape=pointwise_kernel_shape,
        initializer=self.pointwise_initializer,
        regularizer=self.pointwise_regularizer,
        constraint=self.pointwise_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    raise NotImplementedError

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
        'depth_multiplier':
            self.depth_multiplier,
        'dilation_rate':
            self.dilation_rate,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'depthwise_initializer':
            initializers.serialize(self.depthwise_initializer),
        'pointwise_initializer':
            initializers.serialize(self.pointwise_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'depthwise_regularizer':
            regularizers.serialize(self.depthwise_regularizer),
        'pointwise_regularizer':
            regularizers.serialize(self.pointwise_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'depthwise_constraint':
            constraints.serialize(self.depthwise_constraint),
        'pointwise_constraint':
            constraints.serialize(self.pointwise_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint)
    }
    base_config = super(SeparableConv, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.SeparableConv1D',
              'keras.layers.SeparableConvolution1D')
class SeparableConv1D(SeparableConv):
  """Depthwise separable 1D convolution.

  This layer performs a depthwise convolution that acts separately on
  channels, followed by a pointwise convolution that mixes channels.
  If `use_bias` is True and a bias initializer is provided,
  it adds a bias vector to the output.
  It then optionally applies an activation function to produce the final output.

  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: A single integer specifying the spatial
      dimensions of the filters.
    strides: A single integer specifying the strides
      of the convolution.
      Specifying any `stride` value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"`, `"same"`, or `"causal"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch_size, channels, length)`.
    dilation_rate: A single integer, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to `num_filters_in * depth_multiplier`.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias.
    depthwise_initializer: An initializer for the depthwise convolution kernel (
      see `keras.initializers`).
    pointwise_initializer: An initializer for the pointwise convolution kernel (
      see `keras.initializers`).
    bias_initializer: An initializer for the bias vector. If None, the default
      initializer will be used (see `keras.initializers`).
    depthwise_regularizer: Optional regularizer for the depthwise
      convolution kernel (see `keras.regularizers`).
    pointwise_regularizer: Optional regularizer for the pointwise
      convolution kernel (see `keras.regularizers`).
    bias_regularizer: Optional regularizer for the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Optional regularizer function for the output (
      see `keras.regularizers`).
    depthwise_constraint: Optional projection function to be applied to the
      depthwise kernel after being updated by an `Optimizer` (e.g. used for
      norm constraints or value constraints for layer weights). The function
      must take as input the unprojected variable and must return the
      projected variable (which must have the same shape). Constraints are
      not safe to use when doing asynchronous distributed training (
      see `keras.constraints`).
    pointwise_constraint: Optional projection function to be applied to the
      pointwise kernel after being updated by an `Optimizer` (
      see `keras.constraints`).
    bias_constraint: Optional projection function to be applied to the
      bias after being updated by an `Optimizer` (
      see `keras.constraints`).
    trainable: Boolean, if `True` the weights of this layer will be marked as
      trainable (and listed in `layer.trainable_weights`).
    name: A string, the name of the layer.

  Input shape:
    3D tensor with shape:
    `(batch_size, channels, steps)` if data_format='channels_first'
    or 5D tensor with shape:
    `(batch_size, steps, channels)` if data_format='channels_last'.

  Output shape:
    3D tensor with shape:
    `(batch_size, filters, new_steps)` if data_format='channels_first'
    or 3D tensor with shape:
    `(batch_size,  new_steps, filters)` if data_format='channels_last'.
    `new_steps` value might have changed due to padding or strides.

  Returns:
    A tensor of rank 3 representing
    `activation(separableconv1d(inputs, kernel) + bias)`.

  Raises:
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
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
    super(SeparableConv1D, self).__init__(
        rank=1,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        depth_multiplier=depth_multiplier,
        activation=activations.get(activation),
        use_bias=use_bias,
        depthwise_initializer=initializers.get(depthwise_initializer),
        pointwise_initializer=initializers.get(pointwise_initializer),
        bias_initializer=initializers.get(bias_initializer),
        depthwise_regularizer=regularizers.get(depthwise_regularizer),
        pointwise_regularizer=regularizers.get(pointwise_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        depthwise_constraint=constraints.get(depthwise_constraint),
        pointwise_constraint=constraints.get(pointwise_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

  def call(self, inputs):
    if self.padding == 'causal':
      inputs = array_ops.pad(inputs, self._compute_causal_padding())
    if self.data_format == 'channels_last':
      strides = (1,) + self.strides * 2 + (1,)
      spatial_start_dim = 1
    else:
      strides = (1, 1) + self.strides * 2
      spatial_start_dim = 2

    # Explicitly broadcast inputs and kernels to 4D.
    # TODO(fchollet): refactor when a native separable_conv1d op is available.
    inputs = array_ops.expand_dims(inputs, spatial_start_dim)
    depthwise_kernel = array_ops.expand_dims(self.depthwise_kernel, 0)
    pointwise_kernel = array_ops.expand_dims(self.pointwise_kernel, 0)
    dilation_rate = (1,) + self.dilation_rate

    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    outputs = nn.separable_conv2d(
        inputs,
        depthwise_kernel,
        pointwise_kernel,
        strides=strides,
        padding=op_padding.upper(),
        rate=dilation_rate,
        data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

    if self.use_bias:
      outputs = nn.bias_add(
          outputs,
          self.bias,
          data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

    outputs = array_ops.squeeze(outputs, [spatial_start_dim])

    if self.activation is not None:
      return self.activation(outputs)
    return outputs


@keras_export('keras.layers.SeparableConv2D',
              'keras.layers.SeparableConvolution2D')
class SeparableConv2D(SeparableConv):
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
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    dilation_rate: An integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    depth_multiplier: The number of depthwise convolution output channels
      for each input channel.
      The total number of depthwise convolution output
      channels will be equal to `filters_in * depth_multiplier`.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    depthwise_initializer: Initializer for the depthwise kernel matrix (
      see `keras.initializers`).
    pointwise_initializer: Initializer for the pointwise kernel matrix (
      see `keras.initializers`).
    bias_initializer: Initializer for the bias vector (
      see `keras.initializers`).
    depthwise_regularizer: Regularizer function applied to
      the depthwise kernel matrix (see `keras.regularizers`).
    pointwise_regularizer: Regularizer function applied to
      the pointwise kernel matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation") (
      see `keras.regularizers`).
    depthwise_constraint: Constraint function applied to
      the depthwise kernel matrix (
      see `keras.constraints`).
    pointwise_constraint: Constraint function applied to
      the pointwise kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).

  Input shape:
    4D tensor with shape:
    `(batch_size, channels, rows, cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(batch_size, rows, cols, channels)` if data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `(batch_size, filters, new_rows, new_cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(batch_size, new_rows, new_cols, filters)` if data_format='channels_last'.
    `rows` and `cols` values might have changed due to padding.

  Returns:
    A tensor of rank 4 representing
    `activation(separableconv2d(inputs, kernel) + bias)`.

  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
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
        rank=2,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        depth_multiplier=depth_multiplier,
        activation=activations.get(activation),
        use_bias=use_bias,
        depthwise_initializer=initializers.get(depthwise_initializer),
        pointwise_initializer=initializers.get(pointwise_initializer),
        bias_initializer=initializers.get(bias_initializer),
        depthwise_regularizer=regularizers.get(depthwise_regularizer),
        pointwise_regularizer=regularizers.get(pointwise_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        depthwise_constraint=constraints.get(depthwise_constraint),
        pointwise_constraint=constraints.get(pointwise_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

  def call(self, inputs):
    # Apply the actual ops.
    if self.data_format == 'channels_last':
      strides = (1,) + self.strides + (1,)
    else:
      strides = (1, 1) + self.strides
    outputs = nn.separable_conv2d(
        inputs,
        self.depthwise_kernel,
        self.pointwise_kernel,
        strides=strides,
        padding=self.padding.upper(),
        rate=self.dilation_rate,
        data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

    if self.use_bias:
      outputs = nn.bias_add(
          outputs,
          self.bias,
          data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

    if self.activation is not None:
      return self.activation(outputs)
    return outputs


@keras_export('keras.layers.DepthwiseConv2D')
class DepthwiseConv2D(Conv2D):
  """Depthwise separable 2D convolution.

  Depthwise Separable convolutions consists in performing
  just the first step in a depthwise spatial convolution
  (which acts on each input channel separately).
  The `depth_multiplier` argument controls how many
  output channels are generated per input channel in the depthwise step.

  Arguments:
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: one of `'valid'` or `'same'` (case-insensitive).
    depth_multiplier: The number of depthwise convolution output channels
      for each input channel.
      The total number of depthwise convolution output
      channels will be equal to `filters_in * depth_multiplier`.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be 'channels_last'.
    dilation_rate: An integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied (
      see `keras.activations`).
    use_bias: Boolean, whether the layer uses a bias vector.
    depthwise_initializer: Initializer for the depthwise kernel matrix (
      see `keras.initializers`).
    bias_initializer: Initializer for the bias vector (
      see `keras.initializers`).
    depthwise_regularizer: Regularizer function applied to
      the depthwise kernel matrix (see `keras.regularizers`).
    bias_regularizer: Regularizer function applied to the bias vector (
      see `keras.regularizers`).
    activity_regularizer: Regularizer function applied to
      the output of the layer (its 'activation') (
      see `keras.regularizers`).
    depthwise_constraint: Constraint function applied to
      the depthwise kernel matrix (
      see `keras.constraints`).
    bias_constraint: Constraint function applied to the bias vector (
      see `keras.constraints`).

  Input shape:
    4D tensor with shape:
    `[batch_size, channels, rows, cols]` if data_format='channels_first'
    or 4D tensor with shape:
    `[batch_size, rows, cols, channels]` if data_format='channels_last'.

  Output shape:
    4D tensor with shape:
    `[batch_size, filters, new_rows, new_cols]` if data_format='channels_first'
    or 4D tensor with shape:
    `[batch_size, new_rows, new_cols, filters]` if data_format='channels_last'.
    `rows` and `cols` values might have changed due to padding.

  Returns:
    A tensor of rank 4 representing
    `activation(depthwiseconv2d(inputs, kernel) + bias)`.

  Raises:
    ValueError: if `padding` is "causal".
    ValueError: when both `strides` > 1 and `dilation_rate` > 1.
  """

  def __init__(self,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               depth_multiplier=1,
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(DepthwiseConv2D, self).__init__(
        filters=None,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        bias_constraint=bias_constraint,
        **kwargs)
    self.depth_multiplier = depth_multiplier
    self.depthwise_initializer = initializers.get(depthwise_initializer)
    self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
    self.depthwise_constraint = constraints.get(depthwise_constraint)
    self.bias_initializer = initializers.get(bias_initializer)

  def build(self, input_shape):
    if len(input_shape) < 4:
      raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                       'Received input shape:', str(input_shape))
    input_shape = tensor_shape.TensorShape(input_shape)
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs to '
                       '`DepthwiseConv2D` '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    depthwise_kernel_shape = (self.kernel_size[0],
                              self.kernel_size[1],
                              input_dim,
                              self.depth_multiplier)

    self.depthwise_kernel = self.add_weight(
        shape=depthwise_kernel_shape,
        initializer=self.depthwise_initializer,
        name='depthwise_kernel',
        regularizer=self.depthwise_regularizer,
        constraint=self.depthwise_constraint)

    if self.use_bias:
      self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
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
    outputs = backend.depthwise_conv2d(
        inputs,
        self.depthwise_kernel,
        strides=self.strides,
        padding=self.padding,
        dilation_rate=self.dilation_rate,
        data_format=self.data_format)

    if self.use_bias:
      outputs = backend.bias_add(
          outputs,
          self.bias,
          data_format=self.data_format)

    if self.activation is not None:
      return self.activation(outputs)

    return outputs

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if self.data_format == 'channels_first':
      rows = input_shape[2]
      cols = input_shape[3]
      out_filters = input_shape[1] * self.depth_multiplier
    elif self.data_format == 'channels_last':
      rows = input_shape[1]
      cols = input_shape[2]
      out_filters = input_shape[3] * self.depth_multiplier

    rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                         self.padding,
                                         self.strides[0],
                                         self.dilation_rate[0])
    cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                         self.padding,
                                         self.strides[1],
                                         self.dilation_rate[1])
    if self.data_format == 'channels_first':
      return (input_shape[0], out_filters, rows, cols)
    elif self.data_format == 'channels_last':
      return (input_shape[0], rows, cols, out_filters)

  def get_config(self):
    config = super(DepthwiseConv2D, self).get_config()
    config.pop('filters')
    config.pop('kernel_initializer')
    config.pop('kernel_regularizer')
    config.pop('kernel_constraint')
    config['depth_multiplier'] = self.depth_multiplier
    config['depthwise_initializer'] = initializers.serialize(
        self.depthwise_initializer)
    config['depthwise_regularizer'] = regularizers.serialize(
        self.depthwise_regularizer)
    config['depthwise_constraint'] = constraints.serialize(
        self.depthwise_constraint)
    return config


@keras_export('keras.layers.UpSampling1D')
class UpSampling1D(Layer):
  """Upsampling layer for 1D inputs.

  Repeats each temporal step `size` times along the time axis.

  Examples:

  >>> input_shape = (2, 2, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[ 0  1  2]
    [ 3  4  5]]
   [[ 6  7  8]
    [ 9 10 11]]]
  >>> y = tf.keras.layers.UpSampling1D(size=2)(x)
  >>> print(y)
  tf.Tensor(
    [[[ 0  1  2]
      [ 0  1  2]
      [ 3  4  5]
      [ 3  4  5]]
     [[ 6  7  8]
      [ 6  7  8]
      [ 9 10 11]
      [ 9 10 11]]], shape=(2, 4, 3), dtype=int64)

  Arguments:
    size: Integer. Upsampling factor.

  Input shape:
    3D tensor with shape: `(batch_size, steps, features)`.

  Output shape:
    3D tensor with shape: `(batch_size, upsampled_steps, features)`.
  """

  def __init__(self, size=2, **kwargs):
    super(UpSampling1D, self).__init__(**kwargs)
    self.size = int(size)
    self.input_spec = InputSpec(ndim=3)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    size = self.size * input_shape[1] if input_shape[1] is not None else None
    return tensor_shape.TensorShape([input_shape[0], size, input_shape[2]])

  def call(self, inputs):
    output = backend.repeat_elements(inputs, self.size, axis=1)
    return output

  def get_config(self):
    config = {'size': self.size}
    base_config = super(UpSampling1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.UpSampling2D')
class UpSampling2D(Layer):
  """Upsampling layer for 2D inputs.

  Repeats the rows and columns of the data
  by `size[0]` and `size[1]` respectively.

  Examples:

  >>> input_shape = (2, 2, 1, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[[ 0  1  2]]
    [[ 3  4  5]]]
   [[[ 6  7  8]]
    [[ 9 10 11]]]]
  >>> y = tf.keras.layers.UpSampling2D(size=(1, 2))(x)
  >>> print(y)
  tf.Tensor(
    [[[[ 0  1  2]
       [ 0  1  2]]
      [[ 3  4  5]
       [ 3  4  5]]]
     [[[ 6  7  8]
       [ 6  7  8]]
      [[ 9 10 11]
       [ 9 10 11]]]], shape=(2, 2, 2, 3), dtype=int64)

  Arguments:
    size: Int, or tuple of 2 integers.
      The upsampling factors for rows and columns.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    interpolation: A string, one of `nearest` or `bilinear`.

  Input shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, rows, cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, rows, cols)`

  Output shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, upsampled_rows, upsampled_cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, upsampled_rows, upsampled_cols)`
  """

  def __init__(self,
               size=(2, 2),
               data_format=None,
               interpolation='nearest',
               **kwargs):
    super(UpSampling2D, self).__init__(**kwargs)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.size = conv_utils.normalize_tuple(size, 2, 'size')
    if interpolation not in {'nearest', 'bilinear'}:
      raise ValueError('`interpolation` argument should be one of `"nearest"` '
                       'or `"bilinear"`.')
    self.interpolation = interpolation
    self.input_spec = InputSpec(ndim=4)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      height = self.size[0] * input_shape[
          2] if input_shape[2] is not None else None
      width = self.size[1] * input_shape[
          3] if input_shape[3] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], height, width])
    else:
      height = self.size[0] * input_shape[
          1] if input_shape[1] is not None else None
      width = self.size[1] * input_shape[
          2] if input_shape[2] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], height, width, input_shape[3]])

  def call(self, inputs):
    return backend.resize_images(
        inputs, self.size[0], self.size[1], self.data_format,
        interpolation=self.interpolation)

  def get_config(self):
    config = {
        'size': self.size,
        'data_format': self.data_format,
        'interpolation': self.interpolation
    }
    base_config = super(UpSampling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.UpSampling3D')
class UpSampling3D(Layer):
  """Upsampling layer for 3D inputs.

  Repeats the 1st, 2nd and 3rd dimensions
  of the data by `size[0]`, `size[1]` and `size[2]` respectively.

  Examples:

  >>> input_shape = (2, 1, 2, 1, 3)
  >>> x = tf.constant(1, shape=input_shape)
  >>> y = tf.keras.layers.UpSampling3D(size=2)(x)
  >>> print(y.shape)
  (2, 2, 4, 2, 3)

  Arguments:
    size: Int, or tuple of 3 integers.
      The upsampling factors for dim1, dim2 and dim3.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      while `channels_first` corresponds to inputs with shape
      `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

  Input shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, dim1, dim2, dim3, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, dim1, dim2, dim3)`

  Output shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)`
  """

  def __init__(self, size=(2, 2, 2), data_format=None, **kwargs):
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.size = conv_utils.normalize_tuple(size, 3, 'size')
    self.input_spec = InputSpec(ndim=5)
    super(UpSampling3D, self).__init__(**kwargs)

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      dim1 = self.size[0] * input_shape[
          2] if input_shape[2] is not None else None
      dim2 = self.size[1] * input_shape[
          3] if input_shape[3] is not None else None
      dim3 = self.size[2] * input_shape[
          4] if input_shape[4] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], dim1, dim2, dim3])
    else:
      dim1 = self.size[0] * input_shape[
          1] if input_shape[1] is not None else None
      dim2 = self.size[1] * input_shape[
          2] if input_shape[2] is not None else None
      dim3 = self.size[2] * input_shape[
          3] if input_shape[3] is not None else None
      return tensor_shape.TensorShape(
          [input_shape[0], dim1, dim2, dim3, input_shape[4]])

  def call(self, inputs):
    return backend.resize_volumes(
        inputs, self.size[0], self.size[1], self.size[2], self.data_format)

  def get_config(self):
    config = {'size': self.size, 'data_format': self.data_format}
    base_config = super(UpSampling3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.ZeroPadding1D')
class ZeroPadding1D(Layer):
  """Zero-padding layer for 1D input (e.g. temporal sequence).

  Examples:

  >>> input_shape = (2, 2, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[ 0  1  2]
    [ 3  4  5]]
   [[ 6  7  8]
    [ 9 10 11]]]
  >>> y = tf.keras.layers.ZeroPadding1D(padding=2)(x)
  >>> print(y)
  tf.Tensor(
    [[[ 0  0  0]
      [ 0  0  0]
      [ 0  1  2]
      [ 3  4  5]
      [ 0  0  0]
      [ 0  0  0]]
     [[ 0  0  0]
      [ 0  0  0]
      [ 6  7  8]
      [ 9 10 11]
      [ 0  0  0]
      [ 0  0  0]]], shape=(2, 6, 3), dtype=int64)

  Arguments:
      padding: Int, or tuple of int (length 2), or dictionary.
          - If int:
          How many zeros to add at the beginning and end of
          the padding dimension (axis 1).
          - If tuple of int (length 2):
          How many zeros to add at the beginning and at the end of
          the padding dimension (`(left_pad, right_pad)`).

  Input shape:
      3D tensor with shape `(batch_size, axis_to_pad, features)`

  Output shape:
      3D tensor with shape `(batch_size, padded_axis, features)`
  """

  def __init__(self, padding=1, **kwargs):
    super(ZeroPadding1D, self).__init__(**kwargs)
    self.padding = conv_utils.normalize_tuple(padding, 2, 'padding')
    self.input_spec = InputSpec(ndim=3)

  def compute_output_shape(self, input_shape):
    if input_shape[1] is not None:
      length = input_shape[1] + self.padding[0] + self.padding[1]
    else:
      length = None
    return tensor_shape.TensorShape([input_shape[0], length, input_shape[2]])

  def call(self, inputs):
    return backend.temporal_padding(inputs, padding=self.padding)

  def get_config(self):
    config = {'padding': self.padding}
    base_config = super(ZeroPadding1D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.ZeroPadding2D')
class ZeroPadding2D(Layer):
  """Zero-padding layer for 2D input (e.g. picture).

  This layer can add rows and columns of zeros
  at the top, bottom, left and right side of an image tensor.

  Examples:

  >>> input_shape = (1, 1, 2, 2)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[[0 1]
     [2 3]]]]
  >>> y = tf.keras.layers.ZeroPadding2D(padding=1)(x)
  >>> print(y)
  tf.Tensor(
    [[[[0 0]
       [0 0]
       [0 0]
       [0 0]]
      [[0 0]
       [0 1]
       [2 3]
       [0 0]]
      [[0 0]
       [0 0]
       [0 0]
       [0 0]]]], shape=(1, 3, 4, 2), dtype=int64)

  Arguments:
    padding: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
      - If int: the same symmetric padding
        is applied to height and width.
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
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

  Input shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, rows, cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, rows, cols)`

  Output shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, padded_rows, padded_cols, channels)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, channels, padded_rows, padded_cols)`
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

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      if input_shape[2] is not None:
        rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
      else:
        rows = None
      if input_shape[3] is not None:
        cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
      else:
        cols = None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], rows, cols])
    elif self.data_format == 'channels_last':
      if input_shape[1] is not None:
        rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
      else:
        rows = None
      if input_shape[2] is not None:
        cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
      else:
        cols = None
      return tensor_shape.TensorShape(
          [input_shape[0], rows, cols, input_shape[3]])

  def call(self, inputs):
    return backend.spatial_2d_padding(
        inputs, padding=self.padding, data_format=self.data_format)

  def get_config(self):
    config = {'padding': self.padding, 'data_format': self.data_format}
    base_config = super(ZeroPadding2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.ZeroPadding3D')
class ZeroPadding3D(Layer):
  """Zero-padding layer for 3D data (spatial or spatio-temporal).

  Examples:

  >>> input_shape = (1, 1, 2, 2, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> y = tf.keras.layers.ZeroPadding3D(padding=2)(x)
  >>> print(y.shape)
  (1, 5, 6, 6, 3)

  Arguments:
    padding: Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
      - If int: the same symmetric padding
        is applied to height and width.
      - If tuple of 3 ints:
        interpreted as two different
        symmetric padding values for height and width:
        `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
      - If tuple of 3 tuples of 2 ints:
        interpreted as
        `((left_dim1_pad, right_dim1_pad), (left_dim2_pad,
          right_dim2_pad), (left_dim3_pad, right_dim3_pad))`
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      while `channels_first` corresponds to inputs with shape
      `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

  Input shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad,
          depth)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, depth, first_axis_to_pad, second_axis_to_pad,
          third_axis_to_pad)`

  Output shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
        `(batch_size, first_padded_axis, second_padded_axis, third_axis_to_pad,
          depth)`
    - If `data_format` is `"channels_first"`:
        `(batch_size, depth, first_padded_axis, second_padded_axis,
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

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      if input_shape[2] is not None:
        dim1 = input_shape[2] + 2 * self.padding[0][0]
      else:
        dim1 = None
      if input_shape[3] is not None:
        dim2 = input_shape[3] + 2 * self.padding[1][0]
      else:
        dim2 = None
      if input_shape[4] is not None:
        dim3 = input_shape[4] + 2 * self.padding[2][0]
      else:
        dim3 = None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], dim1, dim2, dim3])
    elif self.data_format == 'channels_last':
      if input_shape[1] is not None:
        dim1 = input_shape[1] + 2 * self.padding[0][1]
      else:
        dim1 = None
      if input_shape[2] is not None:
        dim2 = input_shape[2] + 2 * self.padding[1][1]
      else:
        dim2 = None
      if input_shape[3] is not None:
        dim3 = input_shape[3] + 2 * self.padding[2][1]
      else:
        dim3 = None
      return tensor_shape.TensorShape(
          [input_shape[0], dim1, dim2, dim3, input_shape[4]])

  def call(self, inputs):
    return backend.spatial_3d_padding(
        inputs, padding=self.padding, data_format=self.data_format)

  def get_config(self):
    config = {'padding': self.padding, 'data_format': self.data_format}
    base_config = super(ZeroPadding3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.layers.Cropping1D')
class Cropping1D(Layer):
  """Cropping layer for 1D input (e.g. temporal sequence).

  It crops along the time dimension (axis 1).

  Examples:

  >>> input_shape = (2, 3, 2)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> print(x)
  [[[ 0  1]
    [ 2  3]
    [ 4  5]]
   [[ 6  7]
    [ 8  9]
    [10 11]]]
  >>> y = tf.keras.layers.Cropping1D(cropping=1)(x)
  >>> print(y)
  tf.Tensor(
    [[[2 3]]
     [[8 9]]], shape=(2, 1, 2), dtype=int64)

  Arguments:
    cropping: Int or tuple of int (length 2)
      How many units should be trimmed off at the beginning and end of
      the cropping dimension (axis 1).
      If a single int is provided, the same value will be used for both.

  Input shape:
    3D tensor with shape `(batch_size, axis_to_crop, features)`

  Output shape:
    3D tensor with shape `(batch_size, cropped_axis, features)`
  """

  def __init__(self, cropping=(1, 1), **kwargs):
    super(Cropping1D, self).__init__(**kwargs)
    self.cropping = conv_utils.normalize_tuple(cropping, 2, 'cropping')
    self.input_spec = InputSpec(ndim=3)

  def compute_output_shape(self, input_shape):
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


@keras_export('keras.layers.Cropping2D')
class Cropping2D(Layer):
  """Cropping layer for 2D input (e.g. picture).

  It crops along spatial dimensions, i.e. height and width.

  Examples:

  >>> input_shape = (2, 28, 28, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> y = tf.keras.layers.Cropping2D(cropping=((2, 2), (4, 4)))(x)
  >>> print(y.shape)
  (2, 24, 20, 3)

  Arguments:
    cropping: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
      - If int: the same symmetric cropping
        is applied to height and width.
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
      `(batch_size, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch_size, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

  Input shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
      `(batch_size, rows, cols, channels)`
    - If `data_format` is `"channels_first"`:
      `(batch_size, channels, rows, cols)`

  Output shape:
    4D tensor with shape:
    - If `data_format` is `"channels_last"`:
      `(batch_size, cropped_rows, cropped_cols, channels)`
    - If `data_format` is `"channels_first"`:
      `(batch_size, channels, cropped_rows, cropped_cols)`
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

  def compute_output_shape(self, input_shape):
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


@keras_export('keras.layers.Cropping3D')
class Cropping3D(Layer):
  """Cropping layer for 3D data (e.g. spatial or spatio-temporal).

    Examples:

  >>> input_shape = (2, 28, 28, 10, 3)
  >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
  >>> y = tf.keras.layers.Cropping3D(cropping=(2, 4, 2))(x)
  >>> print(y.shape)
  (2, 24, 20, 6, 3)

  Arguments:
    cropping: Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
      - If int: the same symmetric cropping
        is applied to depth, height, and width.
      - If tuple of 3 ints: interpreted as two different
        symmetric cropping values for depth, height, and width:
        `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`.
      - If tuple of 3 tuples of 2 ints: interpreted as
        `((left_dim1_crop, right_dim1_crop), (left_dim2_crop,
          right_dim2_crop), (left_dim3_crop, right_dim3_crop))`
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      while `channels_first` corresponds to inputs with shape
      `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".

  Input shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
      `(batch_size, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop,
        depth)`
    - If `data_format` is `"channels_first"`:
      `(batch_size, depth, first_axis_to_crop, second_axis_to_crop,
        third_axis_to_crop)`

  Output shape:
    5D tensor with shape:
    - If `data_format` is `"channels_last"`:
      `(batch_size, first_cropped_axis, second_cropped_axis, third_cropped_axis,
        depth)`
    - If `data_format` is `"channels_first"`:
      `(batch_size, depth, first_cropped_axis, second_cropped_axis,
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

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    # pylint: disable=invalid-unary-operand-type
    if self.data_format == 'channels_first':
      if input_shape[2] is not None:
        dim1 = input_shape[2] - self.cropping[0][0] - self.cropping[0][1]
      else:
        dim1 = None
      if input_shape[3] is not None:
        dim2 = input_shape[3] - self.cropping[1][0] - self.cropping[1][1]
      else:
        dim2 = None
      if input_shape[4] is not None:
        dim3 = input_shape[4] - self.cropping[2][0] - self.cropping[2][1]
      else:
        dim3 = None
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], dim1, dim2, dim3])
    elif self.data_format == 'channels_last':
      if input_shape[1] is not None:
        dim1 = input_shape[1] - self.cropping[0][0] - self.cropping[0][1]
      else:
        dim1 = None
      if input_shape[2] is not None:
        dim2 = input_shape[2] - self.cropping[1][0] - self.cropping[1][1]
      else:
        dim2 = None
      if input_shape[3] is not None:
        dim3 = input_shape[3] - self.cropping[2][0] - self.cropping[2][1]
      else:
        dim3 = None
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
        return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1], self.
                      cropping[1][0]:, self.cropping[2][0]:-self.cropping[2][1]]
      elif self.cropping[2][1] == 0:
        return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1], self.
                      cropping[1][0]:-self.cropping[1][1], self.cropping[2][0]:]
      return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
                    self.cropping[1][0]:-self.cropping[1][1], self.cropping[2][
                        0]:-self.cropping[2][1]]
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
            0]:-self.cropping[1][1], self.cropping[2][0]:
                      -self.cropping[2][1], :]
      elif self.cropping[1][1] == 0:
        return inputs[:, self.cropping[0][
            0]:-self.cropping[0][1], self.cropping[1][0]:, self.cropping[2][0]:
                      -self.cropping[2][1], :]
      elif self.cropping[2][1] == 0:
        return inputs[:, self.cropping[0][0]:-self.cropping[0][1],
                      self.cropping[1][0]:-self.cropping[1][1], self.cropping[
                          2][0]:, :]
      return inputs[:, self.cropping[0][0]:-self.cropping[0][1], self.cropping[
          1][0]:-self.cropping[1][1], self.cropping[2][0]:  # pylint: disable=invalid-unary-operand-type
                    -self.cropping[2][1], :]  # pylint: disable=invalid-unary-operand-type
    # pylint: enable=invalid-unary-operand-type

  def get_config(self):
    config = {'cropping': self.cropping, 'data_format': self.data_format}
    base_config = super(Cropping3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# Aliases

Convolution1D = Conv1D
Convolution2D = Conv2D
Convolution3D = Conv3D
SeparableConvolution1D = SeparableConv1D
SeparableConvolution2D = SeparableConv2D
Convolution2DTranspose = Conv2DTranspose
Convolution3DTranspose = Conv3DTranspose
Deconvolution2D = Deconv2D = Conv2DTranspose
Deconvolution3D = Deconv3D = Conv3DTranspose
