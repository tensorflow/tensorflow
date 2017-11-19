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
# =============================================================================

# pylint: disable=unused-import,g-bad-import-order
"""Contains the convolutional layer classes and their functional aliases.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops


class _Conv(base.Layer):
  """Abstract nD convolution layer (private, used as implementation base).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

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
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    dilation_rate: An integer or tuple/list of n integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, no bias will
      be applied.
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
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
  """

  def __init__(self, rank,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(_Conv, self).__init__(trainable=trainable, name=name,
                                activity_regularizer=activity_regularizer,
                                **kwargs)
    self.rank = rank
    self.filters = filters
    self.kernel_size = utils.normalize_tuple(kernel_size, rank, 'kernel_size')
    self.strides = utils.normalize_tuple(strides, rank, 'strides')
    self.padding = utils.normalize_padding(padding)
    self.data_format = utils.normalize_data_format(data_format)
    self.dilation_rate = utils.normalize_tuple(
        dilation_rate, rank, 'dilation_rate')
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint
    self.input_spec = base.InputSpec(ndim=self.rank + 2)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis].value
    kernel_shape = self.kernel_size + (input_dim, self.filters)

    self.kernel = self.add_variable(name='kernel',
                                    shape=kernel_shape,
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint,
                                    trainable=True,
                                    dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_variable(name='bias',
                                    shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    trainable=True,
                                    dtype=self.dtype)
    else:
      self.bias = None
    self.input_spec = base.InputSpec(ndim=self.rank + 2,
                                     axes={channel_axis: input_dim})
    self._convolution_op = nn_ops.Convolution(
        input_shape,
        filter_shape=self.kernel.get_shape(),
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=self.padding.upper(),
        data_format=utils.convert_data_format(self.data_format,
                                              self.rank + 2))
    self.built = True

  def call(self, inputs):
    outputs = self._convolution_op(inputs, self.kernel)

    if self.use_bias:
      if self.data_format == 'channels_first':
        if self.rank == 1:
          # nn.bias_add does not accept a 1D input tensor.
          bias = array_ops.reshape(self.bias, (1, self.filters, 1))
          outputs += bias
        if self.rank == 2:
          outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
        if self.rank == 3:
          # As of Mar 2017, direct addition is significantly slower than
          # bias_add when computing gradients. To use bias_add, we collapse Z
          # and Y into a single dimension to obtain a 4D input tensor.
          outputs_shape = outputs.shape.as_list()
          outputs_4d = array_ops.reshape(outputs,
                                         [outputs_shape[0], outputs_shape[1],
                                          outputs_shape[2] * outputs_shape[3],
                                          outputs_shape[4]])
          outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
          outputs = array_ops.reshape(outputs_4d, outputs_shape)
      else:
        outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = utils.conv_output_length(
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
        new_dim = utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                      new_space)


class Conv1D(_Conv):
  """1D convolution layer (e.g. temporal convolution).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of a single integer, specifying the
      length of the 1D convolution window.
    strides: An integer or tuple/list of a single integer,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, length)`.
    dilation_rate: An integer or tuple/list of a single integer, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, no bias will
      be applied.
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
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
  """

  def __init__(self, filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(Convolution1D, self).__init__(
        rank=1,
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
        trainable=trainable,
        name=name, **kwargs)


def conv1d(inputs,
           filters,
           kernel_size,
           strides=1,
           padding='valid',
           data_format='channels_last',
           dilation_rate=1,
           activation=None,
           use_bias=True,
           kernel_initializer=None,
           bias_initializer=init_ops.zeros_initializer(),
           kernel_regularizer=None,
           bias_regularizer=None,
           activity_regularizer=None,
           kernel_constraint=None,
           bias_constraint=None,
           trainable=True,
           name=None,
           reuse=None):
  """Functional interface for 1D convolution layer (e.g. temporal convolution).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  Arguments:
    inputs: Tensor input.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of a single integer, specifying the
      length of the 1D convolution window.
    strides: An integer or tuple/list of a single integer,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, length)`.
    dilation_rate: An integer or tuple/list of a single integer, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, no bias will
      be applied.
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
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
  """
  layer = Conv1D(
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
      trainable=trainable,
      name=name,
      dtype=inputs.dtype.base_dtype,
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs)


class Conv2D(_Conv):
  """2D convolution layer (e.g. spatial convolution over images).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
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
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.

    dilation_rate: An integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, no bias will
      be applied.
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
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
  """

  def __init__(self, filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format='channels_last',
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
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
        trainable=trainable,
        name=name, **kwargs)


def conv2d(inputs,
           filters,
           kernel_size,
           strides=(1, 1),
           padding='valid',
           data_format='channels_last',
           dilation_rate=(1, 1),
           activation=None,
           use_bias=True,
           kernel_initializer=None,
           bias_initializer=init_ops.zeros_initializer(),
           kernel_regularizer=None,
           bias_regularizer=None,
           activity_regularizer=None,
           kernel_constraint=None,
           bias_constraint=None,
           trainable=True,
           name=None,
           reuse=None):
  """Functional interface for the 2D convolution layer.

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  Arguments:
    inputs: Tensor input.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
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
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.

    dilation_rate: An integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, no bias will
      be applied.
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
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
  """
  layer = Conv2D(
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
      trainable=trainable,
      name=name,
      dtype=inputs.dtype.base_dtype,
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs)


class Conv3D(_Conv):
  """3D convolution layer (e.g. spatial convolution over volumes).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of 3 integers, specifying the
      depth, height and width of the 3D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the convolution along the depth,
      height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, depth, height, width)`.
    dilation_rate: An integer or tuple/list of 3 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, no bias will
      be applied.
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
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
  """

  def __init__(self, filters,
               kernel_size,
               strides=(1, 1, 1),
               padding='valid',
               data_format='channels_last',
               dilation_rate=(1, 1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
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
        trainable=trainable,
        name=name, **kwargs)


def conv3d(inputs,
           filters,
           kernel_size,
           strides=(1, 1, 1),
           padding='valid',
           data_format='channels_last',
           dilation_rate=(1, 1, 1),
           activation=None,
           use_bias=True,
           kernel_initializer=None,
           bias_initializer=init_ops.zeros_initializer(),
           kernel_regularizer=None,
           bias_regularizer=None,
           activity_regularizer=None,
           kernel_constraint=None,
           bias_constraint=None,
           trainable=True,
           name=None,
           reuse=None):
  """Functional interface for the 3D convolution layer.

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.

  Arguments:
    inputs: Tensor input.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of 3 integers, specifying the
      depth, height and width of the 3D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the convolution along the depth,
      height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, depth, height, width)`.
    dilation_rate: An integer or tuple/list of 3 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, no bias will
      be applied.
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
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
  """
  layer = Conv3D(
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
      trainable=trainable,
      name=name,
      dtype=inputs.dtype.base_dtype,
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs)


class SeparableConv2D(Conv2D):
  """Depthwise separable 2D convolution.

  This layer performs a depthwise convolution that acts separately on
  channels, followed by a pointwise convolution that mixes channels.
  If `use_bias` is True and a bias initializer is provided,
  it adds a bias vector to the output.
  It then optionally applies an activation function to produce the final output.

  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: A tuple or list of 2 integers specifying the spatial
      dimensions of the filters. Can be a single integer to specify the same
      value for all spatial dimensions.
    strides: A tuple or list of 2 positive integers specifying the strides
      of the convolution. Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any `stride` value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.

    dilation_rate: An integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to `num_filters_in * depth_multiplier`.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    depthwise_initializer: An initializer for the depthwise convolution kernel.
    pointwise_initializer: An initializer for the pointwise convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, no bias will
      be applied.
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
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
  """

  def __init__(self, filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format='channels_last',
               dilation_rate=(1, 1),
               depth_multiplier=1,
               activation=None,
               use_bias=True,
               depthwise_initializer=None,
               pointwise_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
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
    super(SeparableConv2D, self).__init__(
        filters=filters,
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
        trainable=trainable,
        name=name,
        **kwargs)
    self.data_format = data_format
    self.depth_multiplier = depth_multiplier
    self.depthwise_initializer = depthwise_initializer
    self.pointwise_initializer = pointwise_initializer
    self.depthwise_regularizer = depthwise_regularizer
    self.pointwise_regularizer = pointwise_regularizer
    self.depthwise_constraint = depthwise_constraint
    self.pointwise_constraint = pointwise_constraint

  def build(self, input_shape):
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
    self.input_spec = base.InputSpec(ndim=4, axes={channel_axis: input_dim})
    depthwise_kernel_shape = (self.kernel_size[0],
                              self.kernel_size[1],
                              input_dim,
                              self.depth_multiplier)
    pointwise_kernel_shape = (1, 1,
                              self.depth_multiplier * input_dim,
                              self.filters)

    self.depthwise_kernel = self.add_variable(
        name='depthwise_kernel',
        shape=depthwise_kernel_shape,
        initializer=self.depthwise_initializer,
        regularizer=self.depthwise_regularizer,
        constraint=self.depthwise_constraint,
        trainable=True,
        dtype=self.dtype)
    self.pointwise_kernel = self.add_variable(
        name='pointwise_kernel',
        shape=pointwise_kernel_shape,
        initializer=self.pointwise_initializer,
        regularizer=self.pointwise_regularizer,
        constraint=self.pointwise_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_variable(name='bias',
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
        data_format=utils.convert_data_format(self.data_format, ndim=4))

    if self.use_bias:
      outputs = nn.bias_add(
          outputs,
          self.bias,
          data_format=utils.convert_data_format(self.data_format, ndim=4))

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

    rows = utils.conv_output_length(rows, self.kernel_size[0],
                                    self.padding, self.strides[0])
    cols = utils.conv_output_length(cols, self.kernel_size[1],
                                    self.padding, self.strides[1])
    if self.data_format == 'channels_first':
      return tensor_shape.TensorShape(
          [input_shape[0], self.filters, rows, cols])
    else:
      return tensor_shape.TensorShape(
          [input_shape[0], rows, cols, self.filters])


def separable_conv2d(inputs,
                     filters,
                     kernel_size,
                     strides=(1, 1),
                     padding='valid',
                     data_format='channels_last',
                     dilation_rate=(1, 1),
                     depth_multiplier=1,
                     activation=None,
                     use_bias=True,
                     depthwise_initializer=None,
                     pointwise_initializer=None,
                     bias_initializer=init_ops.zeros_initializer(),
                     depthwise_regularizer=None,
                     pointwise_regularizer=None,
                     bias_regularizer=None,
                     activity_regularizer=None,
                     depthwise_constraint=None,
                     pointwise_constraint=None,
                     bias_constraint=None,
                     trainable=True,
                     name=None,
                     reuse=None):
  """Functional interface for the depthwise separable 2D convolution layer.

  This layer performs a depthwise convolution that acts separately on
  channels, followed by a pointwise convolution that mixes channels.
  If `use_bias` is True and a bias initializer is provided,
  it adds a bias vector to the output.
  It then optionally applies an activation function to produce the final output.

  Arguments:
    inputs: Input tensor.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: A tuple or list of 2 integers specifying the spatial
      dimensions of the filters. Can be a single integer to specify the same
      value for all spatial dimensions.
    strides: A tuple or list of 2 positive integers specifying the strides
      of the convolution. Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any `stride` value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.

    dilation_rate: An integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to `num_filters_in * depth_multiplier`.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    depthwise_initializer: An initializer for the depthwise convolution kernel.
    pointwise_initializer: An initializer for the pointwise convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, no bias will
      be applied.
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
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
  """
  layer = SeparableConv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      depth_multiplier=depth_multiplier,
      activation=activation,
      use_bias=use_bias,
      depthwise_initializer=depthwise_initializer,
      pointwise_initializer=pointwise_initializer,
      bias_initializer=bias_initializer,
      depthwise_regularizer=depthwise_regularizer,
      pointwise_regularizer=pointwise_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      depthwise_constraint=depthwise_constraint,
      pointwise_constraint=pointwise_constraint,
      bias_constraint=bias_constraint,
      trainable=trainable,
      name=name,
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs)


class Conv2DTranspose(Conv2D):
  """Transposed 2D convolution layer (sometimes called 2D Deconvolution).

  The need for transposed convolutions generally arises
  from the desire to use a transformation going in the opposite direction
  of a normal convolution, i.e., from something that has the shape of the
  output of some convolution to something that has the shape of its input
  while maintaining a connectivity pattern that is compatible with
  said convolution.

  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: A tuple or list of 2 positive integers specifying the spatial
      dimensions of the filters. Can be a single integer to specify the same
      value for all spatial dimensions.
    strides: A tuple or list of 2 positive integers specifying the strides
      of the convolution. Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, no bias will
      be applied.
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
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
  """

  def __init__(self, filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format='channels_last',
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
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
        trainable=trainable,
        name=name,
        **kwargs)
    self.input_spec = base.InputSpec(ndim=4)

  def build(self, input_shape):
    if len(input_shape) != 4:
      raise ValueError('Inputs should have rank 4. Received input shape: ' + str(input_shape))
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis]
    self.input_spec = base.InputSpec(ndim=4, axes={channel_axis: input_dim})
    kernel_shape = self.kernel_size + (self.filters, input_dim)

    self.kernel = self.add_variable(name='kernel',
                                    shape=kernel_shape,
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint,
                                    trainable=True,
                                    dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_variable(name='bias',
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
      c_axis, h_axis, w_axis = 1, 2, 3
    else:
      c_axis, h_axis, w_axis = 3, 1, 2

    height, width = inputs_shape[h_axis], inputs_shape[w_axis]
    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides

    # Infer the dynamic output shape:
    out_height = utils.deconv_output_length(height,
                                            kernel_h,
                                            self.padding,
                                            stride_h)
    out_width = utils.deconv_output_length(width,
                                           kernel_w,
                                           self.padding,
                                           stride_w)
    if self.data_format == 'channels_first':
      output_shape = (batch_size, self.filters, out_height, out_width)
      strides = (1, 1, stride_h, stride_w)
    else:
      output_shape = (batch_size, out_height, out_width, self.filters)
      strides = (1, stride_h, stride_w, 1)

    output_shape_tensor = array_ops.stack(output_shape)
    outputs = nn.conv2d_transpose(
        inputs,
        self.kernel,
        output_shape_tensor,
        strides,
        padding=self.padding.upper(),
        data_format=utils.convert_data_format(self.data_format, ndim=4))

    if context.in_graph_mode():
      # Infer the static output shape:
      out_shape = inputs.get_shape().as_list()
      out_shape[c_axis] = self.filters
      out_shape[h_axis] = utils.deconv_output_length(out_shape[h_axis],
                                                     kernel_h,
                                                     self.padding,
                                                     stride_h)
      out_shape[w_axis] = utils.deconv_output_length(out_shape[w_axis],
                                                     kernel_w,
                                                     self.padding,
                                                     stride_w)
      outputs.set_shape(out_shape)

    if self.use_bias:
      outputs = nn.bias_add(
          outputs,
          self.bias,
          data_format=utils.convert_data_format(self.data_format, ndim=4))

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
    output_shape[h_axis] = utils.deconv_output_length(
        output_shape[h_axis], kernel_h, self.padding, stride_h)
    output_shape[w_axis] = utils.deconv_output_length(
        output_shape[w_axis], kernel_w, self.padding, stride_w)
    return tensor_shape.TensorShape(output_shape)


def conv2d_transpose(inputs,
                     filters,
                     kernel_size,
                     strides=(1, 1),
                     padding='valid',
                     data_format='channels_last',
                     activation=None,
                     use_bias=True,
                     kernel_initializer=None,
                     bias_initializer=init_ops.zeros_initializer(),
                     kernel_regularizer=None,
                     bias_regularizer=None,
                     activity_regularizer=None,
                     kernel_constraint=None,
                     bias_constraint=None,
                     trainable=True,
                     name=None,
                     reuse=None):
  """Functional interface for transposed 2D convolution layer.

  The need for transposed convolutions generally arises
  from the desire to use a transformation going in the opposite direction
  of a normal convolution, i.e., from something that has the shape of the
  output of some convolution to something that has the shape of its input
  while maintaining a connectivity pattern that is compatible with
  said convolution.

  Arguments:
    inputs: Input tensor.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: A tuple or list of 2 positive integers specifying the spatial
      dimensions of the filters. Can be a single integer to specify the same
      value for all spatial dimensions.
    strides: A tuple or list of 2 positive integers specifying the strides
      of the convolution. Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    activation: Activation function. Set it to `None` to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If `None`, then no
      bias will be applied.
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
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
  """
  layer = Conv2DTranspose(
      filters=filters,
      kernel_size=kernel_size,
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
      trainable=trainable,
      name=name,
      dtype=inputs.dtype.base_dtype,
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs)


class Conv3DTranspose(Conv3D):
  """Transposed 3D convolution layer (sometimes called 3D Deconvolution).

  Arguments:
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of 3 integers, specifying the
      depth, height and width of the 3D convolution window.
      Can be a single integer to specify the same value for all spatial
      dimensions.
    strides: An integer or tuple/list of 3 integers, specifying the strides
      of the convolution along the depth, height and width.
      Can be a single integer to specify the same value for all spatial
      dimensions.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, depth, height, width)`.
    activation: Activation function. Set it to `None` to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If `None`, then no
      bias will be applied.
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
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1, 1),
               padding='valid',
               data_format='channels_last',
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
    super(Conv3DTranspose, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
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
        trainable=trainable,
        name=name,
        **kwargs)
    self.input_spec = base.InputSpec(ndim=5)

  def build(self, input_shape):
    if len(input_shape) != 5:
      raise ValueError('Inputs should have rank 5, received input shape:',
                       str(input_shape))
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined, found None: ' + str(input_shape))
    input_dim = input_shape[channel_axis]
    kernel_shape = self.kernel_size + (self.filters, input_dim)

    self.kernel = self.add_variable(
        'kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_variable(
          'bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None

  def call(self, inputs):
    inputs_shape = array_ops.shape(inputs)
    batch_size = inputs_shape[0]
    if self.data_format == 'channels_first':
      c_axis, d_axis, h_axis, w_axis = 1, 2, 3, 4
    else:
      c_axis, d_axis, h_axis, w_axis = 4, 1, 2, 3

    self.input_spec = base.InputSpec(ndim=5,
                                     axes={c_axis: inputs_shape[c_axis]})

    depth = inputs_shape[d_axis]
    height = inputs_shape[h_axis]
    width = inputs_shape[w_axis]

    kernel_d, kernel_h, kernel_w = self.kernel_size
    stride_d, stride_h, stride_w = self.strides

    # Infer the dynamic output shape:
    out_depth = utils.deconv_output_length(depth,
                                           kernel_d,
                                           self.padding,
                                           stride_d)
    out_height = utils.deconv_output_length(height,
                                            kernel_h,
                                            self.padding,
                                            stride_h)
    out_width = utils.deconv_output_length(width,
                                           kernel_w,
                                           self.padding,
                                           stride_w)
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
        data_format=utils.convert_data_format(self.data_format, ndim=5),
        padding=self.padding.upper())

    if context.in_graph_mode():
      # Infer the static output shape:
      out_shape = inputs.get_shape().as_list()
      out_shape[c_axis] = self.filters
      out_shape[d_axis] = utils.deconv_output_length(out_shape[d_axis],
                                                     kernel_d,
                                                     self.padding,
                                                     stride_d)
      out_shape[h_axis] = utils.deconv_output_length(out_shape[h_axis],
                                                     kernel_h,
                                                     self.padding,
                                                     stride_h)
      out_shape[w_axis] = utils.deconv_output_length(out_shape[w_axis],
                                                     kernel_w,
                                                     self.padding,
                                                     stride_w)
      outputs.set_shape(out_shape)

    if self.use_bias:
      outputs_shape = outputs.shape.as_list()
      if self.data_format == 'channels_first':
        outputs_4d = array_ops.reshape(outputs, [
            outputs_shape[0], outputs_shape[1],
            outputs_shape[2] * outputs_shape[3], outputs_shape[4]
        ])
      else:
        outputs_4d = array_ops.reshape(outputs, [
            outputs_shape[0], outputs_shape[1] * outputs_shape[2],
            outputs_shape[3], outputs_shape[4]
        ])
      outputs_4d = nn.bias_add(
          outputs_4d,
          self.bias,
          data_format=utils.convert_data_format(self.data_format, ndim=4))
      outputs = array_ops.reshape(outputs_4d, outputs_shape)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = list(input_shape)
    if self.data_format == 'channels_first':
      c_axis, d_axis, h_axis, w_axis = 1, 2, 3, 4
    else:
      c_axis, d_axis, h_axis, w_axis = 4, 1, 2, 3

    kernel_d, kernel_h, kernel_w = self.kernel_size
    stride_d, stride_h, stride_w = self.strides

    output_shape[c_axis] = self.filters
    output_shape[d_axis] = utils.deconv_output_length(
        output_shape[d_axis], stride_d, kernel_d, self.padding)
    output_shape[h_axis] = utils.deconv_output_length(
        output_shape[h_axis], stride_h, kernel_h, self.padding)
    output_shape[w_axis] = utils.deconv_output_length(
        output_shape[w_axis], stride_w, kernel_w, self.padding)
    return tensor_shape.TensorShape(output_shape)


def conv3d_transpose(inputs,
                     filters,
                     kernel_size,
                     strides=(1, 1, 1),
                     padding='valid',
                     data_format='channels_last',
                     activation=None,
                     use_bias=True,
                     kernel_initializer=None,
                     bias_initializer=init_ops.zeros_initializer(),
                     kernel_regularizer=None,
                     bias_regularizer=None,
                     activity_regularizer=None,
                     kernel_constraint=None,
                     bias_constraint=None,
                     trainable=True,
                     name=None,
                     reuse=None):
  """Functional interface for transposed 3D convolution layer.

  Arguments:
    inputs: Input tensor.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: A tuple or list of 3 positive integers specifying the spatial
      dimensions of the filters. Can be a single integer to specify the same
      value for all spatial dimensions.
    strides: A tuple or list of 3 positive integers specifying the strides
      of the convolution. Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, depth, height, width)`.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, no bias will
      be applied.
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
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: A string, the name of the layer.
    reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
  """
  layer = Conv3DTranspose(
      filters=filters,
      kernel_size=kernel_size,
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
      trainable=trainable,
      name=name,
      dtype=inputs.dtype.base_dtype,
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs)


# Aliases

Convolution1D = Conv1D
Convolution2D = Conv2D
Convolution3D = Conv3D
SeparableConvolution2D = SeparableConv2D
Convolution2DTranspose = Deconvolution2D = Deconv2D = Conv2DTranspose
Convolution3DTranspose = Deconvolution3D = Deconv3D = Conv3DTranspose
convolution1d = conv1d
convolution2d = conv2d
convolution3d = conv3d
separable_convolution2d = separable_conv2d
convolution2d_transpose = deconvolution2d = deconv2d = conv2d_transpose
convolution3d_transpose = deconvolution3d = deconv3d = conv3d_transpose
