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

"""Contains the convolutional layer classes and their functional aliases.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.layers import base
from tensorflow.python.ops import init_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=['layers.Conv1D'])
class Conv1D(keras_layers.Conv1D, base.Layer):
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
    super(Conv1D, self).__init__(
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


@deprecation.deprecated(
    date=None,
    instructions='Use keras.layers.conv1d instead.')
@tf_export(v1=['layers.conv1d'])
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
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs)


@tf_export(v1=['layers.Conv2D'])
class Conv2D(keras_layers.Conv2D, base.Layer):
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


@deprecation.deprecated(
    date=None,
    instructions='Use keras.layers.conv2d instead.')
@tf_export(v1=['layers.conv2d'])
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
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs)


@tf_export(v1=['layers.Conv3D'])
class Conv3D(keras_layers.Conv3D, base.Layer):
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


@deprecation.deprecated(
    date=None,
    instructions='Use keras.layers.conv3d instead.')
@tf_export(v1=['layers.conv3d'])
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
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs)


@tf_export(v1=['layers.SeparableConv1D'])
class SeparableConv1D(keras_layers.SeparableConv1D, base.Layer):
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
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, length)`.
    dilation_rate: A single integer, specifying
      the dilation rate to use for dilated convolution.
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
    super(SeparableConv1D, self).__init__(
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
        **kwargs)


@tf_export(v1=['layers.SeparableConv2D'])
class SeparableConv2D(keras_layers.SeparableConv2D, base.Layer):
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
        **kwargs)


@deprecation.deprecated(
    date=None,
    instructions='Use keras.layers.separable_conv1d instead.')
@tf_export(v1=['layers.separable_conv1d'])
def separable_conv1d(inputs,
                     filters,
                     kernel_size,
                     strides=1,
                     padding='valid',
                     data_format='channels_last',
                     dilation_rate=1,
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
  """Functional interface for the depthwise separable 1D convolution layer.

  This layer performs a depthwise convolution that acts separately on
  channels, followed by a pointwise convolution that mixes channels.
  If `use_bias` is True and a bias initializer is provided,
  it adds a bias vector to the output.
  It then optionally applies an activation function to produce the final output.

  Arguments:
    inputs: Input tensor.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: A single integer specifying the spatial
      dimensions of the filters.
    strides: A single integer specifying the strides
      of the convolution.
      Specifying any `stride` value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, length)`.
    dilation_rate: A single integer, specifying
      the dilation rate to use for dilated convolution.
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
  layer = SeparableConv1D(
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


@deprecation.deprecated(
    date=None,
    instructions='Use keras.layers.separable_conv2d instead.')
@tf_export(v1=['layers.separable_conv2d'])
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


@tf_export(v1=['layers.Conv2DTranspose'])
class Conv2DTranspose(keras_layers.Conv2DTranspose, base.Layer):
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


@deprecation.deprecated(
    date=None,
    instructions='Use keras.layers.conv2d_transpose instead.')
@tf_export(v1=['layers.conv2d_transpose'])
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
    bias_initializer: An initializer for the bias vector. If `None`, the default
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
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs)


@tf_export(v1=['layers.Conv3DTranspose'])
class Conv3DTranspose(keras_layers.Conv3DTranspose, base.Layer):
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
    bias_initializer: An initializer for the bias vector. If `None`, the default
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


@deprecation.deprecated(
    date=None,
    instructions='Use keras.layers.conv3d_transpose instead.')
@tf_export(v1=['layers.conv3d_transpose'])
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
