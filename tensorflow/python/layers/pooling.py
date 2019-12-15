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

"""Contains the pooling layer classes and their functional aliases.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.layers import base
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=['layers.AveragePooling1D'])
class AveragePooling1D(keras_layers.AveragePooling1D, base.Layer):
  """Average Pooling layer for 1D inputs.

  Arguments:
    pool_size: An integer or tuple/list of a single integer,
      representing the size of the pooling window.
    strides: An integer or tuple/list of a single integer, specifying the
      strides of the pooling operation.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, length)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    if strides is None:
      raise ValueError('Argument `strides` must not be None.')
    super(AveragePooling1D, self).__init__(
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name,
        **kwargs)


@deprecation.deprecated(
    date=None, instructions='Use keras.layers.AveragePooling1D instead.')
@tf_export(v1=['layers.average_pooling1d'])
def average_pooling1d(inputs, pool_size, strides,
                      padding='valid', data_format='channels_last',
                      name=None):
  """Average Pooling layer for 1D inputs.

  Arguments:
    inputs: The tensor over which to pool. Must have rank 3.
    pool_size: An integer or tuple/list of a single integer,
      representing the size of the pooling window.
    strides: An integer or tuple/list of a single integer, specifying the
      strides of the pooling operation.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, length)`.
    name: A string, the name of the layer.

  Returns:
    The output tensor, of rank 3.

  Raises:
    ValueError: if eager execution is enabled.
  """
  layer = AveragePooling1D(pool_size=pool_size,
                           strides=strides,
                           padding=padding,
                           data_format=data_format,
                           name=name)
  return layer.apply(inputs)


@tf_export(v1=['layers.MaxPooling1D'])
class MaxPooling1D(keras_layers.MaxPooling1D, base.Layer):
  """Max Pooling layer for 1D inputs.

  Arguments:
    pool_size: An integer or tuple/list of a single integer,
      representing the size of the pooling window.
    strides: An integer or tuple/list of a single integer, specifying the
      strides of the pooling operation.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, length)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    if strides is None:
      raise ValueError('Argument `strides` must not be None.')
    super(MaxPooling1D, self).__init__(
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name,
        **kwargs)


@deprecation.deprecated(
    date=None, instructions='Use keras.layers.MaxPooling1D instead.')
@tf_export(v1=['layers.max_pooling1d'])
def max_pooling1d(inputs, pool_size, strides,
                  padding='valid', data_format='channels_last',
                  name=None):
  """Max Pooling layer for 1D inputs.

  Arguments:
    inputs: The tensor over which to pool. Must have rank 3.
    pool_size: An integer or tuple/list of a single integer,
      representing the size of the pooling window.
    strides: An integer or tuple/list of a single integer, specifying the
      strides of the pooling operation.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, length, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, length)`.
    name: A string, the name of the layer.

  Returns:
    The output tensor, of rank 3.

  Raises:
    ValueError: if eager execution is enabled.
  """
  layer = MaxPooling1D(pool_size=pool_size,
                       strides=strides,
                       padding=padding,
                       data_format=data_format,
                       name=name)
  return layer.apply(inputs)


@tf_export(v1=['layers.AveragePooling2D'])
class AveragePooling2D(keras_layers.AveragePooling2D, base.Layer):
  """Average pooling layer for 2D inputs (e.g. images).

  Arguments:
    pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    if strides is None:
      raise ValueError('Argument `strides` must not be None.')
    super(AveragePooling2D, self).__init__(
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, name=name, **kwargs)


@deprecation.deprecated(
    date=None, instructions='Use keras.layers.AveragePooling2D instead.')
@tf_export(v1=['layers.average_pooling2d'])
def average_pooling2d(inputs,
                      pool_size, strides,
                      padding='valid', data_format='channels_last',
                      name=None):
  """Average pooling layer for 2D inputs (e.g. images).

  Arguments:
    inputs: The tensor over which to pool. Must have rank 4.
    pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    name: A string, the name of the layer.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
  """
  layer = AveragePooling2D(pool_size=pool_size, strides=strides,
                           padding=padding, data_format=data_format,
                           name=name)
  return layer.apply(inputs)


@tf_export(v1=['layers.MaxPooling2D'])
class MaxPooling2D(keras_layers.MaxPooling2D, base.Layer):
  """Max pooling layer for 2D inputs (e.g. images).

  Arguments:
    pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    if strides is None:
      raise ValueError('Argument `strides` must not be None.')
    super(MaxPooling2D, self).__init__(
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, name=name, **kwargs)


@deprecation.deprecated(
    date=None, instructions='Use keras.layers.MaxPooling2D instead.')
@tf_export(v1=['layers.max_pooling2d'])
def max_pooling2d(inputs,
                  pool_size, strides,
                  padding='valid', data_format='channels_last',
                  name=None):
  """Max pooling layer for 2D inputs (e.g. images).

  Arguments:
    inputs: The tensor over which to pool. Must have rank 4.
    pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    name: A string, the name of the layer.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
  """
  layer = MaxPooling2D(pool_size=pool_size, strides=strides,
                       padding=padding, data_format=data_format,
                       name=name)
  return layer.apply(inputs)


@tf_export(v1=['layers.AveragePooling3D'])
class AveragePooling3D(keras_layers.AveragePooling3D, base.Layer):
  """Average pooling layer for 3D inputs (e.g. volumes).

  Arguments:
    pool_size: An integer or tuple/list of 3 integers:
      (pool_depth, pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, depth, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    if strides is None:
      raise ValueError('Argument `strides` must not be None.')
    super(AveragePooling3D, self).__init__(
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, name=name, **kwargs)


@deprecation.deprecated(
    date=None, instructions='Use keras.layers.AveragePooling3D instead.')
@tf_export(v1=['layers.average_pooling3d'])
def average_pooling3d(inputs,
                      pool_size, strides,
                      padding='valid', data_format='channels_last',
                      name=None):
  """Average pooling layer for 3D inputs (e.g. volumes).

  Arguments:
    inputs: The tensor over which to pool. Must have rank 5.
    pool_size: An integer or tuple/list of 3 integers:
      (pool_depth, pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, depth, height, width)`.
    name: A string, the name of the layer.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
  """
  layer = AveragePooling3D(pool_size=pool_size, strides=strides,
                           padding=padding, data_format=data_format,
                           name=name)
  return layer.apply(inputs)


@tf_export(v1=['layers.MaxPooling3D'])
class MaxPooling3D(keras_layers.MaxPooling3D, base.Layer):
  """Max pooling layer for 3D inputs (e.g. volumes).

  Arguments:
    pool_size: An integer or tuple/list of 3 integers:
      (pool_depth, pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, depth, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    if strides is None:
      raise ValueError('Argument `strides` must not be None.')
    super(MaxPooling3D, self).__init__(
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, name=name, **kwargs)


@deprecation.deprecated(
    date=None, instructions='Use keras.layers.MaxPooling3D instead.')
@tf_export(v1=['layers.max_pooling3d'])
def max_pooling3d(inputs,
                  pool_size, strides,
                  padding='valid', data_format='channels_last',
                  name=None):
  """Max pooling layer for 3D inputs (e.g.

  volumes).

  Arguments:
    inputs: The tensor over which to pool. Must have rank 5.
    pool_size: An integer or tuple/list of 3 integers: (pool_depth, pool_height,
      pool_width) specifying the size of the pooling window. Can be a single
      integer to specify the same value for all spatial dimensions.
    strides: An integer or tuple/list of 3 integers, specifying the strides of
      the pooling operation. Can be a single integer to specify the same value
      for all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    data_format: A string. The ordering of the dimensions in the inputs.
      `channels_last` (default) and `channels_first` are supported.
      `channels_last` corresponds to inputs with shape `(batch, depth, height,
      width, channels)` while `channels_first` corresponds to inputs with shape
      `(batch, channels, depth, height, width)`.
    name: A string, the name of the layer.

  Returns:
    Output tensor.

  Raises:
    ValueError: if eager execution is enabled.
  """
  layer = MaxPooling3D(pool_size=pool_size, strides=strides,
                       padding=padding, data_format=data_format,
                       name=name)
  return layer.apply(inputs)

# Aliases

AvgPool2D = AveragePooling2D
MaxPool2D = MaxPooling2D
max_pool2d = max_pooling2d
avg_pool2d = average_pooling2d
