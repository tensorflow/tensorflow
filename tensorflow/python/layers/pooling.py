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
"""Contains the pooling layer classes and their functional aliases.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


class _Pooling1D(base.Layer):
  """Pooling layer for arbitrary pooling functions, for 1D inputs.

  This class only exists for code reuse. It will never be an exposed API.

  Arguments:
    pool_function: The pooling function to apply, e.g. `tf.nn.max_pool`.
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

  def __init__(self, pool_function, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    super(_Pooling1D, self).__init__(name=name, **kwargs)
    self.pool_function = pool_function
    self.pool_size = utils.normalize_tuple(pool_size, 1, 'pool_size')
    self.strides = utils.normalize_tuple(strides, 1, 'strides')
    self.padding = utils.normalize_padding(padding)
    self.data_format = utils.normalize_data_format(data_format)
    self.input_spec = base.InputSpec(ndim=3)

  def call(self, inputs):
    # There is no TF op for 1D pooling, hence we make the inputs 4D.
    if self.data_format == 'channels_last':
      inputs = array_ops.expand_dims(inputs, 2)
      pool_shape = (1,) + self.pool_size + (1, 1)
      strides = (1,) + self.strides + (1, 1)
      data_format = 'NHWC'
    else:
      inputs = array_ops.expand_dims(inputs, 1)
      pool_shape = (1, 1) + self.pool_size + (1,)
      strides = (1, 1) + self.strides + (1,)
      data_format = 'NCHW'

    outputs = self.pool_function(
        inputs,
        ksize=pool_shape,
        strides=strides,
        padding=self.padding.upper(),
        data_format=data_format)

    if self.data_format == 'channels_last':
      return array_ops.squeeze(outputs, 2)
    else:
      return array_ops.squeeze(outputs, 1)

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    length = utils.conv_output_length(input_shape[1], self.pool_size[0],
                                      self.padding, self.strides[0])
    return tensor_shape.TensorShape([input_shape[0], length, input_shape[2]])


class AveragePooling1D(_Pooling1D):
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
    super(AveragePooling1D, self).__init__(
        nn.avg_pool,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name,
        **kwargs)


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
  """
  layer = AveragePooling1D(pool_size=pool_size,
                           strides=strides,
                           padding=padding,
                           data_format=data_format,
                           name=name)
  return layer.apply(inputs)


class MaxPooling1D(_Pooling1D):
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
    super(MaxPooling1D, self).__init__(
        nn.max_pool,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name,
        **kwargs)


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
  """
  layer = MaxPooling1D(pool_size=pool_size,
                       strides=strides,
                       padding=padding,
                       data_format=data_format,
                       name=name)
  return layer.apply(inputs)


class _Pooling2D(base.Layer):
  """Pooling layer for arbitrary pooling functions, for 2D inputs (e.g. images).

  This class only exists for code reuse. It will never be an exposed API.

  Arguments:
    pool_function: The pooling function to apply, e.g. `tf.nn.max_pool`.
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
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_function, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    super(_Pooling2D, self).__init__(name=name, **kwargs)
    self.pool_function = pool_function
    self.pool_size = utils.normalize_tuple(pool_size, 2, 'pool_size')
    self.strides = utils.normalize_tuple(strides, 2, 'strides')
    self.padding = utils.normalize_padding(padding)
    self.data_format = utils.normalize_data_format(data_format)
    self.input_spec = base.InputSpec(ndim=4)

  def call(self, inputs):
    if self.data_format == 'channels_last':
      pool_shape = (1,) + self.pool_size + (1,)
      strides = (1,) + self.strides + (1,)
    else:
      pool_shape = (1, 1) + self.pool_size
      strides = (1, 1) + self.strides
    outputs = self.pool_function(
        inputs,
        ksize=pool_shape,
        strides=strides,
        padding=self.padding.upper(),
        data_format=utils.convert_data_format(self.data_format, 4))
    return outputs

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      rows = input_shape[2]
      cols = input_shape[3]
    else:
      rows = input_shape[1]
      cols = input_shape[2]
    rows = utils.conv_output_length(rows, self.pool_size[0], self.padding,
                                    self.strides[0])
    cols = utils.conv_output_length(cols, self.pool_size[1], self.padding,
                                    self.strides[1])
    if self.data_format == 'channels_first':
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], rows, cols])
    else:
      return tensor_shape.TensorShape(
          [input_shape[0], rows, cols, input_shape[3]])


class AveragePooling2D(_Pooling2D):
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
    super(AveragePooling2D, self).__init__(
        nn.avg_pool,
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, name=name, **kwargs)


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
  """
  layer = AveragePooling2D(pool_size=pool_size, strides=strides,
                           padding=padding, data_format=data_format,
                           name=name)
  return layer.apply(inputs)


class MaxPooling2D(_Pooling2D):
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
    super(MaxPooling2D, self).__init__(
        nn.max_pool,
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, name=name, **kwargs)


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
  """
  layer = MaxPooling2D(pool_size=pool_size, strides=strides,
                       padding=padding, data_format=data_format,
                       name=name)
  return layer.apply(inputs)



def max_pool_2d_nxn_regions(inputs, pool_dimension, mode):
  """
  Args:
    inputs: The tensor over which to pool. Must have rank 4.
    pool_dimension: The dimenstion level(bin size)
      over which spatial pooling is performed.
    mode: Pooling mode 'max' or 'avg'.

  Returns:
    The output list of (pool_dimension * pool_dimension) tensors.

  """
  inputs_shape = array_ops.shape(inputs)
  h = math_ops.cast(array_ops.gather(inputs_shape, 1), dtypes.int32)
  w = math_ops.cast(array_ops.gather(inputs_shape, 2), dtypes.int32)

  if mode == 'max':
    pooling_op = math_ops.reduce_max
  elif mode == 'avg':
    pooling_op = math_ops.reduce_mean
  else:
    msg = "Mode must be either 'max' or 'avg'. Got '{0}'"
    raise ValueError(msg.format(mode))

  result = []
  n = pool_dimension
  for row in range(pool_dimension):
    for col in range(pool_dimension):
      # start_h = floor(row / n * h)
      start_h = math_ops.cast(math_ops.floor(math_ops.multiply(math_ops.divide(row, n), math_ops.cast(h, dtypes.float32))), dtypes.int32)
      # end_h = ceil((row + 1) / n * h)
      end_h = math_ops.cast(math_ops.ceil(math_ops.multiply(math_ops.divide((row + 1), n), math_ops.cast(h, dtypes.float32))), dtypes.int32)
      # start_w = floor(col / n * w)
      start_w = math_ops.cast(math_ops.floor(math_ops.multiply(math_ops.divide(col, n), math_ops.cast(w, dtypes.float32))), dtypes.int32)
      # end_w = ceil((col + 1) / n * w)
      end_w = math_ops.cast(math_ops.ceil(math_ops.multiply(math_ops.divide((col + 1), n), math_ops.cast(w, dtypes.float32))), dtypes.int32)
      pooling_region = inputs[:, start_h:end_h, start_w:end_w, :]
      pool_result = pooling_op(pooling_region, axis=(1, 2))
      result.append(pool_result)
  return result


def spatial_pyramid_pooling(inputs, dimensions=None,
                            mode='max', implementation='kaiming'):
  """
    Spatial pyramid pooling (SPP) is a pooling strategy to result in an output of fixed size.
    It will turn a 2D input of arbitrary size into an output of fixed dimension.
    Hence, the convlutional part of a DNN can be connected to a dense part
    with a fixed number of nodes even if the dimensions of the input
    image are unknown.
    The pooling is performed over :math:`l` pooling levels.
    Each pooling level :math:`i` will create :math:`M_i` output features.
    :math:`M_i` is given by :math:`n_i * n_i`, with :math:`n_i` as the number
    of pooling operations per dimension level :math:`i`.
    The length of the parameter dimensions is the level of the spatial pyramid.

  Args:
    inputs: The tensor over which to pool. Must have rank 4.
    dimensions: The list of bin sizes over which pooling is to be done.
    mode: Pooling mode 'max' or 'avg'.
    implementation: The implementation to use, either 'kaiming' or 'fast'.
      kamming is the original implementation from the paper, and supports variable
      sizes of input vectors, which fast does not support.

  Returns:
    Output tensor.
  """
  layer = SpatialPyramidPooling(dimensions=dimensions,
                                mode=mode,
                                implementation=implementation)
  return layer.apply(inputs)


class SpatialPyramidPooling(base.Layer):
  """
    Spatial pyramid pooling (SPP) is a pooling strategy to result in an output of fixed size.
    Arguments:
        dimensions: The list of :math:`n_i`'s that define the output dimension
          of each pooling level :math:`i`. The length of dimensions is the level of
          the spatial pyramid.
        mode: Pooling mode 'max' or 'avg'.
        implementation: The implementation to use, either 'kaiming' or 'fast'.
          kaiming is the original implementation from the paper, and supports variable
          sizes of input vectors, which fast does not support.
    Notes:
        SPP should be inserted between the convolutional part of a Deep Network and it's
        dense part. Convolutions can be used for arbitrary input dimensions, but
        the size of their output will depend on their input dimensions.
        Connecting the output of the convolutional to the dense part then
        usually demands us to fix the dimensons of the network's input.
        The spatial pyramid pooling layer, however, allows us to leave
        the network input dimensions arbitrary.
        The advantage over a global pooling layer is the added robustness
        against object deformations due to the pooling on different scales.
    References:
        [1] He, Kaiming et al (2015): Spatial Pyramid Pooling in Deep Convolutional Networks
            for Visual Recognition. https://arxiv.org/pdf/1406.4729.pdf.
    Ported from: https://github.com/Lasagne/Lasagne/pull/799
  """


  def __init__(self, dimensions=None, mode='max', implementation='kaiming', **kwargs):
    super(SpatialPyramidPooling, self).__init__(**kwargs)
    self.implementation = implementation
    self.mode = mode
    self.dimensions = dimensions if dimensions is not None else [4, 2, 1]

  def call(self, inputs):
    pool_list = []
    if self.implementation == 'kaiming':
      for pool_dim in self.dimensions:
        pool_list += max_pool_2d_nxn_regions(inputs, pool_dim, self.mode)
    else:
      input_shape = inputs.get_shape().as_list()
      for pool_dim in self.dimensions:
        h, w = input_shape[1], input_shape[2]

        ph = np.ceil(h * 1.0 / pool_dim).astype(np.int32)
        pw = np.ceil(w * 1.0 / pool_dim).astype(np.int32)
        sh = np.floor(h * 1.0 / pool_dim + 1).astype(np.int32)
        sw = np.floor(w * 1.0 / pool_dim + 1).astype(np.int32)
        pool_result = nn.max_pool(inputs,
                                  ksize=[1, ph, pw, 1],
                                  strides=[1, sh, sw, 1],
                                  padding='SAME')
        pool_list.append(array_ops.reshape(pool_result, [array_ops.shape(inputs)[0], -1]))
    return array_ops.concat(values=pool_list, axis=1)

  def _compute_output_shape(self, input_shape):
    num_features = sum(p * p for p in self.dimensions)
    return tensor_shape.TensorShape([None, input_shape[0] * num_features])

class _Pooling3D(base.Layer):
  """Pooling layer for arbitrary pooling functions, for 3D inputs.

  This class only exists for code reuse. It will never be an exposed API.

  Arguments:
    pool_function: The pooling function to apply, e.g. `tf.nn.max_pool`.
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
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, depth, height, width, channels)`
      while `channels_first` corresponds to
      inputs with shape `(batch, channels, depth, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_function, pool_size, strides,
               padding='valid', data_format='channels_last',
               name=None, **kwargs):
    super(_Pooling3D, self).__init__(name=name, **kwargs)
    self.pool_function = pool_function
    self.pool_size = utils.normalize_tuple(pool_size, 3, 'pool_size')
    self.strides = utils.normalize_tuple(strides, 3, 'strides')
    self.padding = utils.normalize_padding(padding)
    self.data_format = utils.normalize_data_format(data_format)
    self.input_spec = base.InputSpec(ndim=5)

  def call(self, inputs):
    pool_shape = (1,) + self.pool_size + (1,)
    strides = (1,) + self.strides + (1,)

    if self.data_format == 'channels_first':
      # TF does not support `channels_first` with 3D pooling operations,
      # so we must handle this case manually.
      # TODO(fchollet): remove this when TF pooling is feature-complete.
      inputs = array_ops.transpose(inputs, (0, 2, 3, 4, 1))

    outputs = self.pool_function(
        inputs,
        ksize=pool_shape,
        strides=strides,
        padding=self.padding.upper())

    if self.data_format == 'channels_first':
      outputs = array_ops.transpose(outputs, (0, 4, 1, 2, 3))
    return outputs

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      len_dim1 = input_shape[2]
      len_dim2 = input_shape[3]
      len_dim3 = input_shape[4]
    else:
      len_dim1 = input_shape[1]
      len_dim2 = input_shape[2]
      len_dim3 = input_shape[3]
    len_dim1 = utils.conv_output_length(len_dim1, self.pool_size[0],
                                        self.padding, self.strides[0])
    len_dim2 = utils.conv_output_length(len_dim2, self.pool_size[1],
                                        self.padding, self.strides[1])
    len_dim3 = utils.conv_output_length(len_dim3, self.pool_size[2],
                                        self.padding, self.strides[2])
    if self.data_format == 'channels_first':
      return tensor_shape.TensorShape(
          [input_shape[0], input_shape[1], len_dim1, len_dim2, len_dim3])
    else:
      return tensor_shape.TensorShape(
          [input_shape[0], len_dim1, len_dim2, len_dim3, input_shape[4]])


class AveragePooling3D(_Pooling3D):
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
    super(AveragePooling3D, self).__init__(
        nn.avg_pool3d,
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, name=name, **kwargs)


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
  """
  layer = AveragePooling3D(pool_size=pool_size, strides=strides,
                           padding=padding, data_format=data_format,
                           name=name)
  return layer.apply(inputs)


class MaxPooling3D(_Pooling3D):
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
    super(MaxPooling3D, self).__init__(
        nn.max_pool3d,
        pool_size=pool_size, strides=strides,
        padding=padding, data_format=data_format, name=name, **kwargs)


def max_pooling3d(inputs,
                  pool_size, strides,
                  padding='valid', data_format='channels_last',
                  name=None):
  """Max pooling layer for 3D inputs (e.g. volumes).

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
