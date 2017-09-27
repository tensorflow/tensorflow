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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


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
                            mode='max', implementation='spp'):
  """
    Spatial pyramid pooling (SPP) is a pooling strategy to result in an output of fixed size.
    It will turn a 2D input of arbitrary size into an output of fixed dimension.
    Hence, the convolutional part of a DNN can be connected to a dense part
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
    implementation: The implementation to use, either 'spp' or 'spp_optimized'.
      `spp` is the original implementation from the paper, and supports variable
      sizes of input vectors, which `spp_optimized` does not support.

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
        implementation: The implementation to use, either 'spp' or 'spp_optimized'.
          `spp` is the original implementation from the paper, and supports variable
          sizes of input vectors, which `spp_optimized` does not support.
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


  def __init__(self, dimensions=None, mode='max', implementation='spp', **kwargs):
    super(SpatialPyramidPooling, self).__init__(**kwargs)
    self.implementation = implementation
    self.mode = mode
    self.dimensions = dimensions if dimensions is not None else [4, 2, 1]

  def call(self, inputs):
    pool_list = []
    if self.implementation == 'spp':
      for pool_dim in self.dimensions:
        pool_list += max_pool_2d_nxn_regions(inputs, pool_dim, self.mode)
    elif self.implementation == 'spp_optimized':
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
    else:
        raise ValueError('Unknown implementation', self.implementation)


    return array_ops.concat(values=pool_list, axis=1)

  def _compute_output_shape(self, input_shape):
    num_features = sum(p * p for p in self.dimensions)
    return tensor_shape.TensorShape([None, input_shape[0] * num_features])
