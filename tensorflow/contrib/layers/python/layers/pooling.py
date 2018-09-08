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

from tensorflow.contrib.keras.python.keras.engine import InputSpec
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def spatial_pyramid_pooling(inputs, bin_dimensions=None, pooling_mode='max'):
  """Spatial pyramid pooling (SPP) is a pooling strategy to result in an output of fixed size.

    This layer allows the network to use arbitrary input dimensions. It generates fixed-length representations
    for each feature map. It should be inserted before any fully-connected layer in the network.
    Pyramid pooling is robust to object deformations as pooling is done at different scales.

  Args:
    inputs: The tensor over which to pool. Must have rank 4.
    bin_dimensions: The list of number of pool region for pooling. The length of pool_dims
      is the level of the spatial pyramid. Each int in the list is the number of regions in
      that pool. For e.g. [1, 2, 4] would be 3 regions with 1x1, 2x2 and 4x4 pools, so 21
      outputs per feature map.
    pooling_mode: Pooling mode 'max' or 'avg'.

  Returns:
    Output tensor.
  """
  layer = SpatialPyramidPooling(bin_dimensions=bin_dimensions, pooling_mode=pooling_mode)
  return layer.apply(inputs)


class SpatialPyramidPooling(base.Layer):
  """Spatial pyramid pooling (SPP) is a pooling strategy to result in an output of fixed size.

    This layer allows the network to use arbitrary input dimensions. It generates fixed-length representations
    for each feature map. It should be inserted before any fully-connected layer in the network.
    Pyramid pooling is robust to object deformations as pooling is done at different scales.

    Arguments:
      bin_dimensions: The list of different scales at which pooling is performed.
        The length of bin_dimensions is the level of the spatial pyramid. Each int in the list
        is the number of regions in that pool. For e.g. [1, 2, 4] would be 3 regions with
        1x1, 2x2 and 4x4 pools, so 21 outputs per feature map.
      pooling_mode: Pooling mode 'max' or 'avg'.
  """

  def __init__(self, bin_dimensions=None, pooling_mode='max', **kwargs):
    super(SpatialPyramidPooling, self).__init__(**kwargs)
    self.pooling_mode = pooling_mode
    self.bin_dimensions = [4, 2, 1]
    if bin_dimensions is not None:
      self.bin_dimensions = bin_dimensions
    self.input_spec = InputSpec(ndim=4)

  def call(self, inputs, **kwargs):
    pool_list = []
    for bin_dimension in self.bin_dimensions:
      pool_list += self._spatial_pooling_in_bins(inputs, bin_dimension)
    return array_ops.concat(values=pool_list, axis=1)

  def _compute_output_shape(self, input_shape):
    num_features = sum(p * p for p in self.bin_dimensions)
    return tensor_shape.TensorShape([None, input_shape[0] * num_features])

  def _spatial_pooling_in_bins(self, inputs, bin_dimension):
    """Spatial pyramid pooling (SPP) is a pooling strategy to result in an output of fixed size.

    Args:
      inputs: The tensor over which to pool. Must have rank 4.
      bin_dimension: It defines the number of pools region for the operation. For e.g. bin_dimension = 2 will result in
        a 2x2 pools per feature map.

    Returns:
      The output list of (bin_dimension * bin_dimension) tensors.

    Raises:
      ValueError: If `mode` is neither `max` nor `avg`.

    """
    inputs_shape = array_ops.shape(inputs)
    input_height = math_ops.cast(array_ops.gather(inputs_shape, 1), dtypes.float32)
    input_width = math_ops.cast(array_ops.gather(inputs_shape, 2), dtypes.float32)

    if self.pooling_mode == 'max':
      pooling_op = math_ops.reduce_max
    elif self.pooling_mode == 'avg':
      pooling_op = math_ops.reduce_mean
    else:
      msg = "Mode must be either 'max' or 'avg'. Got '{0}'"
      raise ValueError(msg.format(self.pooling_mode))

    result = []
    for row in range(bin_dimension):
      for col in range(bin_dimension):
        start_h = math_ops.cast(math_ops.floor(math_ops.multiply(math_ops.divide(row, bin_dimension),
                                                                 input_height)), dtypes.int32)
        end_h = math_ops.cast(math_ops.ceil(math_ops.multiply(math_ops.divide((row + 1), bin_dimension),
                                                              input_height)), dtypes.int32)
        start_w = math_ops.cast(math_ops.floor(math_ops.multiply(math_ops.divide(col, bin_dimension),
                                                                 input_width)), dtypes.int32)
        end_w = math_ops.cast(math_ops.ceil(math_ops.multiply(math_ops.divide((col + 1), bin_dimension),
                                                              input_width)), dtypes.int32)

        pooling_region = inputs[:, start_h:end_h, start_w:end_w, :]
        pool_result = pooling_op(pooling_region, axis=(1, 2))
        result.append(pool_result)
    return result
