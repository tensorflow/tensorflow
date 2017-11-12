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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops




def spatial_pyramid_pooling(inputs, spatial_bin_dimensions=None, mode='max'):
  """Spatial pyramid pooling (SPP) is a pooling strategy to result in an output of fixed size.

  Args:
    inputs: The tensor over which to pool. Must have rank 4.
    spatial_bin_dimensions: The list of bin sizes over which pooling is to be done.
    mode: Pooling mode 'max' or 'avg'.

  Returns:
    Output tensor.
  """
  layer = SpatialPyramidPooling(spatial_bin_dimensions=spatial_bin_dimensions, mode=mode)
  return layer.apply(inputs)


class SpatialPyramidPooling(base.Layer):
  """Spatial pyramid pooling (SPP) is a pooling strategy to result in an output of fixed size.

    Arguments:
        spatial_bin_dimensions: The list of dimensions define the output dimension
          of each pooling level. The value of each dimenstion is the level of
          the spatial pyramid.
        mode: Pooling mode 'max' or 'avg'.
  """

  def __init__(self, spatial_bin_dimensions=None, mode='max', **kwargs):
    super(SpatialPyramidPooling, self).__init__(**kwargs)
    self.mode = mode
    self.spatial_bin_dimensions = spatial_bin_dimensions if spatial_bin_dimensions is not None else [4, 2, 1]

  def call(self, inputs):
    pool_list = []
    for bin_dimension in self.spatial_bin_dimensions:
      pool_list += self.max_pool_2d_nxn_regions(inputs, bin_dimension)
    return array_ops.concat(values=pool_list, axis=1)

  def _compute_output_shape(self, input_shape):
    num_features = sum(p * p for p in self.spatial_bin_dimensions)
    return tensor_shape.TensorShape([None, input_shape[0] * num_features])


  def max_pool_2d_nxn_regions(self, inputs, bin_dimension):
    """
    Args:
      inputs: The tensor over which to pool. Must have rank 4.
      bin_dimension: The list of bin dimenstions (bin size) over which
        spatial pooling is performed.
      mode: Pooling mode `max` or `avg`.

    Returns:
      The output list of (bin_dimension * bin_dimension) tensors.

    Raises:
      ValueError: If `mode` is neither `max` nor `avg`.

    """
    inputs_shape = array_ops.shape(inputs)
    input_height = math_ops.cast(array_ops.gather(inputs_shape, 1), dtypes.float32)
    input_width = math_ops.cast(array_ops.gather(inputs_shape, 2), dtypes.float32)

    if self.mode == 'max':
      pooling_op = math_ops.reduce_max
    elif self.mode == 'avg':
      pooling_op = math_ops.reduce_mean
    else:
      msg = "Mode must be either 'max' or 'avg'. Got '{0}'"
      raise ValueError(msg.format(self.mode))

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
