# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Reversible residual network compatible with eager execution.

Customized basic operations.

Reference [The Reversible Residual Network: Backpropagation
Without Storing Activations](https://arxiv.org/pdf/1707.04585.pdf)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def downsample(x, filters, strides, axis=1):
  """Downsample feature map with avg pooling, if filter size doesn't match."""

  def pad_strides(strides, axis=1):
    """Convert length 2 to length 4 strides.

    Needed since `tf.layers.Conv2D` uses length 2 strides, whereas operations
    such as `tf.nn.avg_pool` use length 4 strides.

    Args:
      strides: length 2 list/tuple strides for height and width
      axis: integer specifying feature dimension according to data format
    Returns:
      length 4 strides padded with 1 on batch and channel dimension
    """

    assert len(strides) == 2

    if axis == 1:
      return [1, 1, strides[0], strides[1]]
    return [1, strides[0], strides[1], 1]

  assert len(x.shape) == 4 and (axis == 1 or axis == 3)

  data_format = "NCHW" if axis == 1 else "NHWC"
  strides_ = pad_strides(strides, axis=axis)

  if strides[0] > 1:
    x = tf.nn.avg_pool(
        x, strides_, strides_, padding="VALID", data_format=data_format)

  in_filter = x.shape[axis]
  out_filter = filters

  if in_filter < out_filter:
    pad_size = [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]
    if axis == 1:
      x = tf.pad(x, [[0, 0], pad_size, [0, 0], [0, 0]])
    else:
      x = tf.pad(x, [[0, 0], [0, 0], [0, 0], pad_size])
  # In case `tape.gradient(x, [x])` produces a list of `None`
  return x + 0.
