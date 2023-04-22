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
"""A library of common shape functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.framework import tensor_shape


def _broadcast_shape_helper(shape_x, shape_y):
  """Helper functions for is_broadcast_compatible and broadcast_shape.

  Args:
    shape_x: A `TensorShape`
    shape_y: A `TensorShape`

  Returns:
    Returns None if the shapes are not broadcast compatible,
    a list of the broadcast dimensions otherwise.
  """
  # To compute the broadcasted dimensions, we zip together shape_x and shape_y,
  # and pad with 1 to make them the same length.
  broadcasted_dims = reversed(list(six.moves.zip_longest(
      reversed(shape_x.dims),
      reversed(shape_y.dims),
      fillvalue=tensor_shape.Dimension(1))))
  # Next we combine the dimensions according to the numpy broadcasting rules.
  # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
  return_dims = []
  for (dim_x, dim_y) in broadcasted_dims:
    if dim_x.value is None or dim_y.value is None:
      # One or both dimensions is unknown. If either dimension is greater than
      # 1, we assume that the program is correct, and the other dimension will
      # be broadcast to match it.
      # TODO(mrry): If we eliminate the shape checks in C++, we must still
      # assert that the unknown dim is either 1 or the same as the known dim.
      if dim_x.value is not None and dim_x.value > 1:
        return_dims.append(dim_x)
      elif dim_y.value is not None and dim_y.value > 1:
        return_dims.append(dim_y)
      else:
        return_dims.append(None)
    elif dim_x.value == 1:
      # We will broadcast dim_x to dim_y.
      return_dims.append(dim_y)
    elif dim_y.value == 1:
      # We will broadcast dim_y to dim_x.
      return_dims.append(dim_x)
    elif dim_x.value == dim_y.value:
      # The dimensions are compatible, so output is the same size in that
      # dimension.
      return_dims.append(dim_x.merge_with(dim_y))
    else:
      return None
  return return_dims


def is_broadcast_compatible(shape_x, shape_y):
  """Returns True if `shape_x` and `shape_y` are broadcast compatible.

  Args:
    shape_x: A `TensorShape`
    shape_y: A `TensorShape`

  Returns:
    True if a shape exists that both `shape_x` and `shape_y` can be broadcasted
    to.  False otherwise.
  """
  if shape_x.ndims is None or shape_y.ndims is None:
    return False
  return _broadcast_shape_helper(shape_x, shape_y) is not None


def broadcast_shape(shape_x, shape_y):
  """Returns the broadcasted shape between `shape_x` and `shape_y`.

  Args:
    shape_x: A `TensorShape`
    shape_y: A `TensorShape`

  Returns:
    A `TensorShape` representing the broadcasted shape.

  Raises:
    ValueError: If the two shapes can not be broadcasted.
  """
  if shape_x.ndims is None or shape_y.ndims is None:
    return tensor_shape.unknown_shape()
  return_dims = _broadcast_shape_helper(shape_x, shape_y)
  if return_dims is None:
    raise ValueError("Incompatible shapes for broadcasting: %s and %s"
                     % (shape_x, shape_y))
  return tensor_shape.TensorShape(return_dims)
