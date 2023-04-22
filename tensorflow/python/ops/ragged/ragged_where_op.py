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
"""where operation for RaggedTensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_shape


def where_v2(condition, x=None, y=None, name=None):
  """Return the elements where `condition` is `True`.

  : If both `x` and `y` are None: Retrieve indices of true elements.

    Returns the coordinates of true elements of `condition`. The coordinates
    are returned in a 2-D tensor with shape
    `[num_true_values, dim_size(condition)]`, where `result[i]` is the
    coordinates of the `i`th true value (in row-major order).

  : If both `x` and `y` are non-`None`: Multiplex between `x` and `y`.

    Choose an output shape  from the shapes of `condition`, `x`, and `y` that
    all three shapes are broadcastable to; and then use the broadcasted
    `condition` tensor as a mask that chooses whether the corredsponding element
    in the output should be taken from `x` (if `condition` is true) or `y` (if
    `condition` is false).

  >>> # Example: retrieve indices of true elements
  >>> tf.where(tf.ragged.constant([[True, False], [True]]))
  <tf.Tensor: shape=(2, 2), dtype=int64, numpy= array([[0, 0], [1, 0]])>

  >>> # Example: multiplex between `x` and `y`
  >>> tf.where(tf.ragged.constant([[True, False], [True, False, True]]),
  ...          tf.ragged.constant([['A', 'B'], ['C', 'D', 'E']]),
  ...          tf.ragged.constant([['a', 'b'], ['c', 'd', 'e']]))
  <tf.RaggedTensor [[b'A', b'b'], [b'C', b'd', b'E']]>

  Args:
    condition: A potentially ragged tensor of type `bool`
    x: A potentially ragged tensor (optional).
    y: A potentially ragged tensor (optional).  Must be specified if `x` is
      specified.  Must have the same rank and type as `x`.
    name: A name of the operation (optional).

  Returns:
    : If both `x` and `y` are `None`:
      A `Tensor` with shape `(num_true, rank(condition))`.
    : Otherwise:
      A potentially ragged tensor with the same type as `x` and `y`, and whose
      shape is broadcast-compatible with `x`, `y`, and `condition`.

  Raises:
    ValueError: When exactly one of `x` or `y` is non-`None`; or when
      `condition`, `x`, and `y` have incompatible shapes.
  """
  if (x is None) != (y is None):
    raise ValueError('x and y must be either both None or both non-None')

  with ops.name_scope('RaggedWhere', name, [condition, x, y]):
    condition = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        condition, name='condition')
    if x is None:
      return _coordinate_where(condition)
    else:
      x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, name='x')
      y = ragged_tensor.convert_to_tensor_or_ragged_tensor(y, name='y')
      condition, x, y = ragged_tensor.match_row_splits_dtypes(condition, x, y)
      return _elementwise_where_v2(condition, x, y)


def where(condition, x=None, y=None, name=None):
  """Return the elements, either from `x` or `y`, depending on the `condition`.

  : If both `x` and `y` are `None`:
    Returns the coordinates of true elements of `condition`. The coordinates
    are returned in a 2-D tensor with shape
    `[num_true_values, dim_size(condition)]`, where `result[i]` is the
    coordinates of the `i`th true value (in row-major order).

  : If both `x` and `y` are non-`None`:
    Returns a tensor formed by selecting values from `x` where condition is
    true, and from `y` when condition is false.  In particular:

    : If `condition`, `x`, and `y` all have the same shape:

      * `result[i1...iN] = x[i1...iN]` if `condition[i1...iN]` is true.
      * `result[i1...iN] = y[i1...iN]` if `condition[i1...iN]` is false.

    : Otherwise:

      * `condition` must be a vector.
      * `x` and `y` must have the same number of dimensions.
      * The outermost dimensions of `condition`, `x`, and `y` must all have the
        same size.
      * `result[i] = x[i]` if `condition[i]` is true.
      * `result[i] = y[i]` if `condition[i]` is false.

  Args:
    condition: A potentially ragged tensor of type `bool`
    x: A potentially ragged tensor (optional).
    y: A potentially ragged tensor (optional).  Must be specified if `x` is
      specified.  Must have the same rank and type as `x`.
    name: A name of the operation (optional)

  Returns:
    : If both `x` and `y` are `None`:
      A `Tensor` with shape `(num_true, dim_size(condition))`.
    : Otherwise:
      A potentially ragged tensor with the same type, rank, and outermost
      dimension size as `x` and `y`.
      `result.ragged_rank = max(x.ragged_rank, y.ragged_rank)`.

  Raises:
    ValueError: When exactly one of `x` or `y` is non-`None`; or when
      `condition`, `x`, and `y` have incompatible shapes.

  #### Examples:

  >>> # Coordinates where condition is true.
  >>> condition = tf.ragged.constant([[True, False, True], [False, True]])
  >>> print(where(condition))
  tf.Tensor( [[0 0] [0 2] [1 1]], shape=(3, 2), dtype=int64)

  >>> # Elementwise selection between x and y, based on condition.
  >>> condition = tf.ragged.constant([[True, False, True], [False, True]])
  >>> x = tf.ragged.constant([['A', 'B', 'C'], ['D', 'E']])
  >>> y = tf.ragged.constant([['a', 'b', 'c'], ['d', 'e']])
  >>> print(where(condition, x, y))
  <tf.RaggedTensor [[b'A', b'b', b'C'], [b'd', b'E']]>

  >>> # Row selection between x and y, based on condition.
  >>> condition = [True, False]
  >>> x = tf.ragged.constant([['A', 'B', 'C'], ['D', 'E']])
  >>> y = tf.ragged.constant([['a', 'b', 'c'], ['d', 'e']])
  >>> print(where(condition, x, y))
  <tf.RaggedTensor [[b'A', b'B', b'C'], [b'd', b'e']]>
  """
  if (x is None) != (y is None):
    raise ValueError('x and y must be either both None or both non-None')
  with ops.name_scope('RaggedWhere', name, [condition, x, y]):
    condition = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        condition, name='condition')
    if x is None:
      return _coordinate_where(condition)
    else:
      x = ragged_tensor.convert_to_tensor_or_ragged_tensor(x, name='x')
      y = ragged_tensor.convert_to_tensor_or_ragged_tensor(y, name='y')
      condition, x, y = ragged_tensor.match_row_splits_dtypes(condition, x, y)
      return _elementwise_where(condition, x, y)


def _elementwise_where(condition, x, y):
  """Ragged version of tf.where(condition, x, y)."""
  condition_is_ragged = isinstance(condition, ragged_tensor.RaggedTensor)
  x_is_ragged = isinstance(x, ragged_tensor.RaggedTensor)
  y_is_ragged = isinstance(y, ragged_tensor.RaggedTensor)

  if not (condition_is_ragged or x_is_ragged or y_is_ragged):
    return array_ops.where(condition, x, y)

  elif condition_is_ragged and x_is_ragged and y_is_ragged:
    return ragged_functional_ops.map_flat_values(array_ops.where, condition, x,
                                                 y)
  elif not condition_is_ragged:
    # Concatenate x and y, and then use `gather` to assemble the selected rows.
    condition.shape.assert_has_rank(1)
    x_and_y = ragged_concat_ops.concat([x, y], axis=0)
    x_nrows = _nrows(x, out_type=x_and_y.row_splits.dtype)
    y_nrows = _nrows(y, out_type=x_and_y.row_splits.dtype)
    indices = array_ops.where(condition, math_ops.range(x_nrows),
                              x_nrows + math_ops.range(y_nrows))
    return ragged_gather_ops.gather(x_and_y, indices)

  else:
    raise ValueError('Input shapes do not match.')


def _elementwise_where_v2(condition, x, y):
  """Ragged version of tf.where_v2(condition, x, y)."""
  # Broadcast x, y, and condition to have the same shape.
  if not (condition.shape.is_fully_defined() and x.shape.is_fully_defined() and
          y.shape.is_fully_defined() and x.shape == y.shape and
          condition.shape == x.shape):
    shape_c = ragged_tensor_shape.RaggedTensorDynamicShape.from_tensor(
        condition)
    shape_x = ragged_tensor_shape.RaggedTensorDynamicShape.from_tensor(x)
    shape_y = ragged_tensor_shape.RaggedTensorDynamicShape.from_tensor(y)
    shape = ragged_tensor_shape.broadcast_dynamic_shape(
        shape_c, ragged_tensor_shape.broadcast_dynamic_shape(shape_x, shape_y))
    condition = ragged_tensor_shape.broadcast_to(condition, shape)
    x = ragged_tensor_shape.broadcast_to(x, shape)
    y = ragged_tensor_shape.broadcast_to(y, shape)

  condition_is_ragged = isinstance(condition, ragged_tensor.RaggedTensor)
  x_is_ragged = isinstance(x, ragged_tensor.RaggedTensor)
  y_is_ragged = isinstance(y, ragged_tensor.RaggedTensor)
  if not (condition_is_ragged or x_is_ragged or y_is_ragged):
    return array_ops.where_v2(condition, x, y)

  return ragged_functional_ops.map_flat_values(array_ops.where_v2, condition, x,
                                               y)


def _coordinate_where(condition):
  """Ragged version of tf.where(condition)."""
  if not isinstance(condition, ragged_tensor.RaggedTensor):
    return array_ops.where(condition)

  # The coordinate for each `true` value in condition.values.
  selected_coords = _coordinate_where(condition.values)

  # Convert the first index in each coordinate to a row index and column index.
  condition = condition.with_row_splits_dtype(selected_coords.dtype)
  first_index = selected_coords[:, 0]
  selected_rows = array_ops.gather(condition.value_rowids(), first_index)
  selected_row_starts = array_ops.gather(condition.row_splits, selected_rows)
  selected_cols = first_index - selected_row_starts

  # Assemble the row & column index with the indices for inner dimensions.
  return array_ops.concat([
      array_ops.expand_dims(selected_rows, 1),
      array_ops.expand_dims(selected_cols, 1), selected_coords[:, 1:]
  ],
                          axis=1)


def _nrows(rt_input, out_type):
  if isinstance(rt_input, ragged_tensor.RaggedTensor):
    return rt_input.nrows(out_type=out_type)
  else:
    return array_ops.shape(rt_input, out_type=out_type)[0]
