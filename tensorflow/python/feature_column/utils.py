# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Defines functions common to multiple feature column files."""

import six

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest


def sequence_length_from_sparse_tensor(sp_tensor, num_elements=1):
  """Returns a [batch_size] Tensor with per-example sequence length."""
  with ops.name_scope(None, 'sequence_length') as name_scope:
    row_ids = sp_tensor.indices[:, 0]
    column_ids = sp_tensor.indices[:, 1]
    # Add one to convert column indices to element length
    column_ids += array_ops.ones_like(column_ids)
    # Get the number of elements we will have per example/row
    seq_length = math_ops.segment_max(column_ids, segment_ids=row_ids)

    # The raw values are grouped according to num_elements;
    # how many entities will we have after grouping?
    # Example: orig tensor [[1, 2], [3]], col_ids = (0, 1, 1),
    # row_ids = (0, 0, 1), seq_length = [2, 1]. If num_elements = 2,
    # these will get grouped, and the final seq_length is [1, 1]
    seq_length = math_ops.cast(
        math_ops.ceil(seq_length / num_elements), dtypes.int64)

    # If the last n rows do not have ids, seq_length will have shape
    # [batch_size - n]. Pad the remaining values with zeros.
    n_pad = array_ops.shape(sp_tensor)[:1] - array_ops.shape(seq_length)[:1]
    padding = array_ops.zeros(n_pad, dtype=seq_length.dtype)
    return array_ops.concat([seq_length, padding], axis=0, name=name_scope)


def assert_string_or_int(dtype, prefix):
  if (dtype != dtypes.string) and (not dtype.is_integer):
    raise ValueError(
        '{} dtype must be string or integer. dtype: {}.'.format(prefix, dtype))


def assert_key_is_string(key):
  if not isinstance(key, six.string_types):
    raise ValueError(
        'key must be a string. Got: type {}. Given key: {}.'.format(
            type(key), key))


def check_default_value(shape, default_value, dtype, key):
  """Returns default value as tuple if it's valid, otherwise raises errors.

  This function verifies that `default_value` is compatible with both `shape`
  and `dtype`. If it is not compatible, it raises an error. If it is compatible,
  it casts default_value to a tuple and returns it. `key` is used only
  for error message.

  Args:
    shape: An iterable of integers specifies the shape of the `Tensor`.
    default_value: If a single value is provided, the same value will be applied
      as the default value for every item. If an iterable of values is
      provided, the shape of the `default_value` should be equal to the given
      `shape`.
    dtype: defines the type of values. Default value is `tf.float32`. Must be a
      non-quantized, real integer or floating point type.
    key: Column name, used only for error messages.

  Returns:
    A tuple which will be used as default value.

  Raises:
    TypeError: if `default_value` is an iterable but not compatible with `shape`
    TypeError: if `default_value` is not compatible with `dtype`.
    ValueError: if `dtype` is not convertible to `tf.float32`.
  """
  if default_value is None:
    return None

  if isinstance(default_value, int):
    return _create_tuple(shape, default_value)

  if isinstance(default_value, float) and dtype.is_floating:
    return _create_tuple(shape, default_value)

  if callable(getattr(default_value, 'tolist', None)):  # Handles numpy arrays
    default_value = default_value.tolist()

  if nest.is_sequence(default_value):
    if not _is_shape_and_default_value_compatible(default_value, shape):
      raise ValueError(
          'The shape of default_value must be equal to given shape. '
          'default_value: {}, shape: {}, key: {}'.format(
              default_value, shape, key))
    # Check if the values in the list are all integers or are convertible to
    # floats.
    is_list_all_int = all(
        isinstance(v, int) for v in nest.flatten(default_value))
    is_list_has_float = any(
        isinstance(v, float) for v in nest.flatten(default_value))
    if is_list_all_int:
      return _as_tuple(default_value)
    if is_list_has_float and dtype.is_floating:
      return _as_tuple(default_value)
  raise TypeError('default_value must be compatible with dtype. '
                  'default_value: {}, dtype: {}, key: {}'.format(
                      default_value, dtype, key))


def _create_tuple(shape, value):
  """Returns a tuple with given shape and filled with value."""
  if shape:
    return tuple([_create_tuple(shape[1:], value) for _ in range(shape[0])])
  return value


def _as_tuple(value):
  if not nest.is_sequence(value):
    return value
  return tuple([_as_tuple(v) for v in value])


def _is_shape_and_default_value_compatible(default_value, shape):
  """Verifies compatibility of shape and default_value."""
  # Invalid condition:
  #  * if default_value is not a scalar and shape is empty
  #  * or if default_value is an iterable and shape is not empty
  if nest.is_sequence(default_value) != bool(shape):
    return False
  if not shape:
    return True
  if len(default_value) != shape[0]:
    return False
  for i in range(shape[0]):
    if not _is_shape_and_default_value_compatible(default_value[i], shape[1:]):
      return False
  return True
