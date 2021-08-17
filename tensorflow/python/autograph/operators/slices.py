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
"""Operators specific to slicing operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops


# TODO(mdan): Support extended slices.


class GetItemOpts(collections.namedtuple('GetItemOpts', ('element_dtype',))):
  pass


def get_item(target, i, opts):
  """The slice read operator (i.e. __getitem__).

  Note: it is unspecified whether target will be mutated or not. In general,
  if target is mutable (like Python lists), it will be mutated.

  Args:
    target: An entity that supports getitem semantics.
    i: Index to read from.
    opts: A GetItemOpts object.

  Returns:
    The read element.

  Raises:
    ValueError: if target is not of a supported type.
  """
  assert isinstance(opts, GetItemOpts)

  if isinstance(target, tensor_array_ops.TensorArray):
    return _tf_tensorarray_get_item(target, i)
  elif tensor_util.is_tf_type(target):
    if target.dtype == dtypes.variant:
      return _tf_tensor_list_get_item(target, i, opts)
    elif target.dtype == dtypes.string and target.shape.ndims == 0:
      return _tf_tensor_string_get_item(target, i)
    else:
      return _tf_tensor_get_item(target, i)
  else:
    return _py_get_item(target, i)


def _tf_tensorarray_get_item(target, i):
  """Overload of get_item that stages a TensorArray read."""
  return target.read(i)


def _tf_tensor_list_get_item(target, i, opts):
  """Overload of get_item that stages a Tensor list read."""
  if opts.element_dtype is None:
    raise ValueError('cannot retrieve from a list without knowing its '
                     'element type; use set_element_type to annotate it')
  x = list_ops.tensor_list_get_item(target, i, element_dtype=opts.element_dtype)
  return x


def _tf_tensor_get_item(target, i):
  """Overload of get_item that stages a Tensor (not Tensor list) read."""
  return target[i]


def _tf_tensor_string_get_item(target, i):
  """Overload of get_item that stages a Tensor string read."""
  x = gen_string_ops.substr(target, i, 1)
  return x


def _py_get_item(target, i):
  """Overload of get_item that executes a Python list modification."""
  return target[i]


def set_item(target, i, x):
  """The slice write operator (i.e. __setitem__).

  Note: it is unspecified whether target will be mutated or not. In general,
  if target is mutable (like Python lists), it will be mutated.

  Args:
    target: An entity that supports setitem semantics.
    i: Index to modify.
    x: The new element value.

  Returns:
    Same as target, after the update was performed.

  Raises:
    ValueError: if target is not of a supported type.
  """
  if isinstance(target, tensor_array_ops.TensorArray):
    return _tf_tensorarray_set_item(target, i, x)
  elif tensor_util.is_tf_type(target):
    if target.dtype == dtypes.variant:
      return _tf_tensor_list_set_item(target, i, x)
    else:
      return _tf_tensor_set_item(target, i, x)
  else:
    return _py_set_item(target, i, x)


def _tf_tensorarray_set_item(target, i, x):
  """Overload of set_item that stages a TensorArray write."""
  return target.write(i, x)


def _tf_tensor_list_set_item(target, i, x):
  """Overload of set_item that stages a Tensor list update."""
  return list_ops.tensor_list_set_item(target, i, x)


def _tf_tensor_set_item(target, i, x):
  """Overload of set_item that stages a Tensor scatter update."""
  return gen_array_ops.tensor_scatter_update(target, ((i,),), (x,))


def _py_set_item(target, i, x):
  """Overload of set_item that executes a Python list modification."""
  target[i] = x
  return target
