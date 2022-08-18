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
"""Utilities for computing default gradients."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops


def get_zeros_dtype(t):
  """Return the dtype for the default gradient for a Tensor."""
  if t.dtype == dtypes.resource:
    handle_data = resource_variable_ops.get_eager_safe_handle_data(t)
    if (handle_data is None or not handle_data.is_set or
        len(handle_data.shape_and_type) != 1):
      raise ValueError("Internal error: Tried to take gradients (or similar) "
                       "of a variable without handle data:\n%s" % str(t))
    return handle_data.shape_and_type[0].dtype
  return t.dtype


def shape_and_dtype(t):
  """Return the shape and dtype for the default gradient for a Tensor."""
  if t.dtype == dtypes.resource:
    handle_data = resource_variable_ops.get_eager_safe_handle_data(t)
    if (handle_data is None or not handle_data.is_set or
        len(handle_data.shape_and_type) != 1):
      raise ValueError("Internal error: Tried to take gradients (or similar) "
                       "of a variable without handle data:\n%s" % str(t))
    shape_and_type = handle_data.shape_and_type[0]
    return (tensor_shape.TensorShape(shape_and_type.shape),
            dtypes.as_dtype(shape_and_type.dtype))
  return t.shape, t.dtype


def zeros_like(t):
  """Like array_ops.zeros_like, but respects resource handles."""
  if t.dtype == dtypes.resource:
    return array_ops.zeros(*shape_and_dtype(t))
  else:
    return array_ops.zeros_like(t)


def ones_like(t):
  """Like array_ops.ones_like, but respects resource handles."""
  if t.dtype == dtypes.resource:
    return array_ops.ones(*shape_and_dtype(t))
  else:
    return array_ops.ones_like(t)


def supports_default_grad(t):
  """Whether tensor `t` supports creating a default gradient.

  This function assumes that `t` is of a trainable type.

  Args:
    t: Tensor

  Returns:
    Bool
  """
  if t.dtype == dtypes.resource:
    handle_data = resource_variable_ops.get_eager_safe_handle_data(t)
    if (handle_data is None or not handle_data.is_set or
        len(handle_data.shape_and_type) != 1):
      return False
  return True
