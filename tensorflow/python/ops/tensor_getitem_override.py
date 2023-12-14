# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# Tests for this file live in python/kernel_tests/array_ops_test.py
"""Tensor __getitem__ override logic."""

import numbers
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


# We override the 'slice' for the "slice" op, so we keep Python's
# existing 'slice' for later use in this module.
_BaseSlice = slice


_SLICE_TYPE_ERROR = (
    "Only integers, slices (`:`), ellipsis (`...`), "
    "tf.newaxis (`None`) and scalar tf.int32/tf.int64 tensors are valid "
    "indices")


_SUPPORTED_SLICE_DTYPES = (dtypes.int16, dtypes.int32, dtypes.int32_ref,
                           dtypes.int64, dtypes.int64_ref)


def _is_undefined_dimension(d):
  return isinstance(d, tensor_shape.Dimension) and d.value is None


def _check_index(idx):
  """Check if a given value is a valid index into a tensor."""
  if isinstance(idx, (numbers.Integral, tensor_shape.Dimension)):
    return

  # Optimistic check. Assumptions:
  # * any object with a dtype is supported
  # * any object with a dtype has a sizeable shape attribute.
  dtype = getattr(idx, "dtype", None)
  if (dtype is None or dtypes.as_dtype(dtype) not in _SUPPORTED_SLICE_DTYPES or
      idx.shape and len(idx.shape) == 1):
    # TODO(slebedev): IndexError seems more appropriate here, but it
    # will break `_slice_helper` contract.
    raise TypeError(_SLICE_TYPE_ERROR + ", got {!r}".format(idx))


@tf_export("__operators__.getitem", v1=[])
@dispatch.add_dispatch_support
def _slice_helper(tensor, slice_spec, var=None):
  """Overload for Tensor.__getitem__.

  This operation extracts the specified region from the tensor.
  The notation is similar to NumPy with the restriction that
  currently only support basic indexing. That means that
  using a non-scalar tensor as input is not currently allowed.

  Some useful examples:

  ```python
  # Strip leading and trailing 2 elements
  foo = tf.constant([1,2,3,4,5,6])
  print(foo[2:-2])  # => [3,4]

  # Skip every other row and reverse the order of the columns
  foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
  print(foo[::2,::-1])  # => [[3,2,1], [9,8,7]]

  # Use scalar tensors as indices on both dimensions
  print(foo[tf.constant(0), tf.constant(2)])  # => 3

  # Insert another dimension
  foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
  print(foo[tf.newaxis, :, :]) # => [[[1,2,3], [4,5,6], [7,8,9]]]
  print(foo[:, tf.newaxis, :]) # => [[[1,2,3]], [[4,5,6]], [[7,8,9]]]
  print(foo[:, :, tf.newaxis]) # => [[[1],[2],[3]], [[4],[5],[6]],
  [[7],[8],[9]]]

  # Ellipses (3 equivalent operations)
  foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
  print(foo[tf.newaxis, :, :])  # => [[[1,2,3], [4,5,6], [7,8,9]]]
  print(foo[tf.newaxis, ...])  # => [[[1,2,3], [4,5,6], [7,8,9]]]
  print(foo[tf.newaxis])  # => [[[1,2,3], [4,5,6], [7,8,9]]]

  # Masks
  foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
  print(foo[foo > 2])  # => [3, 4, 5, 6, 7, 8, 9]
  ```

  Notes:
    - `tf.newaxis` is `None` as in NumPy.
    - An implicit ellipsis is placed at the end of the `slice_spec`
    - NumPy advanced indexing is currently not supported.

  Purpose in the API:

    This method is exposed in TensorFlow's API so that library developers
    can register dispatching for `Tensor.__getitem__` to allow it to handle
    custom composite tensors & other custom objects.

    The API symbol is not intended to be called by users directly and does
    appear in TensorFlow's generated documentation.

  Args:
    tensor: An tensor.Tensor object.
    slice_spec: The arguments to Tensor.__getitem__.
    var: In the case of variable slice assignment, the Variable object to slice
      (i.e. tensor is the read-only view of this variable).

  Returns:
    The appropriate slice of "tensor", based on "slice_spec".

  Raises:
    ValueError: If a slice range is negative size.
    TypeError: If the slice indices aren't int, slice, ellipsis,
      tf.newaxis or scalar int32/int64 tensors.
  """
  from tensorflow.python.framework import constant_op  # pylint: disable=g-import-not-at-top
  from tensorflow.python.ops import array_ops  # pylint: disable=g-import-not-at-top
  tensor = ops.convert_to_tensor(tensor)
  # TODO(wangpeng): Consider supporting var
  if var is None and ops._numpy_style_slicing:  # pylint: disable=protected-access
    return tensor._numpy_style_getitem(slice_spec)  # pylint: disable=protected-access

  if (isinstance(slice_spec, bool)
      or (isinstance(slice_spec, tensor_lib.Tensor)
          and slice_spec.dtype == dtypes.bool)
      or (isinstance(slice_spec, np.ndarray)
          and slice_spec.dtype == bool)):
    return array_ops.boolean_mask(tensor=tensor, mask=slice_spec)

  if not isinstance(slice_spec, (list, tuple)):
    slice_spec = [slice_spec]

  begin, end, strides = [], [], []
  index = 0

  new_axis_mask, shrink_axis_mask = 0, 0
  begin_mask, end_mask = 0, 0
  ellipsis_mask = 0
  for s in slice_spec:
    if isinstance(s, _BaseSlice):
      # Finds the best dtype for begin, end, and strides.
      dtype = None
      for t in [s.start, s.stop, s.step]:
        if t is None or not isinstance(t, tensor_lib.Tensor):
          continue
        if t.dtype == dtypes.int64:
          dtype = dtypes.int64
        elif t.dtype == dtypes.int32 and dtype != dtypes.int64:
          dtype = dtypes.int32
        elif t.dtype == dtypes.int16 and dtype is None:
          dtype = dtypes.int16

      if s.start is not None and not _is_undefined_dimension(s.start):
        _check_index(s.start)
        begin.append(s.start)
      else:
        if dtype is not None:
          begin.append(constant_op.constant(0, dtype=dtype))
        else:
          begin.append(0)
        begin_mask |= (1 << index)
      if s.stop is not None and not _is_undefined_dimension(s.stop):
        _check_index(s.stop)
        end.append(s.stop)
      else:
        if dtype is not None:
          end.append(constant_op.constant(0, dtype=dtype))
        else:
          end.append(0)
        end_mask |= (1 << index)
      if s.step is not None and not _is_undefined_dimension(s.step):
        _check_index(s.step)
        strides.append(s.step)
      else:
        if dtype is not None:
          strides.append(constant_op.constant(1, dtype=dtype))
        else:
          strides.append(1)
    elif s is Ellipsis:
      begin.append(0)
      end.append(0)
      strides.append(1)
      ellipsis_mask |= (1 << index)
    elif s is array_ops.newaxis:
      begin.append(0)
      end.append(0)
      strides.append(1)
      new_axis_mask |= (1 << index)
    else:
      _check_index(s)
      begin.append(s)
      end.append(s + 1)
      # TODO(mdan): Investigate why we can't set int32 here.
      if (
          isinstance(s, tensor_lib.Tensor)
          and (s.dtype == dtypes.int16 or s.dtype == dtypes.int64)):
        strides.append(constant_op.constant(1, dtype=s.dtype))
      else:
        strides.append(1)
      shrink_axis_mask |= (1 << index)
    index += 1

  # stack possibly involves no tensors, so we must use op_scope correct graph.
  with ops.name_scope(
      None,
      "strided_slice", [tensor] + begin + end + strides,
      skip_on_eager=False) as name:
    if begin:
      from tensorflow.python.ops import array_ops_stack  # pylint: disable=g-import-not-at-top
      packed_begin, packed_end, packed_strides = (
          array_ops_stack.stack(begin),
          array_ops_stack.stack(end),
          array_ops_stack.stack(strides))
      # TODO(mdan): Instead of implicitly casting, it's better to enforce the
      # same dtypes.
      if (packed_begin.dtype == dtypes.int64 or
          packed_end.dtype == dtypes.int64 or
          packed_strides.dtype == dtypes.int64):
        if packed_begin.dtype != dtypes.int64:
          packed_begin = gen_math_ops.cast(packed_begin, dtypes.int64)
        if packed_end.dtype != dtypes.int64:
          packed_end = gen_math_ops.cast(packed_end, dtypes.int64)
        if packed_strides.dtype != dtypes.int64:
          packed_strides = gen_math_ops.cast(packed_strides, dtypes.int64)
      elif (packed_begin.dtype == dtypes.int16 and
            packed_end.dtype == dtypes.int16 and
            packed_strides.dtype == dtypes.int16):
        if packed_begin.dtype != dtypes.int16:
          packed_begin = gen_math_ops.cast(packed_begin, dtypes.int16)
        if packed_end.dtype != dtypes.int16:
          packed_end = gen_math_ops.cast(packed_end, dtypes.int16)
        if packed_strides.dtype != dtypes.int16:
          packed_strides = gen_math_ops.cast(packed_strides, dtypes.int16)
    else:
      var_empty = constant_op.constant([], dtype=dtypes.int32)
      packed_begin = packed_end = packed_strides = var_empty
    return array_ops.strided_slice(
        tensor,
        packed_begin,
        packed_end,
        packed_strides,
        begin_mask=begin_mask,
        end_mask=end_mask,
        shrink_axis_mask=shrink_axis_mask,
        new_axis_mask=new_axis_mask,
        ellipsis_mask=ellipsis_mask,
        var=var,
        name=name)


def _slice_helper_var(var, slice_spec):
  """Creates a slice helper object given a variable.

  This allows creating a sub-tensor from part of the current contents
  of a variable. See `tf.Tensor.__getitem__` for detailed examples
  of slicing.

  This function in addition also allows assignment to a sliced range.
  This is similar to `__setitem__` functionality in Python. However,
  the syntax is different so that the user can capture the assignment
  operation for grouping or passing to `sess.run()` in TF1.
  For example,

  ```python
  import tensorflow as tf
  A = tf.Variable([[1,2,3], [4,5,6], [7,8,9]], dtype=tf.float32)
  print(A[:2, :2])  # => [[1,2], [4,5]]

  A[:2,:2].assign(22. * tf.ones((2, 2))))
  print(A) # => [[22, 22, 3], [22, 22, 6], [7,8,9]]
  ```

  Note that assignments currently do not support NumPy broadcasting
  semantics.

  Args:
    var: An `ops.Variable` object.
    slice_spec: The arguments to `Tensor.__getitem__`.

  Returns:
    The appropriate slice of "tensor", based on "slice_spec".
    As an operator. The operator also has a `assign()` method
    that can be used to generate an assignment operator.

  Raises:
    ValueError: If a slice range is negative size.
    TypeError: TypeError: If the slice indices aren't int, slice,
      ellipsis, tf.newaxis or int32/int64 tensors.

  """

  return _slice_helper(var.value(), slice_spec, var)


tensor_lib.Tensor._override_operator("__getitem__", _slice_helper)  # pylint: disable=protected-access
