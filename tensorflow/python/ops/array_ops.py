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

"""## Casting

TensorFlow provides several operations that you can use to cast tensor data
types in your graph.

@@string_to_number
@@to_double
@@to_float
@@to_bfloat16
@@to_int32
@@to_int64
@@cast
@@bitcast
@@saturate_cast

## Shapes and Shaping

TensorFlow provides several operations that you can use to determine the shape
of a tensor and change the shape of a tensor.

@@shape
@@shape_n
@@size
@@rank
@@reshape
@@squeeze
@@expand_dims
@@meshgrid

## Slicing and Joining

TensorFlow provides several operations to slice or extract parts of a tensor,
or join multiple tensors together.

@@slice
@@strided_slice
@@split
@@tile
@@pad
@@concat
@@pack
@@unpack
@@reverse_sequence
@@reverse
@@transpose
@@extract_image_patches
@@space_to_batch_nd
@@space_to_batch
@@required_space_to_batch_paddings
@@batch_to_space_nd
@@batch_to_space
@@space_to_depth
@@depth_to_space
@@gather
@@gather_nd
@@unique_with_counts
@@dynamic_partition
@@dynamic_stitch
@@boolean_mask
@@one_hot
@@sequence_mask

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
# 'Constant' gets imported in the module 'array_ops'.
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import gen_math_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_array_ops import *
# pylint: enable=wildcard-import


# Used for slicing to specify a new 1 size dimension
newaxis = None

# We override the 'slice' for the "slice" op, so we keep python's
# existing 'slice' for later use in this module.
_baseslice = slice


# Aliases for some automatically-generated names.
listdiff = gen_array_ops.list_diff


def shape(input, name=None, out_type=dtypes.int32):
  # pylint: disable=redefined-builtin
  """Returns the shape of a tensor.

  This operation returns a 1-D integer tensor representing the shape of `input`.

  For example:

  ```python
  # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
  shape(t) ==> [2, 2, 3]
  ```

  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).
    out_type: (Optional) The specified output type of the operation
      (`int32` or `int64`). Defaults to tf.int32.

  Returns:
    A `Tensor` of type `out_type`.
  """
  return shape_internal(input, name, optimize=True, out_type=out_type)


def shape_internal(input, name=None, optimize=True, out_type=dtypes.int32):
  # pylint: disable=redefined-builtin
  """Returns the shape of a tensor.

  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).
    optimize: if true, encode the shape as a constant when possible.
    out_type: (Optional) The specified output type of the operation
      (`int32` or `int64`). Defaults to tf.int32.

  Returns:
    A `Tensor` of type `out_type`.

  """
  with ops.name_scope(name, "Shape", [input]) as name:
    if isinstance(input, (ops.SparseTensor, ops.SparseTensorValue)):
      return gen_math_ops.cast(input.shape, out_type)
    else:
      input_tensor = ops.convert_to_tensor(input)
      input_shape = input_tensor.get_shape()
      if optimize and input_shape.is_fully_defined():
        return constant(input_shape.as_list(), out_type, name=name)
      return gen_array_ops.shape(input, name=name, out_type=out_type)


def size(input, name=None, out_type=dtypes.int32):
  # pylint: disable=redefined-builtin
  """Returns the size of a tensor.

  This operation returns an integer representing the number of elements in
  `input`.

  For example:

  ```python
  # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
  size(t) ==> 12
  ```

  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).
    out_type: (Optional) The specified output type of the operation
      (`int32` or `int64`). Defaults to tf.int32.

  Returns:
    A `Tensor` of type `out_type`. Defaults to tf.int32.
  """
  return size_internal(input, name, optimize=True, out_type=out_type)


def size_internal(input, name=None, optimize=True, out_type=dtypes.int32):
  # pylint: disable=redefined-builtin,protected-access
  """Returns the size of a tensor.

  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).
    optimize: if true, encode the size as a constant when possible.
    out_type: (Optional) The specified output type of the operation
      (`int32` or `int64`). Defaults to tf.int32.

  Returns:
    A `Tensor` of type `out_type`.
  """
  with ops.name_scope(name, "Size", [input]) as name:
    if isinstance(input, (ops.SparseTensor, ops.SparseTensorValue)):
      return gen_math_ops._prod(
          gen_math_ops.cast(input.shape, out_type), 0, name=name)
    else:
      input_tensor = ops.convert_to_tensor(input)
      input_shape = input_tensor.get_shape()
      if optimize and input_shape.is_fully_defined():
        return constant(input_shape.num_elements(), out_type, name=name)
      return gen_array_ops.size(input, name=name, out_type=out_type)


def rank(input, name=None):
  # pylint: disable=redefined-builtin
  """Returns the rank of a tensor.

  This operation returns an integer representing the rank of `input`.

  For example:

  ```python
  # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
  # shape of tensor 't' is [2, 2, 3]
  rank(t) ==> 3
  ```

  **Note**: The rank of a tensor is not the same as the rank of a matrix. The
  rank of a tensor is the number of indices required to uniquely select each
  element of the tensor. Rank is also known as "order", "degree", or "ndims."

  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  return rank_internal(input, name, optimize=True)


def rank_internal(input, name=None, optimize=True):
  # pylint: disable=redefined-builtin
  """Returns the rank of a tensor.

  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).
    optimize: if true, encode the rank as a constant when possible.

  Returns:
    A `Tensor` of type `int32`.
  """
  with ops.name_scope(name, "Rank", [input]) as name:
    if isinstance(input, (ops.SparseTensor, ops.SparseTensorValue)):
      return gen_array_ops.size(input.shape, name=name)
    else:
      input_tensor = ops.convert_to_tensor(input)
      input_shape = input_tensor.get_shape()
      if optimize and input_shape.ndims is not None:
        return constant(input_shape.ndims, dtypes.int32, name=name)
      return gen_array_ops.rank(input, name=name)


def _SliceHelper(tensor, slice_spec, var=None):
  """Overload for Tensor.__getitem__.

  This operation extracts the specified region from the tensor.
  The notation is similar to NumPy with the restriction that
  currently only support basic indexing. That means that
  using a tensor as input is not currently allowed

  Some useful examples:

  ```
  # strip leading and trailing 2 elements
  foo = tf.constant([1,2,3,4,5,6])
  print(foo[2:-2].eval()) # => [3,4]

  # skip every row and reverse every column
  foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
  print(foo[::2,::-1].eval()) # => [[3,2,1], [9,8,7]]

  # Insert another dimension
  foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
  print(foo[tf.newaxis, :, :].eval()) # => [[[3,2,1], [9,8,7]]]
  print(foo[:, tf.newaxis, :].eval()) # => [[[3,2,1]], [[9,8,7]]]
  print(foo[:, :, tf.newaxis].eval()) # => [[[3],[2],[1]], [[9],[8],[7]]]

  # Ellipses (3 equivalent operations)
  print(foo[tf.newaxis, :, :].eval()) # => [[[3,2,1], [9,8,7]]]
  print(foo[tf.newaxis, ...].eval()) # => [[[3,2,1], [9,8,7]]]
  print(foo[tf.newaxis].eval()) # => [[[3,2,1], [9,8,7]]]
  ```

  Notes:
    - `tf.newaxis` is `None` as in NumPy.
    - An implicit ellipsis is placed at the end of the `slice_spec`
    - NumPy advanced indexing is currently not supported.

  Args:
    tensor: An ops.Tensor object.
    slice_spec: The arguments to Tensor.__getitem__.
    var: In the case of variable slice assignment, the Variable
      object to slice (i.e. tensor is the read-only view of this
      variable).

  Returns:
    The appropriate slice of "tensor", based on "slice_spec".

  Raises:
    ValueError: If a slice range is negative size.
    TypeError: If the slice indices aren't int, slice, or Ellipsis.
  """

  if not isinstance(slice_spec, (list, tuple)):
    slice_spec = [slice_spec]

  begin, end, strides = [], [], []
  index = 0

  new_axis_mask, shrink_axis_mask = 0, 0
  begin_mask, end_mask = 0, 0
  ellipsis_mask = 0
  for s in slice_spec:
    if isinstance(s, _baseslice):
      strides.append(s.step if s.step is not None else 1)
      # python doesn't always use None when constructing ranges
      # for example a[:] gives slice(None,sys.maxsize,None)
      # whereas a[::1] gives slice(None,None,None)
      if s.start is not None and s.start is not sys.maxsize:
        begin.append(s.start)
      else:
        begin.append(0)
        begin_mask |= (1 << index)
      if s.stop is not None and s.stop != sys.maxsize:
        end.append(s.stop)
      else:
        end.append(0)
        end_mask |= (1 << index)
    elif s is Ellipsis:
      begin.append(0)
      end.append(0)
      strides.append(1)
      ellipsis_mask |= (1 << index)
    elif s is newaxis:
      begin.append(0)
      end.append(0)
      strides.append(1)
      new_axis_mask |= (1 << index)
    else:
      begin.append(s)
      end.append(s + 1)
      strides.append(1)
      shrink_axis_mask |= (1 << index)
    index += 1

  # pack possibly involves often involves no tensors, so we must use op_scope
  # correct graph
  with ops.name_scope(None, "strided_slice",
                      [tensor] + begin + end + strides) as name:
    if begin:
      packed_begin, packed_end, packed_strides = pack(begin), pack(end), pack(
          strides)
    else:
      empty = constant([], dtype=dtypes.int32)
      packed_begin = packed_end = packed_strides = empty
    return strided_slice(
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


# pylint: disable=undefined-variable,protected-access
def slice(input_, begin, size, name=None):
  # pylint: disable=redefined-builtin
  """Extracts a slice from a tensor.

  This operation extracts a slice of size `size` from a tensor `input` starting
  at the location specified by `begin`. The slice `size` is represented as a
  tensor shape, where `size[i]` is the number of elements of the 'i'th dimension
  of `input` that you want to slice. The starting location (`begin`) for the
  slice is represented as an offset in each dimension of `input`. In other
  words, `begin[i]` is the offset into the 'i'th dimension of `input` that you
  want to slice from.

  `begin` is zero-based; `size` is one-based. If `size[i]` is -1,
  all remaining elements in dimension i are included in the
  slice. In other words, this is equivalent to setting:

  `size[i] = input.dim_size(i) - begin[i]`

  This operation requires that:

  `0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n]`

  For example:

  ```
  # 'input' is [[[1, 1, 1], [2, 2, 2]],
  #             [[3, 3, 3], [4, 4, 4]],
  #             [[5, 5, 5], [6, 6, 6]]]
  tf.slice(input, [1, 0, 0], [1, 1, 3]) ==> [[[3, 3, 3]]]
  tf.slice(input, [1, 0, 0], [1, 2, 3]) ==> [[[3, 3, 3],
                                              [4, 4, 4]]]
  tf.slice(input, [1, 0, 0], [2, 1, 3]) ==> [[[3, 3, 3]],
                                             [[5, 5, 5]]]
  ```

  Args:
    input_: A `Tensor`.
    begin: An `int32` or `int64` `Tensor`.
    size: An `int32` or `int64` `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` the same type as `input`.
  """
  return gen_array_ops._slice(input_, begin, size, name=name)


# pylint: disable=invalid-name
def strided_slice(input_,
                  begin,
                  end,
                  strides,
                  begin_mask=0,
                  end_mask=0,
                  ellipsis_mask=0,
                  new_axis_mask=0,
                  shrink_axis_mask=0,
                  var=None,
                  name=None):
  """Extracts a strided slice from a tensor.

  To a first order, this operation extracts a slice of size `end - begin`
  from a tensor `input`
  starting at the location specified by `begin`. The slice continues by adding
  `stride` to the `begin` index until all dimensions are not less than `end`.
  Note that components of stride can be negative, which causes a reverse
  slice.

  This operation can be thought of an encoding of a numpy style sliced
  range. Given a python slice input[<spec0>, <spec1>, ..., <specn>]
  this function will be called as follows.

  `begin`, `end`, and `strides` will be all length n. n is in general
  not the same dimensionality as `input`.

  For the ith spec,
  `begin_mask`, `end_mask`, `ellipsis_mask`, `new_axis_mask`,
  and `shrink_axis_mask` will have the ith bit corresponding to
  the ith spec.

  If the ith bit of `begin_mask` is non-zero, `begin[i]` is ignored and
  the fullest possible range in that dimension is used instead.
  `end_mask` works analogously, except with the end range.

  `foo[5:,:,:3]` on a 7x8x9 tensor is equivalent to `foo[5:7,0:8,0:3]`.
  `foo[::-1]` reverses a tensor with shape 8.


  If the ith bit of `ellipsis_mask`, as many unspecified dimensions
  as needed will be inserted between other dimensions. Only one
  non-zero bit is allowed in `ellipsis_mask`.

  For example `foo[3:5,...,4:5]` on a shape 10x3x3x10 tensor is
  equivalent to `foo[3:5,:,:,4:5]` and
  `foo[3:5,...]` is equivalent to `foo[3:5,:,:,:]`.

  If the ith bit of `new_axis_mask` is one, then a `begin`,
  `end`, and `stride` are ignored and a new length 1 dimension is
  added at this point in the output tensor.

  For example `foo[3:5,4]` on a 10x8 tensor produces a shape 2 tensor
  whereas `foo[3:5,4:5]` produces a shape 2x1 tensor with shrink_mask
  being 1<<1 == 2.

  If the ith bit of `shrink_axis_mask` is one, then `begin`,
  `end[i]`, and `stride[i]` are used to do a slice in the appropriate
  dimension, but the output tensor will be reduced in dimensionality
  by one. This is only valid if the ith entry of slice[i]==1.

  NOTE: `begin` and `end` are zero-indexed`.
  `strides` entries must be non-zero.


  ```
  # 'input' is [[[1, 1, 1], [2, 2, 2]],
  #             [[3, 3, 3], [4, 4, 4]],
  #             [[5, 5, 5], [6, 6, 6]]]
  tf.slice(input, [1, 0, 0], [2, 1, 3], [1, 1, 1]) ==> [[[3, 3, 3]]]
  tf.slice(input, [1, 0, 0], [2, 2, 3], [1, 1, 1]) ==> [[[3, 3, 3],
                                                         [4, 4, 4]]]
  tf.slice(input, [1, 1, 0], [2, -1, 3], [1, -1, 1]) ==>[[[4, 4, 4],
                                                          [3, 3, 3]]]
  ```

  Args:
    input_: A `Tensor`.
    begin: An `int32` or `int64` `Tensor`.
    end: An `int32` or `int64` `Tensor`.
    strides: An `int32` or `int64` `Tensor`.
    begin_mask: An `int32` mask.
    end_mask: An `int32` mask.
    ellipsis_mask: An `int32` mask.
    new_axis_mask: An `int32` mask.
    shrink_axis_mask: An `int32` mask.
    var: The variable coresponding to `input_` or None
    name: A name for the operation (optional).

  Returns:
    A `Tensor` the same type as `input`.
  """
  op = gen_array_ops.strided_slice(
      input=input_,
      begin=begin,
      end=end,
      strides=strides,
      name=name,
      begin_mask=begin_mask,
      end_mask=end_mask,
      ellipsis_mask=ellipsis_mask,
      new_axis_mask=new_axis_mask,
      shrink_axis_mask=shrink_axis_mask)

  def assign(val):
    """Closure that holds all the arguments to create an assignment."""

    if var is None:
      raise ValueError("Sliced assignment is only supported for variables")

    return gen_array_ops.strided_slice_assign(
        ref=var,
        begin=begin,
        end=end,
        strides=strides,
        value=val,
        name=name + "_assign",
        begin_mask=begin_mask,
        end_mask=end_mask,
        ellipsis_mask=ellipsis_mask,
        new_axis_mask=new_axis_mask,
        shrink_axis_mask=shrink_axis_mask)

  op.assign = assign
  return op


def _SliceHelperVar(var, slice_spec):
  """Creates a slice helper object given a variable.

  This allows creating a sub-tensor from part of the current contents
  of a variable.
  See
  [`Tensor.__getitem__`](../../api_docs/python/framework.md#Tensor.__getitem__)
  for detailed examples of slicing.

  This function in addition also allows assignment to a sliced range.
  This is similar to `__setitem__` functionality in Python. However,
  the syntax is different so that the user can capture the assignment
  operation for grouping or passing to `sess.run()`.
  For example,

  ```prettyprint
  import tensorflow as tf
  A = tf.Variable([[1,2,3], [4,5,6], [7,8,9]], dtype=tf.float32)
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print sess.run(A[:2, :2]) # => [[1,2], [4,5]]

    op = A[:2,:2].assign(22. * tf.ones((2, 2)))
    print sess.run(op) # => [[22, 22, 3], [22, 22, 6], [7,8,9]]
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
    TypeError: If the slice indices aren't int, slice, or Ellipsis.

  """

  return _SliceHelper(var._AsTensor(), slice_spec, var)

ops.Tensor._override_operator("__getitem__", _SliceHelper)


def pack(values, axis=0, name="pack"):
  """Packs a list of rank-`R` tensors into one rank-`(R+1)` tensor.

  Packs the list of tensors in `values` into a tensor with rank one higher than
  each tensor in `values`, by packing them along the `axis` dimension.
  Given a list of length `N` of tensors of shape `(A, B, C)`;

  if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
  if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
  Etc.

  For example:

  ```prettyprint
  # 'x' is [1, 4]
  # 'y' is [2, 5]
  # 'z' is [3, 6]
  pack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
  pack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
  ```

  This is the opposite of unpack.  The numpy equivalent is

      tf.pack([x, y, z]) = np.asarray([x, y, z])

  Args:
    values: A list of `Tensor` objects with the same shape and type.
    axis: An `int`. The axis to pack along. Defaults to the first dimension.
      Supports negative indexes.
    name: A name for this operation (optional).

  Returns:
    output: A packed `Tensor` with the same type as `values`.

  Raises:
    ValueError: If `axis` is out of the range [-(R+1), R+1).
  """
  if axis == 0:
    try:
      # If the input is a constant list, it can be converted to a constant op
      return ops.convert_to_tensor(values, name=name)
    except (TypeError, ValueError):
      pass  # Input list contains non-constant tensors

  value_shape = ops.convert_to_tensor(values[0], name=name).get_shape()
  if value_shape.ndims is not None:
    expanded_num_dims = value_shape.ndims + 1
    if axis < -expanded_num_dims or axis >= expanded_num_dims:
      raise ValueError("axis = %d not in [%d, %d)" %
                       (axis, -expanded_num_dims, expanded_num_dims))

  return gen_array_ops._pack(values, axis=axis, name=name)


# pylint: disable=invalid-name
def _autopacking_helper(list_or_tuple, dtype, name):
  """Converts the given list or tuple to a tensor by packing.

  Args:
    list_or_tuple: A (possibly nested) list or tuple containing a tensor.
    dtype: The element type of the returned tensor.
    name: A name for the returned tensor.

  Returns:
    A `tf.Tensor` with value equivalent to `list_or_tuple`.
  """
  must_pack = False
  converted_elems = []
  with ops.name_scope(name) as scope:
    for i, elem in enumerate(list_or_tuple):
      if ops.is_dense_tensor_like(elem):
        if dtype is not None and elem.dtype.base_dtype != dtype:
          raise TypeError(
              "Cannot convert a list containing a tensor of dtype "
              "%s to %s (Tensor is: %r)" % (elem.dtype, dtype, elem))
        converted_elems.append(elem)
        must_pack = True
      elif isinstance(elem, (list, tuple)):
        converted_elem = _autopacking_helper(elem, dtype, str(i))
        if ops.is_dense_tensor_like(converted_elem):
          must_pack = True
        converted_elems.append(converted_elem)
      else:
        converted_elems.append(elem)
    if must_pack:
      elems_as_tensors = []
      for i, elem in enumerate(converted_elems):
        if ops.is_dense_tensor_like(elem):
          elems_as_tensors.append(elem)
        else:
          # NOTE(mrry): This is inefficient, but it enables us to
          # handle the case where the list arguments are other
          # convertible-to-tensor types, such as numpy arrays.
          elems_as_tensors.append(
              constant_op.constant(elem, dtype=dtype, name=str(i)))
      return gen_array_ops._pack(elems_as_tensors, name=scope)
    else:
      return converted_elems


def _get_dtype_from_nested_lists(list_or_tuple):
  """Returns the dtype of any tensor-like object in `list_or_tuple`, if found.

  Args:
    list_or_tuple: A list or tuple representing an object that can be
      converted to a `tf.Tensor`.

  Returns:
    The dtype of any tensor-like object in `list_or_tuple`, or `None` if no
    such object exists.
  """
  for elem in list_or_tuple:
    if ops.is_dense_tensor_like(elem):
      return elem.dtype.base_dtype
    elif isinstance(elem, (list, tuple)):
      maybe_dtype = _get_dtype_from_nested_lists(elem)
      if maybe_dtype is not None:
        return maybe_dtype
  return None


def _autopacking_conversion_function(v, dtype=None, name=None, as_ref=False):
  """Tensor conversion function that automatically packs arguments."""
  if as_ref:
    return NotImplemented
  inferred_dtype = _get_dtype_from_nested_lists(v)
  if inferred_dtype is None:
    # We did not find any tensor-like objects in the nested lists, so defer to
    # other conversion functions.
    return NotImplemented
  if dtype is not None and dtype != inferred_dtype:
    return NotImplemented
  return _autopacking_helper(v, inferred_dtype, name or "packed")
# pylint: enable=invalid-name


# NOTE: Register this conversion function to run *before* one that
# assumes every element is a value.
ops.register_tensor_conversion_function(
    (list, tuple), _autopacking_conversion_function, 99)


def unpack(value, num=None, axis=0, name="unpack"):
  """Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors.

  Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
  If `num` is not specified (the default), it is inferred from `value`'s shape.
  If `value.shape[axis]` is not known, `ValueError` is raised.

  For example, given a tensor of shape `(A, B, C, D)`;

  If `axis == 0` then the i'th tensor in `output` is the slice
    `value[i, :, :, :]` and each tensor in `output` will have shape `(B, C, D)`.
    (Note that the dimension unpacked along is gone, unlike `split`).

  If `axis == 1` then the i'th tensor in `output` is the slice
    `value[:, i, :, :]` and each tensor in `output` will have shape `(A, C, D)`.
  Etc.

  This is the opposite of pack.  The numpy equivalent is

      tf.unpack(x, n) = list(x)

  Args:
    value: A rank `R > 0` `Tensor` to be unpacked.
    num: An `int`. The length of the dimension `axis`. Automatically inferred
      if `None` (the default).
    axis: An `int`. The axis to unpack along. Defaults to the first
      dimension. Supports negative indexes.
    name: A name for the operation (optional).

  Returns:
    The list of `Tensor` objects unpacked from `value`.

  Raises:
    ValueError: If `num` is unspecified and cannot be inferred.
    ValueError: If `axis` is out of the range [-R, R).
  """
  if num is None:
    value = ops.convert_to_tensor(value)
    value_shape = value.get_shape()
    if value_shape.ndims is not None:
      if axis < -value_shape.ndims or axis >= value_shape.ndims:
        raise ValueError("axis = %d not in [%d, %d)" %
                         (axis, -value_shape.ndims, value_shape.ndims))
      num = value_shape[axis].value
  if num is None:
    raise ValueError("Cannot infer num from shape %s" % value_shape)
  return gen_array_ops._unpack(value, num=num, axis=axis, name=name)


def concat(concat_dim, values, name="concat"):
  """Concatenates tensors along one dimension.

  Concatenates the list of tensors `values` along dimension `concat_dim`.  If
  `values[i].shape = [D0, D1, ... Dconcat_dim(i), ...Dn]`, the concatenated
  result has shape

      [D0, D1, ... Rconcat_dim, ...Dn]

  where

      Rconcat_dim = sum(Dconcat_dim(i))

  That is, the data from the input tensors is joined along the `concat_dim`
  dimension.

  The number of dimensions of the input tensors must match, and all dimensions
  except `concat_dim` must be equal.

  For example:

  ```python
  t1 = [[1, 2, 3], [4, 5, 6]]
  t2 = [[7, 8, 9], [10, 11, 12]]
  tf.concat(0, [t1, t2]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
  tf.concat(1, [t1, t2]) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

  # tensor t3 with shape [2, 3]
  # tensor t4 with shape [2, 3]
  tf.shape(tf.concat(0, [t3, t4])) ==> [4, 3]
  tf.shape(tf.concat(1, [t3, t4])) ==> [2, 6]
  ```

  Note: If you are concatenating along a new axis consider using pack.
  E.g.

  ```python
  tf.concat(axis, [tf.expand_dims(t, axis) for t in tensors])
  ```

  can be rewritten as

  ```python
  tf.pack(tensors, axis=axis)
  ```

  Args:
    concat_dim: 0-D `int32` `Tensor`.  Dimension along which to concatenate.
    values: A list of `Tensor` objects or a single `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` resulting from concatenation of the input tensors.
  """
  if not isinstance(values, (list, tuple)):
    values = [values]
  # TODO(mrry): Change to return values?
  if len(values) == 1:  # Degenerate case of one tensor.
    # Make a throwaway call to convert_to_tensor to make sure
    # that concat_dim is of the correct type, and make sure that
    # the returned tensor is a scalar.
    # TODO(keveman): Implement a standalone type and shape checker.
    with ops.name_scope(name) as scope:
      ops.convert_to_tensor(concat_dim,
                            name="concat_dim",
                            dtype=dtypes.int32).get_shape(
                            ).assert_is_compatible_with(tensor_shape.scalar())
      return identity(values[0], name=scope)
  return gen_array_ops._concat(concat_dim=concat_dim,
                               values=values,
                               name=name)


ops.RegisterShape("Pack")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Unpack")(common_shapes.call_cpp_shape_fn)


@ops.RegisterShape("Concat")
def _ConcatShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[0])


ops.RegisterShape("ConcatOffset")(common_shapes.call_cpp_shape_fn)


def boolean_mask(tensor, mask, name="boolean_mask"):
  """Apply boolean mask to tensor.  Numpy equivalent is `tensor[mask]`.

  ```python
  # 1-D example
  tensor = [0, 1, 2, 3]
  mask = [True, False, True, False]
  boolean_mask(tensor, mask) ==> [0, 2]
  ```

  In general, `0 < dim(mask) = K <= dim(tensor)`, and `mask`'s shape must match
  the first K dimensions of `tensor`'s shape.  We then have:
    `boolean_mask(tensor, mask)[i, j1,...,jd] = tensor[i1,...,iK,j1,...,jd]`
  where `(i1,...,iK)` is the ith `True` entry of `mask` (row-major order).

  Args:
    tensor:  N-D tensor.
    mask:  K-D boolean tensor, K <= N and K must be known statically.
    name:  A name for this operation (optional).

  Returns:
    Tensor populated by entries in `tensor` corresponding to `True` values in
      `mask`.

  Raises:
    ValueError:  If shapes do not conform.

  Examples:

  ```python
  # 2-D example
  tensor = [[1, 2], [3, 4], [5, 6]]
  mask = [True, False, True]
  boolean_mask(tensor, mask) ==> [[1, 2], [5, 6]]
  ```
  """
  def _apply_mask_1d(reshaped_tensor, mask):
    """Mask tensor along dimension 0 with a 1-D mask."""
    indices = squeeze(where(mask), squeeze_dims=[1])
    return gather(reshaped_tensor, indices)

  with ops.name_scope(name, values=[tensor, mask]):
    tensor = ops.convert_to_tensor(tensor, name="tensor")
    mask = ops.convert_to_tensor(mask, name="mask")

    shape_mask = mask.get_shape()
    ndims_mask = shape_mask.ndims
    shape_tensor = tensor.get_shape()
    if ndims_mask == 0:
      raise ValueError("mask cannot be scalar.")
    if ndims_mask is None:
      raise ValueError(
          "mask dimensions must be specified, even if some dimensions are None"
          ".  E.g. shape=[None] is ok, but shape=None is not.")
    shape_tensor[:ndims_mask].assert_is_compatible_with(shape_mask)

    tensor = reshape(tensor, concat(0, [[-1], shape(tensor)[ndims_mask:]]))
    first_dim = shape_tensor[:ndims_mask].num_elements()
    tensor.set_shape(
        tensor_shape.as_shape([first_dim])
        .concatenate(shape_tensor[ndims_mask:]))

    mask = reshape(mask, [-1])
    return _apply_mask_1d(tensor, mask)


def sparse_mask(a, mask_indices, name=None):
  """Masks elements of `IndexedSlices`.

  Given an `IndexedSlices` instance `a`, returns another `IndexedSlices` that
  contains a subset of the slices of `a`. Only the slices at indices not
  specified in `mask_indices` are returned.

  This is useful when you need to extract a subset of slices in an
  `IndexedSlices` object.

  For example:

  ```python
  # `a` contains slices at indices [12, 26, 37, 45] from a large tensor
  # with shape [1000, 10]
  a.indices => [12, 26, 37, 45]
  tf.shape(a.values) => [4, 10]

  # `b` will be the subset of `a` slices at its second and third indices, so
  # we want to mask its first and last indices (which are at absolute
  # indices 12, 45)
  b = tf.sparse_mask(a, [12, 45])

  b.indices => [26, 37]
  tf.shape(b.values) => [2, 10]

  ```

  Args:
    * `a`: An `IndexedSlices` instance.
    * `mask_indices`: Indices of elements to mask.
    * `name`: A name for the operation (optional).

  Returns:
    The masked `IndexedSlices` instance.
  """
  with ops.name_scope(name, "sparse_mask", [a, mask_indices]) as name:
    indices = a.indices
    out_indices, to_gather = listdiff(indices, mask_indices)
    out_values = gather(a.values, to_gather, name=name)
    return ops.IndexedSlices(out_values, out_indices, a.dense_shape)


def split(split_dim, num_split, value, name="split"):
  """Splits a tensor into `num_split` tensors along one dimension.

  Splits `value` along dimension `split_dim` into `num_split` smaller tensors.
  Requires that `num_split` evenly divide `value.shape[split_dim]`.

  For example:

  ```python
  # 'value' is a tensor with shape [5, 30]
  # Split 'value' into 3 tensors along dimension 1
  split0, split1, split2 = tf.split(1, 3, value)
  tf.shape(split0) ==> [5, 10]
  ```

  Note: If you are splitting along an axis by the length of that axis, consider
  using unpack, e.g.

  ```python
  num_items = t.get_shape()[axis].value
  [tf.squeeze(s, [axis]) for s in tf.split(axis, num_items, t)]
  ```

  can be rewritten as

  ```python
  tf.unpack(t, axis=axis)
  ```

  Args:
    split_dim: A 0-D `int32` `Tensor`. The dimension along which to split.
      Must be in the range `[0, rank(value))`.
    num_split: A Python integer. The number of ways to split.
    value: The `Tensor` to split.
    name: A name for the operation (optional).

  Returns:
    `num_split` `Tensor` objects resulting from splitting `value`.
  """
  return gen_array_ops._split(split_dim=split_dim,
                              num_split=num_split,
                              value=value,
                              name=name)


ops.RegisterShape("Reverse")(common_shapes.call_cpp_shape_fn)


def transpose(a, perm=None, name="transpose"):
  """Transposes `a`. Permutes the dimensions according to `perm`.

  The returned tensor's dimension i will correspond to the input dimension
  `perm[i]`. If `perm` is not given, it is set to (n-1...0), where n is
  the rank of the input tensor. Hence by default, this operation performs a
  regular matrix transpose on 2-D input Tensors.

  For example:

  ```python
  # 'x' is [[1 2 3]
  #         [4 5 6]]
  tf.transpose(x) ==> [[1 4]
                       [2 5]
                       [3 6]]

  # Equivalently
  tf.transpose(x, perm=[1, 0]) ==> [[1 4]
                                    [2 5]
                                    [3 6]]

  # 'perm' is more useful for n-dimensional tensors, for n > 2
  # 'x' is   [[[1  2  3]
  #            [4  5  6]]
  #           [[7  8  9]
  #            [10 11 12]]]
  # Take the transpose of the matrices in dimension-0
  tf.transpose(x, perm=[0, 2, 1]) ==> [[[1  4]
                                        [2  5]
                                        [3  6]]

                                       [[7 10]
                                        [8 11]
                                        [9 12]]]
  ```

  Args:
    a: A `Tensor`.
    perm: A permutation of the dimensions of `a`.
    name: A name for the operation (optional).

  Returns:
    A transposed `Tensor`.
  """
  with ops.name_scope(name, "transpose", [a]) as name:
    if perm is None:
      rank = gen_array_ops.rank(a)
      perm = (rank - 1) - gen_math_ops._range(0, rank, 1)
      ret = gen_array_ops.transpose(a, perm, name=name)
      # NOTE(mrry): Setting the shape explicitly because
      #   reverse is not handled by the shape function.
      input_shape = ret.op.inputs[0].get_shape().dims
      if input_shape is not None:
        ret.set_shape(input_shape[::-1])
    else:
      ret = gen_array_ops.transpose(a, perm, name=name)
    return ret


# pylint: disable=invalid-name
def matrix_transpose(a, name="matrix_transpose"):
  """Transposes last two dimensions of tensor `a`.

  For example:

  ```python
  # Matrix with no batch dimension.
  # 'x' is [[1 2 3]
  #         [4 5 6]]
  tf.matrix_transpose(x) ==> [[1 4]
                                   [2 5]
                                   [3 6]]

  # Matrix with two batch dimensions.
  # x.shape is [1, 2, 3, 4]
  # tf.matrix_transpose(x) is shape [1, 2, 4, 3]
  ```

  Args:
    a: A `Tensor` with `rank >= 2`.
    name: A name for the operation (optional).

  Returns:
    A transposed batch matrix `Tensor`.

  Raises:
    ValueError:  If `a` is determined statically to have `rank < 2`.
  """
  with ops.name_scope(name, values=[a]):
    a = ops.convert_to_tensor(a, name="a")

    # If we know the number of dimensions (statically), we can do two things:
    # 1. Check that `a` is a (batch) matrix.
    # 2. Use a python list for perm.  This preserves static shape information
    #    and avoids extra computations.
    a_shape = a.get_shape()
    ndims = a_shape.ndims
    if ndims is not None:
      if ndims < 2:
        raise ValueError(
            "Argument 'a' should be a (batch) matrix, with rank >= 2.  Found: "
            "%s" % a_shape)
      perm = list(range(ndims - 2)) + [ndims - 1] + [ndims - 2]
    else:
      a_rank = rank(a)
      perm = concat(
          0, (gen_math_ops._range(0, a_rank - 2, 1), [a_rank - 1, a_rank - 2]))

    return transpose(a, perm=perm)
# pylint: enable=invalid-name


def zeros(shape, dtype=dtypes.float32, name=None):
  """Creates a tensor with all elements set to zero.

  This operation returns a tensor of type `dtype` with shape `shape` and
  all elements set to zero.

  For example:

  ```python
  tf.zeros([3, 4], tf.int32) ==> [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
  ```

  Args:
    shape: Either a list of integers, or a 1-D `Tensor` of type `int32`.
    dtype: The type of an element in the resulting `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with all elements set to zero.
  """
  dtype = dtypes.as_dtype(dtype).base_dtype
  with ops.name_scope(name, "zeros", [shape]) as name:
    zero = False if dtype == dtypes.bool else 0
    try:
      shape = tensor_shape.as_shape(shape)
      output = constant(zero, shape=shape, dtype=dtype, name=name)
    except (TypeError, ValueError):
      shape = ops.convert_to_tensor(shape, dtype=dtypes.int32, name="shape")
      output = fill(shape, constant(zero, dtype=dtype), name=name)
  assert output.dtype.base_dtype == dtype
  return output


def zeros_like(tensor, dtype=None, name=None, optimize=True):
  """Creates a tensor with all elements set to zero.

  Given a single tensor (`tensor`), this operation returns a tensor of the
  same type and shape as `tensor` with all elements set to zero. Optionally,
  you can use `dtype` to specify a new type for the returned tensor.

  For example:

  ```python
  # 'tensor' is [[1, 2, 3], [4, 5, 6]]
  tf.zeros_like(tensor) ==> [[0, 0, 0], [0, 0, 0]]
  ```

  Args:
    tensor: A `Tensor`.
    dtype: A type for the returned `Tensor`. Must be `float32`, `float64`,
    `int8`, `int16`, `int32`, `int64`, `uint8`, `complex64`, or `complex128`.
    name: A name for the operation (optional).
    optimize: if true, attempt to statically determine the shape of 'tensor'
    and encode it as a constant.

  Returns:
    A `Tensor` with all elements set to zero.
  """
  with ops.name_scope(name, "zeros_like", [tensor]) as name:
    tensor = ops.convert_to_tensor(tensor, name="tensor")
    if dtype is not None and tensor.dtype != dtype:
      ret = zeros(shape_internal(tensor, optimize=optimize), dtype, name=name)
      ret.set_shape(tensor.get_shape())
      return ret
    else:
      return gen_array_ops._zeros_like(tensor, name=name)


def ones_like(tensor, dtype=None, name=None, optimize=True):
  """Creates a tensor with all elements set to 1.

  Given a single tensor (`tensor`), this operation returns a tensor of the same
  type and shape as `tensor` with all elements set to 1. Optionally, you can
  specify a new type (`dtype`) for the returned tensor.

  For example:

  ```python
  # 'tensor' is [[1, 2, 3], [4, 5, 6]]
  tf.ones_like(tensor) ==> [[1, 1, 1], [1, 1, 1]]
  ```

  Args:
    tensor: A `Tensor`.
    dtype: A type for the returned `Tensor`. Must be `float32`, `float64`,
      `int8`, `int16`, `int32`, `int64`, `uint8`, `complex64`, `complex128` or
      `bool`.
    name: A name for the operation (optional).
    optimize: if true, attempt to statically determine the shape of 'tensor'
    and encode it as a constant.

  Returns:
    A `Tensor` with all elements set to 1.
  """
  with ops.name_scope(name, "ones_like", [tensor]) as name:
    tensor = ops.convert_to_tensor(tensor, name="tensor")
    ones_shape = shape_internal(tensor, optimize=optimize)
    if dtype is None:
      dtype = tensor.dtype
    ret = ones(ones_shape, dtype=dtype, name=name)
    ret.set_shape(tensor.get_shape())
    return ret


def ones(shape, dtype=dtypes.float32, name=None):
  """Creates a tensor with all elements set to 1.

  This operation returns a tensor of type `dtype` with shape `shape` and all
  elements set to 1.

  For example:

  ```python
  tf.ones([2, 3], tf.int32) ==> [[1, 1, 1], [1, 1, 1]]
  ```

  Args:
    shape: Either a list of integers, or a 1-D `Tensor` of type `int32`.
    dtype: The type of an element in the resulting `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with all elements set to 1.
  """
  dtype = dtypes.as_dtype(dtype).base_dtype
  with ops.name_scope(name, "ones", [shape]) as name:
    one = True if dtype == dtypes.bool else 1
    try:
      shape = tensor_shape.as_shape(shape)
      output = constant(one, shape=shape, dtype=dtype, name=name)
    except (TypeError, ValueError):
      shape = ops.convert_to_tensor(shape, dtype=dtypes.int32, name="shape")
      output = fill(shape, constant(one, dtype=dtype), name=name)
  assert output.dtype.base_dtype == dtype
  return output


def placeholder(dtype, shape=None, name=None):
  """Inserts a placeholder for a tensor that will be always fed.

  **Important**: This tensor will produce an error if evaluated. Its value must
  be fed using the `feed_dict` optional argument to `Session.run()`,
  `Tensor.eval()`, or `Operation.run()`.

  For example:

  ```python
  x = tf.placeholder(tf.float32, shape=(1024, 1024))
  y = tf.matmul(x, x)

  with tf.Session() as sess:
    print(sess.run(y))  # ERROR: will fail because x was not fed.

    rand_array = np.random.rand(1024, 1024)
    print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
  ```

  Args:
    dtype: The type of elements in the tensor to be fed.
    shape: The shape of the tensor to be fed (optional). If the shape is not
      specified, you can feed a tensor of any shape.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` that may be used as a handle for feeding a value, but not
    evaluated directly.
  """
  shape = tensor_shape.as_shape(shape)
  if shape.is_fully_defined():
    dim_list = shape.as_list()
  else:
    dim_list = []
  ret = gen_array_ops._placeholder(
      dtype=dtype,
      shape=dim_list,
      name=name)
  ret.set_shape(shape)
  return ret


def sparse_placeholder(dtype, shape=None, name=None):
  """Inserts a placeholder for a sparse tensor that will be always fed.

  **Important**: This sparse tensor will produce an error if evaluated.
  Its value must be fed using the `feed_dict` optional argument to
  `Session.run()`, `Tensor.eval()`, or `Operation.run()`.

  For example:

  ```python
  x = tf.sparse_placeholder(tf.float32)
  y = tf.sparse_reduce_sum(x)

  with tf.Session() as sess:
    print(sess.run(y))  # ERROR: will fail because x was not fed.

    indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
    values = np.array([1.0, 2.0], dtype=np.float32)
    shape = np.array([7, 9, 2], dtype=np.int64)
    print(sess.run(y, feed_dict={
      x: tf.SparseTensorValue(indices, values, shape)}))  # Will succeed.
    print(sess.run(y, feed_dict={
      x: (indices, values, shape)}))  # Will succeed.

    sp = tf.SparseTensor(indices=indices, values=values, shape=shape)
    sp_value = sp.eval(session)
    print(sess.run(y, feed_dict={x: sp_value}))  # Will succeed.
  ```

  Args:
    dtype: The type of `values` elements in the tensor to be fed.
    shape: The shape of the tensor to be fed (optional). If the shape is not
      specified, you can feed a sparse tensor of any shape.
    name: A name for prefixing the operations (optional).

  Returns:
    A `SparseTensor` that may be used as a handle for feeding a value, but not
    evaluated directly.
  """
  if shape is None:
    shape = placeholder(
        dtypes.int64, shape=[None],
        name=(name + "/shape") if name is not None else None)
  else:
    shape = ops.convert_to_tensor(
        shape, name=(name + "/shape") if name is not None else None)
  return ops.SparseTensor(
      values=placeholder(
          dtype, shape=[None],
          name=(name + "/values") if name is not None else None),
      indices=placeholder(
          dtypes.int64, shape=[None, None],
          name=(name + "/indices") if name is not None else None),
      shape=shape
  )


def pad(tensor, paddings, mode="CONSTANT", name=None):  # pylint: disable=invalid-name
  """Pads a tensor.

  This operation pads a `tensor` according to the `paddings` you specify.
  `paddings` is an integer tensor with shape `[n, 2]`, where n is the rank of
  `tensor`. For each dimension D of `input`, `paddings[D, 0]` indicates how
  many values to add before the contents of `tensor` in that dimension, and
  `paddings[D, 1]` indicates how many values to add after the contents of
  `tensor` in that dimension. If `mode` is "REFLECT" then both `paddings[D, 0]`
  and `paddings[D, 1]` must be no greater than `tensor.dim_size(D) - 1`. If
  `mode` is "SYMMETRIC" then both `paddings[D, 0]` and `paddings[D, 1]` must be
  no greater than `tensor.dim_size(D)`.

  The padded size of each dimension D of the output is:

  `paddings[D, 0] + tensor.dim_size(D) + paddings[D, 1]`

  For example:

  ```python
  # 't' is [[1, 2, 3], [4, 5, 6]].
  # 'paddings' is [[1, 1,], [2, 2]].
  # rank of 't' is 2.
  pad(t, paddings, "CONSTANT") ==> [[0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 2, 3, 0, 0],
                                    [0, 0, 4, 5, 6, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0]]

  pad(t, paddings, "REFLECT") ==> [[6, 5, 4, 5, 6, 5, 4],
                                   [3, 2, 1, 2, 3, 2, 1],
                                   [6, 5, 4, 5, 6, 5, 4],
                                   [3, 2, 1, 2, 3, 2, 1]]

  pad(t, paddings, "SYMMETRIC") ==> [[2, 1, 1, 2, 3, 3, 2],
                                     [2, 1, 1, 2, 3, 3, 2],
                                     [5, 4, 4, 5, 6, 6, 5],
                                     [5, 4, 4, 5, 6, 6, 5]]
  ```

  Args:
    tensor: A `Tensor`.
    paddings: A `Tensor` of type `int32`.
    mode: One of "CONSTANT", "REFLECT", or "SYMMETRIC".
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.

  Raises:
    ValueError: When mode is not one of "CONSTANT", "REFLECT", or "SYMMETRIC".
  """

  if mode == "CONSTANT":
    return gen_array_ops._pad(tensor, paddings, name=name)
  if mode == "REFLECT":
    return gen_array_ops._mirror_pad(tensor,
                                     paddings,
                                     mode="REFLECT",
                                     name=name)
  if mode == "SYMMETRIC":
    return gen_array_ops._mirror_pad(tensor,
                                     paddings,
                                     mode="SYMMETRIC",
                                     name=name)
  raise ValueError("Unknown padding mode: %s" % mode)


def meshgrid(*args, **kwargs):
  """Broadcasts parameters for evaluation on an N-D grid.

  Given N one-dimensional coordinate arrays `*args`, returns a list `outputs`
  of N-D coordinate arrays for evaluating expressions on an N-D grid.

  Notes:

  `meshgrid` supports cartesian ('xy') and matrix ('ij') indexing conventions.
  When the `indexing` argument is set to 'xy' (the default), the broadcasting
  instructions for the first two dimensions are swapped.

  Examples:

  Calling `X, Y = meshgrid(x, y)` with the tensors

  ```prettyprint
    x = [1, 2, 3]
    y = [4, 5, 6]
  ```

  results in

  ```prettyprint
    X = [[1, 1, 1],
         [2, 2, 2],
         [3, 3, 3]]
    Y = [[4, 5, 6],
         [4, 5, 6],
         [4, 5, 6]]
  ```

  Args:
    *args: `Tensor`s with rank 1
    indexing: Either 'xy' or 'ij' (optional, default: 'xy')
    name: A name for the operation (optional).

  Returns:
    outputs: A list of N `Tensor`s with rank N
  """
  indexing = kwargs.pop("indexing", "xy")
  name = kwargs.pop("name", "meshgrid")
  if len(kwargs) > 0:
    key = list(kwargs.keys())[0]
    raise TypeError("'{}' is an invalid keyword argument "
                    "for this function".format(key))

  if indexing not in ("xy", "ij"):
    raise ValueError("indexing parameter must be either 'xy' or 'ij'")

  with ops.name_scope(name, "meshgrid", args) as name:
    num_inputs = len(args)
    ones = (1,) * num_inputs

    # Cannot import Assert from control_flow_ops, so we incur the
    # penalty of possibly copying from GPU to CPU regardless of the
    # equality of the predicate.
    asserts = [gen_logging_ops._assert(
                 gen_math_ops.equal(rank(x), 1),
                 ["Input %d needs to have rank 1: " % i, rank(x)],
               ) for i, x in enumerate(args)]

    # Prepare reshape by inserting dimensions with size 1 where needed
    shapes = [ones[:i] + (-1,) + ones[i + 1:] for i in range(num_inputs)]
    # Create parameters for broadcasting each tensor to the full size
    sizes = [size(x) for x in args]
    bcast = [sizes[:i] + [1] + sizes[i + 1:] for i in range(num_inputs)]

    # By default, the numpy version swaps the instructions
    # for the first and second dimension
    if indexing == "xy" and num_inputs > 1:
      shapes[0], shapes[1] = shapes[1], shapes[0]
      bcast[0], bcast[1] = bcast[1], bcast[0]

    results = []
    with ops.control_dependencies(asserts):
      for a, r, e in zip(args, shapes, bcast):
        results.append(tile(reshape(a, r), e))

    return results


ops.RegisterShape("Placeholder")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("CheckNumerics")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Identity")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("RefIdentity")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("StopGradient")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("MatrixBandPart")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("QuantizeAndDequantize")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Rank")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Size")(common_shapes.call_cpp_shape_fn)


@ops.RegisterShape("Slice")
def _SliceShape(op):
  """Shape function for array_ops.slice."""
  input_shape = op.inputs[0].get_shape()
  begin_shape = op.inputs[1].get_shape().with_rank(1)
  sizes_shape = op.inputs[2].get_shape().with_rank(1)
  ndims = begin_shape.merge_with(sizes_shape)[0].value
  if ndims is not None:
    input_shape.assert_has_rank(ndims)
  # NOTE(mrry): Use `constant_value_as_shape()` to handle
  # partially-known values.
  begin_value = tensor_util.constant_value_as_shape(
      op.inputs[1]).with_rank(ndims)
  # NOTE(mrry): We can't use `constant_value_as_shape()` for `sizes`
  # because it might contain -1, which can't be represented as a
  # `TensorShape`.
  sizes_value = tensor_util.constant_value(op.inputs[2])
  if sizes_value is not None:
    returned_dims = []
    for i, (slice_size, begin_dim) in enumerate(zip(sizes_value.ravel(),
                                                    begin_value.dims)):
      if slice_size != -1:
        returned_dims.append(slice_size)
      else:
        returned_dims.append(input_shape[i] - begin_dim)
    return [tensor_shape.TensorShape(returned_dims)]
  else:
    if input_shape.ndims is not None:
      return [tensor_shape.unknown_shape(ndims=input_shape.ndims)]
    elif ndims is not None:
      return [tensor_shape.unknown_shape(ndims=ndims)]
    else:
      return [tensor_shape.unknown_shape()]


NEW_AXIS = -1
SHRINK_AXIS = -2


# PEP-8 naming
# pylint: disable=invalid-name
def _compute_size_of_strided_dim(shrink, spec, size):
  """Computes the size of a single strided slice dimension."""

  unknown = None  # Document what None means here.
  use_full_range = None  # Document other use of None.
  # if this is a shrink axis (i.e. a non-range index)
  # it either will produce an error or return 1
  if shrink:
    return 1
  if size is unknown or size.value is unknown:
    return unknown
  size = size.value
  stride = spec.step
  if stride is not unknown:
    if stride == 0:
      return unknown
    stride = spec.step
    valid_range = [0, size] if stride > 0 else [-1, size - 1]

    # PEP-8 naming
    # pylint: disable=invalid-name
    def canonical(x, c):
      if x is use_full_range:
        return valid_range[c] if stride > 0 else valid_range[(c + 1) & 1]
      else:
        x_fwd = size + x if x < 0 else x  # make negative indices positive
        return max(valid_range[0], min(valid_range[1], x_fwd))

    begin = canonical(spec.start, 0)
    end = canonical(spec.stop, 1)
    interval_length = end - begin
    if interval_length == 0 or ((interval_length < 0) != (stride < 0)):
      return 0
    else:
      remainder = 1 if interval_length % stride != 0 else 0
      return interval_length // stride + remainder
  else:
    return unknown  # unknown because stride is unknown


@ops.RegisterShape("StridedSliceGrad")
def _StridedSliceGradShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[0])


ops.RegisterShape("StridedSliceAssign")(common_shapes.call_cpp_shape_fn)


@ops.RegisterShape("StridedSlice")
def _DelegateStridedSliceShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[1, 2, 3])


ops.RegisterShape("Gather")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("GatherNd")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Unique")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("UniqueWithCounts")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("MatrixDiag")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("MatrixSetDiag")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("MatrixDiagPart")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Diag")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("DiagPart")(common_shapes.call_cpp_shape_fn)


@ops.RegisterShape("ExpandDims")
def _ExpandDims(op):
  return common_shapes.call_cpp_shape_fn(
      op, input_tensors_needed=[1])


ops.RegisterShape("Squeeze")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Bitcast")(common_shapes.call_cpp_shape_fn)


@ops.RegisterShape("Reshape")
def _ReshapeShape(op):
  """Shape function for Reshape op."""
  input_shape = op.inputs[0].get_shape()
  if input_shape.ndims is not None:
    num_elements = tensor_shape.Dimension(1)
    for dim in input_shape.dims:
      num_elements *= dim
  else:
    num_elements = tensor_shape.Dimension(None)
  new_shape = tensor_util.constant_value_as_shape(op.inputs[1])
  if new_shape.ndims is None:
    # We have no information about the shape of the output.
    return [new_shape]
  if None not in new_shape.as_list():
    # The new shape is fully defined.
    if (num_elements.value is not None
        and num_elements.value != np.prod(new_shape)):
      raise ValueError(
          "Cannot reshape a tensor with %d elements to shape %s (%d elements)"
          % (num_elements.value, new_shape, np.prod(new_shape)))
  elif num_elements.value is not None:
    # We know the number of elements, so we can calculate the missing
    # dimension in the new_shape.
    known_elements = 1
    unknown_indices = []
    for i, dim in enumerate(new_shape):
      if dim.value is None:
        unknown_indices.append(i)
      else:
        known_elements *= dim.value
    if known_elements != 0:
      if num_elements % known_elements != 0:
        raise ValueError("input has %s elements, which isn't divisible by %d" %
                         (num_elements, known_elements))
      if len(unknown_indices) == 1:
        unknown_index = unknown_indices[0]
        new_shape = new_shape.merge_with(
            new_shape[:unknown_index].concatenate(
                [num_elements // known_elements]).concatenate(
                    new_shape[unknown_index+1:]))
  return [new_shape]


ops.RegisterShape("BroadcastGradientArgs")(common_shapes.call_cpp_shape_fn)


@ops.RegisterShape("Fill")
def _FillShape(op):
  """Shape function for the Fill op.

  This op takes a vector of dimensions and a scalar, and produces a
  tensor with the given dimensions.

  Args:
    op: A Fill Operation.

  Returns:
    A single-element list containing the shape of the output.

  Raises:
    ValueError: If the shapes or arguments are known to be invalid.
  """
  op.inputs[0].get_shape().assert_has_rank(1)
  op.inputs[1].get_shape().assert_has_rank(0)
  fill_dims = tensor_util.constant_value(op.inputs[0])
  if fill_dims is not None and any(d < 0 for d in fill_dims):
    raise ValueError("Fill dimensions must be >= 0")
  return [tensor_util.constant_value_as_shape(op.inputs[0])]


ops.RegisterShape("InvertPermutation")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("ListDiff")(common_shapes.call_cpp_shape_fn)


@ops.RegisterShape("Pad")
@ops.RegisterShape("MirrorPad")
def _PadShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[1])


@ops.RegisterShape("MirrorPadGrad")
def _MirrorPadGradShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[1])


ops.RegisterShape("ReverseSequence")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Shape")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("ShapeN")(common_shapes.call_cpp_shape_fn)


@ops.RegisterShape("Transpose")
def _TransposeShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[1])


@ops.RegisterShape("Split")
def _SplitShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[0])


@ops.RegisterShape("Tile")
def _TileShape(op):
  """Shape function for the Tile op.

  This op has two inputs:

  * input: A rank-N tensor.
  * multiples: A length-N vector, in which the i^th element contains
    the factor by which `input` will be tiled in the i^th dimension.

  It has one output, which has the same rank as input, and additional
  elements according to the values in multiples

  Args:
    op: A Tile Operation.

  Returns:
    A single-element list containing the shape of the output.
  """
  multiples_shape = op.inputs[1].get_shape().with_rank(1)
  input_shape = op.inputs[0].get_shape().with_rank(multiples_shape[0].value)
  # NOTE(mrry): Represent `multiples` as a `TensorShape` because (i)
  # it is a vector of non-negative integers, and (ii) doing so allows
  # us to handle partially-known multiples.
  multiples = tensor_util.constant_value_as_shape(op.inputs[1]).with_rank(
      input_shape.ndims)
  if multiples.ndims is None:
    return [tensor_shape.unknown_shape()]
  else:
    output_dims = []
    for dim, multiple in zip(input_shape.dims, multiples.dims):
      output_dims.append(dim * multiple)
    return [tensor_shape.TensorShape(output_dims)]


@ops.RegisterShape("TileGrad")
def _TileGradShape(op):
  """Shape function for the TileGrad op."""
  multiples_shape = op.inputs[1].get_shape().with_rank(1)
  input_shape = op.inputs[0].get_shape().with_rank(multiples_shape[0])
  # NOTE(mrry): Represent `multiples` as a `TensorShape` because (i)
  # it is a vector of non-negative integers, and (ii) doing so allows
  # us to handle partially-known multiples.
  multiples = tensor_util.constant_value_as_shape(op.inputs[1]).with_rank(
      input_shape.ndims)
  if multiples.ndims is None:
    return [tensor_shape.unknown_shape()]
  else:
    output_dims = []
    for dim, multiple in zip(input_shape.dims, multiples.dims):
      output_dims.append(dim // multiple)
    return [tensor_shape.TensorShape(output_dims)]


ops.RegisterShape("Where")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("ZerosLike")(common_shapes.call_cpp_shape_fn)


def edit_distance(hypothesis, truth, normalize=True, name="edit_distance"):
  """Computes the Levenshtein distance between sequences.

  This operation takes variable-length sequences (`hypothesis` and `truth`),
  each provided as a `SparseTensor`, and computes the Levenshtein distance.
  You can normalize the edit distance by length of `truth` by setting
  `normalize` to true.

  For example, given the following input:

  ```python
  # 'hypothesis' is a tensor of shape `[2, 1]` with variable-length values:
  #   (0,0) = ["a"]
  #   (1,0) = ["b"]
  hypothesis = tf.SparseTensor(
      [[0, 0, 0],
       [1, 0, 0]],
      ["a", "b"]
      (2, 1, 1))

  # 'truth' is a tensor of shape `[2, 2]` with variable-length values:
  #   (0,0) = []
  #   (0,1) = ["a"]
  #   (1,0) = ["b", "c"]
  #   (1,1) = ["a"]
  truth = tf.SparseTensor(
      [[0, 1, 0],
       [1, 0, 0],
       [1, 0, 1],
       [1, 1, 0]]
      ["a", "b", "c", "a"],
      (2, 2, 2))

  normalize = True
  ```

  This operation would return the following:

  ```python
  # 'output' is a tensor of shape `[2, 2]` with edit distances normalized
  # by 'truth' lengths.
  output ==> [[inf, 1.0],  # (0,0): no truth, (0,1): no hypothesis
             [0.5, 1.0]]  # (1,0): addition, (1,1): no hypothesis
  ```

  Args:
    hypothesis: A `SparseTensor` containing hypothesis sequences.
    truth: A `SparseTensor` containing truth sequences.
    normalize: A `bool`. If `True`, normalizes the Levenshtein distance by
      length of `truth.`
    name: A name for the operation (optional).

  Returns:
    A dense `Tensor` with rank `R - 1`, where R is the rank of the
    `SparseTensor` inputs `hypothesis` and `truth`.

  Raises:
    TypeError: If either `hypothesis` or `truth` are not a `SparseTensor`.
  """
  if not isinstance(hypothesis, (ops.SparseTensor, ops.SparseTensorValue)):
    raise TypeError("Hypothesis must be a SparseTensor.")
  if not isinstance(truth, (ops.SparseTensor, ops.SparseTensorValue)):
    raise TypeError("Truth must be a SparseTensor.")

  return gen_array_ops._edit_distance(hypothesis.indices,
                                      hypothesis.values,
                                      hypothesis.shape,
                                      truth.indices,
                                      truth.values,
                                      truth.shape,
                                      normalize=normalize,
                                      name=name)


@ops.RegisterShape("EditDistance")
def _EditDistanceShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[2, 5])


# The remaining ops do not change the shape of their inputs.
@ops.RegisterShape("Quantize")
@ops.RegisterShape("Dequantize")
def _QuantizeDequantizeShape(op):
  unused_min_range = op.inputs[1].get_shape().merge_with(tensor_shape.scalar())
  unused_max_range = op.inputs[2].get_shape().merge_with(tensor_shape.scalar())
  return common_shapes.unchanged_shape(op)


ops.RegisterShape("ExtractImagePatches")(common_shapes.call_cpp_shape_fn)


def required_space_to_batch_paddings(input_shape,
                                     block_shape,
                                     base_paddings=None,
                                     name=None):
  """Calculate padding required to make block_shape divide input_shape.

  This function can be used to calculate a suitable paddings argument for use
  with space_to_batch_nd and batch_to_space_nd.

  Args:
    input_shape: int32 Tensor of shape [N].
    block_shape: int32 Tensor of shape [N].
    base_paddings: Optional int32 Tensor of shape [N, 2].  Specifies the minimum
      amount of padding to use.  All elements must be >= 0.  If not specified,
      defaults to 0.
    name: string.  Optional name prefix.

  Returns:
    (paddings, crops), where:

    `paddings` and `crops` are int32 Tensors of rank 2 and shape [N, 2]
    satisfying:

        paddings[i, 0] = base_paddings[i, 0].
        0 <= paddings[i, 1] - base_paddings[i, 1] < block_shape[i]
        (input_shape[i] + paddings[i, 0] + paddings[i, 1]) % block_shape[i] == 0

        crops[i, 0] = 0
        crops[i, 1] = paddings[i, 1] - base_paddings[i, 1]

  Raises: ValueError if called with incompatible shapes.
  """
  with ops.name_scope(name, "required_space_to_batch_paddings",
                      [input_shape, block_shape]):
    input_shape = ops.convert_to_tensor(input_shape,
                                        dtype=dtypes.int32,
                                        name="input_shape")
    block_shape = ops.convert_to_tensor(block_shape,
                                        dtype=dtypes.int32,
                                        name="block_shape")

    block_shape.get_shape().assert_is_fully_defined()
    block_shape.get_shape().assert_has_rank(1)
    num_block_dims = block_shape.get_shape()[0].value
    if num_block_dims == 0:
      return zeros([0, 2], dtypes.int32), zeros([0, 2], dtypes.int32)

    input_shape.get_shape().assert_is_compatible_with([num_block_dims])

    if base_paddings is not None:
      base_paddings = ops.convert_to_tensor(base_paddings,
                                            dtype=dtypes.int32,
                                            name="base_paddings")
      base_paddings.get_shape().assert_is_compatible_with([num_block_dims, 2])
    else:
      base_paddings = zeros([num_block_dims, 2], dtypes.int32)

    const_block_shape = tensor_util.constant_value(block_shape)
    const_input_shape = tensor_util.constant_value(input_shape)
    const_base_paddings = tensor_util.constant_value(base_paddings)
    if (const_block_shape is not None and const_input_shape is not None and
        const_base_paddings is not None):
      block_shape = const_block_shape
      input_shape = const_input_shape
      base_paddings = const_base_paddings

    # Use same expression for both constant and non-constant case.
    pad_start = base_paddings[:, 0]
    orig_pad_end = base_paddings[:, 1]
    full_input_shape = input_shape + pad_start + orig_pad_end
    pad_end_extra = (block_shape - full_input_shape % block_shape) % block_shape
    pad_end = orig_pad_end + pad_end_extra

    result_paddings = pack(
        [[pad_start[i], pad_end[i]] for i in range(num_block_dims)],
        name="paddings")
    result_crops = pack(
        [[0, pad_end_extra[i]] for i in range(num_block_dims)], name="crops")
    return result_paddings, result_crops


def space_to_batch(input, paddings, block_size, name=None):  # pylint: disable=redefined-builtin
  result = space_to_batch_nd(input,
                             paddings=paddings,
                             block_shape=np.array([block_size, block_size],
                                                  dtype=np.int64),
                             name=name)
  result.set_shape(result.get_shape().with_rank(4))
  return result


space_to_batch.__doc__ = gen_array_ops._space_to_batch.__doc__


def batch_to_space(input, crops, block_size, name=None):  # pylint: disable=redefined-builtin
  result = batch_to_space_nd(input,
                             crops=crops,
                             block_shape=np.array([block_size, block_size],
                                                  dtype=np.int64),
                             name=name)
  result.set_shape(result.get_shape().with_rank(4))
  return result


batch_to_space.__doc__ = gen_array_ops._batch_to_space.__doc__


@ops.RegisterShape("SpaceToBatch")
def _SpaceToBatchShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[1])


@ops.RegisterShape("SpaceToBatchND")
def _SpaceToBatchNDShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[1, 2])


@ops.RegisterShape("BatchToSpace")
def _BatchToSpaceShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[1])


@ops.RegisterShape("BatchToSpaceND")
def _BatchToSpaceNDShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[1, 2])


ops.RegisterShape("SpaceToDepth")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("DepthToSpace")(common_shapes.call_cpp_shape_fn)


def one_hot(indices, depth, on_value=None, off_value=None,
            axis=None, dtype=None, name=None):
  """Returns a one-hot tensor.

  The locations represented by indices in `indices` take value `on_value`,
  while all other locations take value `off_value`.

  `on_value` and `off_value` must have matching data types. If `dtype` is also
  provided, they must be the same data type as specified by `dtype`.

  If `on_value` is not provided, it will default to the value `1` with type
  `dtype`

  If `off_value` is not provided, it will default to the value `0` with type
  `dtype`

  If the input `indices` is rank `N`, the output will have rank `N+1`. The
  new axis is created at dimension `axis` (default: the new axis is appended
  at the end).

  If `indices` is a scalar the output shape will be a vector of length `depth`

  If `indices` is a vector of length `features`, the output shape will be:
  ```
    features x depth if axis == -1
    depth x features if axis == 0
  ```

  If `indices` is a matrix (batch) with shape `[batch, features]`, the output
  shape will be:
  ```
    batch x features x depth if axis == -1
    batch x depth x features if axis == 1
    depth x batch x features if axis == 0
  ```

  If `dtype` is not provided, it will attempt to assume the data type of
  `on_value` or `off_value`, if one or both are passed in. If none of
  `on_value`, `off_value`, or `dtype` are provided, `dtype` will default to the
  value `tf.float32`

  Note: If a non-numeric data type output is desired (tf.string, tf.bool, etc.),
  both `on_value` and `off_value` _must_ be provided to `one_hot`

  Examples
  =========

  Suppose that

  ```
    indices = [0, 2, -1, 1]
    depth = 3
    on_value = 5.0
    off_value = 0.0
    axis = -1
  ```

  Then output is `[4 x 3]`:

  ```
    output =
    [5.0 0.0 0.0]  // one_hot(0)
    [0.0 0.0 5.0]  // one_hot(2)
    [0.0 0.0 0.0]  // one_hot(-1)
    [0.0 5.0 0.0]  // one_hot(1)
  ```

  Suppose that

  ```
    indices = [[0, 2], [1, -1]]
    depth = 3
    on_value = 1.0
    off_value = 0.0
    axis = -1
  ```

  Then output is `[2 x 2 x 3]`:

  ```
    output =
    [
      [1.0, 0.0, 0.0]  // one_hot(0)
      [0.0, 0.0, 1.0]  // one_hot(2)
    ][
      [0.0, 1.0, 0.0]  // one_hot(1)
      [0.0, 0.0, 0.0]  // one_hot(-1)
    ]
  ```

  Using default values for `on_value` and `off_value`:

  ```
    indices = [0, 1, 2]
    depth = 3
  ```

  The output will be

  ```
    output =
    [[1., 0., 0.],
     [0., 1., 0.],
     [0., 0., 1.]]
  ```

  Args:
    indices: A `Tensor` of indices.
    depth: A scalar defining the depth of the one hot dimension.
    on_value: A scalar defining the value to fill in output when `indices[j]
      = i`. (default: 1)
    off_value: A scalar defining the value to fill in output when `indices[j]
      != i`. (default: 0)
    axis: The axis to fill (default: -1, a new inner-most axis).
    dtype: The data type of the output tensor.

  Returns:
    output: The one-hot tensor.

  Raises:
    TypeError: If dtype of either `on_value` or `off_value` don't match `dtype`
    TypeError: If dtype of `on_value` and `off_value` don't match one another
  """
  with ops.name_scope(name, "one_hot", [indices, depth, on_value, off_value,
                                        axis, dtype]) as name:
    on_exists = on_value is not None
    off_exists = off_value is not None

    on_dtype = ops.convert_to_tensor(on_value).dtype.base_dtype if on_exists \
                  else None
    off_dtype = ops.convert_to_tensor(off_value).dtype.base_dtype if off_exists\
                  else None

    if on_exists or off_exists:
      if dtype is not None:
        # Ensure provided on_value and/or off_value match dtype
        if (on_exists and on_dtype != dtype):
          raise TypeError("dtype {0} of on_value does not match " \
                          "dtype parameter {1}".format(on_dtype, dtype))
        if (off_exists and off_dtype != dtype):
          raise TypeError("dtype {0} of off_value does not match " \
                          "dtype parameter {1}".format(off_dtype, dtype))
      else:
        # dtype not provided: automatically assign it
        dtype = on_dtype if on_exists else off_dtype
    elif dtype is None:
      # None of on_value, off_value, or dtype provided. Default dtype to float32
      dtype = dtypes.float32

    if not on_exists:
      # on_value not provided: assign to value 1 of type dtype
      on_value = ops.convert_to_tensor(1, dtype, name="on_value")
      on_dtype = dtype
    if not off_exists:
      # off_value not provided: assign to value 0 of type dtype
      off_value = ops.convert_to_tensor(0, dtype, name="off_value")
      off_dtype = dtype

    if on_dtype != off_dtype:
      raise TypeError("dtype {0} of on_value does not match " \
                      "dtype {1} of off_value".format(on_dtype, off_dtype))

    return gen_array_ops._one_hot(indices, depth, on_value, off_value, axis,
                                  name)


@ops.RegisterShape("OneHot")
def _OneHotShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[1])


@ops.RegisterShape("PlaceholderWithDefault")
def _PlaceholderWithDefaultShape(op):
  """Shape function for the PlaceholderWithDefault op.

  This op acts as an identity when it is not fed (passing through a
  default value), but allows the user to feed it with tensors of a
  possibly less precise shape than its default value.

  Args:
    op: A PlaceholderWithDefault `Operation`.

  Returns:
    A single-element list containing the shape of the output.
  """
  input_shape = op.inputs[0].get_shape()
  output_shape = tensor_shape.TensorShape(op.get_attr("shape"))
  # NOTE(mrry): We don't merge these shapes, because `output_shape`
  # may be *less* precise than `input_shape`.
  input_shape.assert_is_compatible_with(output_shape)
  return [output_shape]


def sequence_mask(lengths, maxlen=None, dtype=dtypes.bool, name=None):
  """Return a mask tensor representing the first N positions of each row.

  Example:
  ```python
  tf.sequence_mask([1, 3, 2], 5) =
    [[True, False, False, False, False],
     [True, True, True, False, False],
     [True, True, False, False, False]]
  ```

  Args:
    lengths: 1D integer tensor, all its values < maxlen.
    maxlen: scalar integer tensor, maximum length of each row. Default: use
            maximum over lengths.
    dtype: output type of the resulting tensor.
    name: name of the op.
  Returns:
    A 2D mask tensor, as shown in the example above, cast to specified dtype.

  Raises:
    ValueError: if the arguments have invalid rank.
  """
  with ops.name_scope(name, "SequenceMask", [lengths, maxlen]):
    lengths = ops.convert_to_tensor(lengths)
    if lengths.get_shape().ndims != 1:
      raise ValueError("lengths must be 1D for sequence_mask")

    if maxlen is None:
      maxlen = gen_math_ops._max(lengths, [0])
    else:
      maxlen = ops.convert_to_tensor(maxlen)
    if maxlen.get_shape().ndims != 0:
      raise ValueError("maxlen must be scalar for sequence_mask")

    # The basic idea is to compare a range row vector of size maxlen:
    # [0, 1, 2, 3, 4]
    # to length as a matrix with 1 column: [[1], [3], [2]].
    # Because of broadcasting on both arguments this comparison results
    # in a matrix of size (len(lengths), maxlen)
    result = gen_math_ops._range(0, maxlen, 1) < expand_dims(lengths, 1)
    if dtype is None or result.dtype.base_dtype == dtype.base_dtype:
      return result
    else:
      return gen_math_ops.cast(result, dtype)
