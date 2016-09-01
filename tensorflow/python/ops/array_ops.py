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
@@saturate_cast

## Shapes and Shaping

TensorFlow provides several operations that you can use to determine the shape
of a tensor and change the shape of a tensor.

@@shape
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
@@space_to_batch
@@batch_to_space
@@space_to_depth
@@depth_to_space
@@gather
@@gather_nd
@@dynamic_partition
@@dynamic_stitch
@@boolean_mask
@@one_hot

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
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import logging_ops
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

def shape(input, name=None):
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

  Returns:
    A `Tensor` of type `int32`.
  """
  return shape_internal(input, name, optimize=True)


def shape_internal(input, name=None, optimize=True):
  """Returns the shape of a tensor.

  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).
    optimize: if true, encode the shape as a constant when possible.

  Returns:
    A `Tensor` of type `int32`.
  """
  with ops.name_scope(name, "Shape", [input]) as name:
    if isinstance(input, ops.SparseTensor):
      return gen_math_ops.cast(input.shape, dtypes.int32)
    else:
      input_tensor = ops.convert_to_tensor(input)
      input_shape = input_tensor.get_shape()
      if optimize and input_shape.is_fully_defined():
        return constant(input_shape.as_list(), dtypes.int32, name=name)
      return gen_array_ops.shape(input, name=name)


def size(input, name=None):
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

  Returns:
    A `Tensor` of type `int32`.
  """
  return size_internal(input, name, optimize=True)


def size_internal(input, name=None, optimize=True):
  """Returns the size of a tensor.

  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).
    optimize: if true, encode the size as a constant when possible.

  Returns:
    A `Tensor` of type `int32`.
  """
  with ops.name_scope(name, "Size", [input]) as name:
    if isinstance(input, ops.SparseTensor):
      return gen_math_ops._prod(gen_math_ops.cast(input.shape, dtypes.int32), 0,
                                name=name)
    else:
      input_tensor = ops.convert_to_tensor(input)
      input_shape = input_tensor.get_shape()
      if optimize and input_shape.is_fully_defined():
        return constant(input_shape.num_elements(), dtypes.int32, name=name)
      return gen_array_ops.size(input, name=name)


def rank(input, name=None):
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
  """Returns the rank of a tensor.

  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).
    optimize: if true, encode the rank as a constant when possible.

  Returns:
    A `Tensor` of type `int32`.
  """
  with ops.name_scope(name, "Rank", [input]) as name:
    if isinstance(input, ops.SparseTensor):
      return gen_array_ops.size(input.shape, name=name)
    else:
      input_tensor = ops.convert_to_tensor(input)
      input_shape = input_tensor.get_shape()
      if optimize and input_shape.ndims is not None:
        return constant(input_shape.ndims, dtypes.int32, name=name)
      return gen_array_ops.rank(input, name=name)


# DEPRECATED use init_ops.zeros_initializer
# TODO(irving) Move it to init_ops.py
def zeros_initializer(shape, dtype=dtypes.float32, partition_info=None):
  """An adaptor for zeros() to match the Initializer spec."""
  return zeros(shape, dtype)


def _SliceHelper(tensor, slice_spec, var=None):
  """Overload for Tensor.__getitem__.

  This operation extracts the specified region from the tensor.
  The notation is similar to numpy with the restriction that
  currently only support basic indexing. That means that
  using a tensor as input is not currently allowed

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


@ops.RegisterShape("Pack")
def _PackShape(op):
  input_shape = op.inputs[0].get_shape()
  if input_shape.ndims is None:
    return [tensor_shape.unknown_shape()]

  for inp in op.inputs[1:]:
    input_shape = input_shape.merge_with(inp.get_shape())

  input_shape = input_shape.as_list()
  axis = op.get_attr("axis")
  if axis < 0: axis += len(input_shape) + 1
  input_shape.insert(axis, len(op.inputs))
  return [tensor_shape.TensorShape(input_shape)]


ops.RegisterShape("Unpack")(common_shapes.call_cpp_shape_fn)

@ops.RegisterShape("Concat")
def _ConcatShape(op):
  concat_dim = tensor_util.constant_value(op.inputs[0])
  if concat_dim is None:
    # Return an unknown shape with the same rank as the inputs, or an
    # unknown rank if no input's rank is known.
    rank = None
    for value in op.inputs[1:]:
      if rank is not None:
        value.get_shape().assert_has_rank(rank)
      else:
        rank = value.get_shape().ndims
    if rank == 0:
      raise ValueError("Can't concatenate scalars (use tf.pack instead)")
    return [tensor_shape.unknown_shape(ndims=rank)]

  else:
    # Merge all the non-concat dims, and sum the concat dim to make an
    # output shape.
    concat_dim = int(concat_dim)
    if concat_dim < 0:
      raise ValueError("Expected concat_dim >= 0, but got %d" % concat_dim)

    output_shape = op.inputs[1].get_shape()
    for value in op.inputs[2:]:
      value_shape = value.get_shape()
      if value_shape.ndims is not None and concat_dim >= value_shape.ndims:
        raise ValueError("Expected concat_dim in range [0, %d), but got %d" %
                         (value_shape.ndims, concat_dim))
      before = output_shape[:concat_dim].merge_with(value_shape[:concat_dim])
      at = output_shape[concat_dim] + value_shape[concat_dim]
      after = output_shape[
          concat_dim + 1:].merge_with(value_shape[concat_dim + 1:])
      output_shape = before.concatenate(at).concatenate(after)
    return [output_shape]


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
def batch_matrix_transpose(a, name="batch_matrix_transpose"):
  """Transposes last two dimensions of batch matrix `a`.

  For example:

  ```python
  # Matrix with no batch dimension.
  # 'x' is [[1 2 3]
  #         [4 5 6]]
  tf.batch_matrixtranspose(x) ==> [[1 4]
                                   [2 5]
                                   [3 6]]

  # Matrix with two batch dimensions.
  # x.shape is [1, 2, 3, 4]
  # tf.batch_matrix_transpose(x) is shape [1, 2, 4, 3]
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
  tf.zeros([3, 4], int32) ==> [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
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
  tf.ones([2, 3], int32) ==> [[1, 1, 1], [1, 1, 1]]
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

    asserts = [logging_ops.Assert(
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


@ops.RegisterShape("Placeholder")
def _PlaceholderShape(op):
  given_shape = tensor_util.TensorShapeProtoToList(op.get_attr("shape"))
  if given_shape:
    return [tensor_shape.TensorShape(given_shape)]
  else:
    return [tensor_shape.unknown_shape()]


ops.RegisterShape("CheckNumerics")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Identity")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("RefIdentity")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("StopGradient")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("BatchMatrixBandPart")(common_shapes.call_cpp_shape_fn)
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
  """Shape function for gradient of array_ops.slice."""
  return [tensor_util.constant_value(op.inputs[0])]


ops.RegisterShape("StridedSliceAssign")(common_shapes.unchanged_shape)

@ops.RegisterShape("StridedSlice")
def _StridedSliceShape(op):
  """Shape function for array_ops.slice."""

  input_shape = op.inputs[0].get_shape()
  if input_shape.ndims is None:
    return [tensor_shape.unknown_shape()]

  ndims = len(input_shape)
  begin_shape = op.inputs[1].get_shape().with_rank(1)
  end_shape = op.inputs[2].get_shape().with_rank(1)
  strides_shape = op.inputs[3].get_shape().with_rank(1)
  # get constant values if available
  begin_value = tensor_util.constant_value(op.inputs[1])
  end_value = tensor_util.constant_value(op.inputs[2])
  strides_value = tensor_util.constant_value(op.inputs[3])

  sparse_dims = begin_shape.merge_with(end_shape).merge_with(strides_shape)[
      0].value
  if (sparse_dims is None or begin_value is None or end_value is None or
      strides_value is None):
    return [tensor_shape.unknown_shape()]

  begin_mask = op.get_attr("begin_mask")
  end_mask = op.get_attr("end_mask")
  ellipsis_mask = op.get_attr("ellipsis_mask")
  new_axis_mask = op.get_attr("new_axis_mask")
  shrink_axis_mask = op.get_attr("shrink_axis_mask")
  # find the ellipsis
  ellipsis_index = -1

  # look for ellipses
  num_add_axis_after_ellipsis = 0
  for i in range(sparse_dims):
    if ellipsis_index != -1 and ((1 << i) & new_axis_mask) != 0:
      num_add_axis_after_ellipsis += 1
    if (1 << i) & ellipsis_mask:
      if ellipsis_index != -1:
        raise ValueError("Multiple ellipses not allowed")
      ellipsis_index = i
  # insert a virtual ellipsis if not seen
  if ellipsis_index == -1:
    ellipsis_mask |= (1 << sparse_dims)
    sparse_dims += 1

  # build the dense specification
  dense_dims = ndims  # not accounting for newaxis and shrink
  final_shape_gather = []
  full_index = 0
  dense_shrink_axis = 0
  dense_specs = []
  for dim in range(sparse_dims):
    bit = 1 << dim
    if bit & ellipsis_mask:
      next_index = min(dense_dims -
                       (sparse_dims - dim) + 1 + num_add_axis_after_ellipsis,
                       dense_dims)
      while full_index < next_index:
        dense_specs.append(_baseslice(None, None, 1))
        final_shape_gather.append(full_index)
        full_index += 1
    elif bit & new_axis_mask:
      final_shape_gather.append(NEW_AXIS)
    else:
      dense_specs.append(_baseslice(
          None if (begin_mask & bit) else begin_value[dim], None if (
              end_mask & bit) else end_value[dim], strides_value[dim]))
      if shrink_axis_mask & bit:
        dense_shrink_axis |= (1 << full_index)
        final_shape_gather.append(SHRINK_AXIS)
      else:
        final_shape_gather.append(full_index)

      full_index += 1

  # Compute each dimensions contribution to the "processing" shape
  final_dims = []
  for dim in range(dense_dims):
    shrink = (dense_shrink_axis & (1 << dim)) != 0
    final_dims.append(
        _compute_size_of_strided_dim(shrink, dense_specs[dim], input_shape.dims[
            dim]))

  # Gather the final shape from the processing shape
  final_shape = []
  for index in final_shape_gather:
    if index == NEW_AXIS:
      final_shape.append(1)
    elif index == SHRINK_AXIS:
      pass
    else:
      final_shape.append(final_dims[index])

  return [tensor_shape.TensorShape(final_shape)]


ops.RegisterShape("Gather")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("GatherNd")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Unique")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("UniqueWithCounts")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("BatchMatrixDiag")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("BatchMatrixSetDiag")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("BatchMatrixDiagPart")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Diag")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("DiagPart")(common_shapes.call_cpp_shape_fn)

@ops.RegisterShape("ExpandDims")
def _ExpandDimsShape(op):
  """Determine shape for expand op's output tensor.

  Args:
    op: Operation for which to determine shape.
        op.inputs[0] is the input tensor.
        op.inputs[1] is the dimension in which to expand.
  Returns:
    Shape of op's output tensor.
  Raises:
    ValueError: If dim is outside of [-rank - 1, rank], where rank is the number
        of dimensions in the input tensor.
  """
  input_shape = op.inputs[0].get_shape()
  if input_shape.dims is None:
    return [tensor_shape.unknown_shape()]
  dim = tensor_util.constant_value(op.inputs[1])
  input_ndims = input_shape.ndims
  if dim < -input_ndims - 1 or dim > input_ndims:
    raise ValueError(
        "dim %d not in [%d, %d]." % (dim, -input_ndims, input_ndims))
  if dim < 0:
    dim += (input_ndims + 1)
  result_shape = list(input_shape.dims)
  result_shape.insert(dim, 1)
  return [tensor_shape.TensorShape(result_shape)]


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
  """Shape function for the Pad op.

  This op has two inputs:

  * input: A rank-N tensor.
  * paddings: An N-by-2 matrix, in which the i^th row contains the
    number of padding elements to add before and after `input` in the
    i^th dimension.

  It has one output, which has the same rank as input, and additional
  elements according to the values in paddings.

  Args:
    op: A Pad Operation.

  Returns:
    A single-element list containing the shape of the output.

  Raises:
    ValueError: If the input shapes are incompatible.
  """
  paddings_shape = op.inputs[1].get_shape().with_rank(2)
  input_shape = op.inputs[0].get_shape()
  input_shape = input_shape.with_rank(paddings_shape[0].value)
  paddings_shape = paddings_shape.merge_with(
      tensor_shape.matrix(input_shape.ndims, 2))
  paddings = tensor_util.constant_value(op.inputs[1])
  if paddings is None:
    return [tensor_shape.unknown_shape(ndims=input_shape.ndims)]
  else:
    output_dims = []
    for i, dim in enumerate(input_shape.dims):
      if paddings[i, 0] < 0 or paddings[i, 1] < 0:
        raise ValueError("paddings must be non-negative")
      output_dims.append(dim + paddings[i, 0] + paddings[i, 1])
    return [tensor_shape.TensorShape(output_dims)]


@ops.RegisterShape("MirrorPadGrad")
def _MirrorPadGradShape(op):
  """Shape function for the MirrorPadGrad op."""
  paddings_shape = op.inputs[1].get_shape().with_rank(2)
  input_shape = op.inputs[0].get_shape().with_rank(paddings_shape[0].value)
  paddings_shape = paddings_shape.merge_with(tensor_shape.matrix(
      input_shape.ndims, 2))
  paddings = tensor_util.constant_value(op.inputs[1])
  if paddings is None:
    return [tensor_shape.unknown_shape(ndims=input_shape.ndims)]

  output_dims = []
  for i, dim in enumerate(input_shape.dims):
    if paddings[i, 0] < 0 or paddings[i, 1] < 0:
      raise ValueError("Paddings must be non-negative.")
    if dim < paddings[i, 0] + paddings[i, 1]:
      raise ValueError("Output dimension is negative.")
    output_dims.append(dim - paddings[i, 0] - paddings[i, 1])
  return [tensor_shape.TensorShape(output_dims)]


ops.RegisterShape("ReverseSequence")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("Shape")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("ShapeN")(common_shapes.call_cpp_shape_fn)


@ops.RegisterShape("Transpose")
def _TransposeShape(op):
  """Shape function for the Transpose op.

  This op takes two inputs:

  * input: a rank-N tensor of arbitrary shape.
  * shuffle: a length-N vector.

  Its output is the rank-N tensor computed by permuting the dimensions
  of input according to shuffle.

  Args:
    op: A Transpose op.

  Returns:
    A single-element list containing the shape of the output.

  Raises:
    ValueError: If the shapes of input and shuffle are incompatible.
    IndexError: If shuffle contains an index that is >= the rank of input.
  """
  input_shape = op.inputs[0].get_shape()
  transpose_shape = op.inputs[1].get_shape().merge_with(tensor_shape.vector(
      input_shape.ndims))
  transpose_vec = tensor_util.constant_value(op.inputs[1])
  if transpose_vec is None:
    return [tensor_shape.unknown_shape(ndims=transpose_shape[0].value)]
  else:
    return [tensor_shape.TensorShape([input_shape[i]
                                      for i in transpose_vec.tolist()])]


@ops.RegisterShape("Split")
def _SplitShape(op):
  """Shape function for the Split op."""
  split_dim = tensor_util.constant_value(op.inputs[0])
  num_split = len(op.outputs)
  input_shape = op.inputs[1].get_shape()
  if split_dim is None:
    return [tensor_shape.unknown_shape(ndims=input_shape.ndims)] * num_split
  else:
    split_dim = int(split_dim)
    input_shape = input_shape.with_rank_at_least(split_dim + 1)
    if not (input_shape[split_dim] % num_split).is_compatible_with(0):
      raise ValueError(
          "Number of ways to split should evenly divide the split "
          "dimension but got split_dim %d (size = %d) and num_split %d" %
          (split_dim, input_shape[split_dim].value, num_split))
    prefix = input_shape[:split_dim]
    size_in_split_dim = input_shape[split_dim] // num_split
    suffix = input_shape[split_dim + 1:]
    output_shape = prefix.concatenate(size_in_split_dim).concatenate(suffix)
    return [output_shape] * num_split


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
  if not isinstance(hypothesis, ops.SparseTensor):
    raise TypeError("Hypothesis must be a SparseTensor")
  if not isinstance(truth, ops.SparseTensor):
    raise TypeError("Truth must be a SparseTensor")

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
  """Shape function for the EditDistance op."""
  hypothesis_shape = tensor_util.constant_value(op.inputs[2])
  truth_shape = tensor_util.constant_value(op.inputs[5])
  if hypothesis_shape is not None and truth_shape is not None:
    if len(hypothesis_shape) != len(truth_shape):
      raise ValueError(
          "Inconsistent ranks in hypothesis and truth.  Saw shapes: %s and %s" %
          (str(hypothesis_shape), str(truth_shape)))
    return [tensor_shape.TensorShape(
        [max(h, t) for h, t in zip(hypothesis_shape[:-1], truth_shape[:-1])])]

  return [tensor_shape.unknown_shape()]


# The remaining ops do not change the shape of their inputs.
@ops.RegisterShape("Quantize")
@ops.RegisterShape("Dequantize")
def _QuantizeDequantizeShape(op):
  unused_min_range = op.inputs[1].get_shape().merge_with(tensor_shape.scalar())
  unused_max_range = op.inputs[2].get_shape().merge_with(tensor_shape.scalar())
  return common_shapes.unchanged_shape(op)


@ops.RegisterShape("ExtractImagePatches")
def _ExtractImagePatchesShape(op):
  """Shape function for the ExtractImagePatches op.

  Args:
    op: An ExtractImagePatches op.

  Raises:
    ValueError: If the strides or padding are invalid.

  Returns:
    The shape of the op output.
  """
  images_shape = op.inputs[0].get_shape().with_rank(4)
  batch = images_shape[0]
  in_rows = images_shape[1]
  in_cols = images_shape[2]
  in_depth = images_shape[3]

  ksize_b, ksize_r, ksize_c, ksize_d = op.get_attr("ksizes")
  if ksize_b != 1 or ksize_d != 1:
    raise ValueError("Current implementation does not yet support "
                     "ksizes in the batch and depth dimensions.")

  stride_b, stride_r, stride_c, stride_d = op.get_attr("strides")
  if stride_b != 1 or stride_d != 1:
    raise ValueError("Current implementation does not yet support "
                     "strides in the batch and depth dimensions.")

  rate_b, rate_r, rate_c, rate_d = op.get_attr("rates")
  if rate_b != 1 or rate_d != 1:
    raise ValueError("Current implementation does not yet support "
                     "rates in the batch and depth dimensions.")

  # Effective patch size, taking into account filter upsampling by rates.
  ksize_r_eff = ksize_r + (ksize_r - 1) * (rate_r - 1)
  ksize_c_eff = ksize_c + (ksize_c - 1) * (rate_c - 1)

  padding = op.get_attr("padding")
  out_rows, out_cols = common_shapes.get2d_conv_output_size(in_rows, in_cols,
                                                            ksize_r_eff,
                                                            ksize_c_eff,
                                                            stride_r, stride_c,
                                                            padding)

  out_depth = None if in_depth is None else ksize_r * ksize_c * int(in_depth)
  output_shape = [batch, out_rows, out_cols, out_depth]

  return [tensor_shape.TensorShape(output_shape)]


@ops.RegisterShape("SpaceToBatch")
def _SpaceToBatchShape(op):
  """Shape function for the SpaceToBatch op.

  The output shape is determined by the following inputs/ attributes:

  * input: A rank-4 tensor with shape [B, H, W, D]
  * paddings: A 2-by-2 matrix, specified as follows:

        paddings = [[pad_top, pad_bottom], [pad_left, pad_right]],

    implying effective padded spatial dimensions:

        Hp = pad_top + H + pad_bottom
        Wp = pad_left + W + pad_right

    Both Hp and Wp must be multiples of block_size.
  * block_size: an int.

  Its output is also a rank-4 tensor with shape:

      [B*block_size*block_size, Hp/block_size, Wp/block_size, D]

  Args:
    op: A SpaceToBatch op.

  Returns:
    A single-element list containing the shape of the output.

  Raises:
    ValueError: If the shapes of inputs are not as expected.
    IndexError: If block_size does not divide Wp or Hp.
  """
  # Check that the input tensor is 4-D.
  try:
    input_shape = op.inputs[0].get_shape().with_rank(4)
  except ValueError:
    raise ValueError(
        "tf.space_to_batch() requires 4-D input tensor.")

  # Check that the paddings tensor is a matrix with shape [2, 2].
  try:
    paddings_shape = op.inputs[1].get_shape().with_rank(2)
  except ValueError:
    raise ValueError(
        "tf.space_to_batch() requires 2-D paddings tensor.")

  if paddings_shape[0] != 2 or paddings_shape[1] != 2:
    raise ValueError(
        "tf.space_to_batch() requires input paddings with shape [2, 2].")

  block_size = op.get_attr("block_size")
  if block_size <= 1:
    raise ValueError("Attribute block_size has to be > 1.")

  paddings = tensor_util.constant_value(op.inputs[1])
  if paddings is not None:
    if (paddings[0, 0] < 0 or paddings[0, 1] < 0 or
        paddings[1, 0] < 0 or paddings[1, 1] < 0):
      raise ValueError("paddings cannot be negative.")

    input_height = input_shape[1] + paddings[0, 0] + paddings[0, 1]
    input_width = input_shape[2] + paddings[1, 0] + paddings[1, 1]

    if input_height % block_size > 0 or input_width % block_size > 0:
      raise IndexError("block_size needs to divide both width and height.")
  else:
    input_height = tensor_shape.Dimension(None)
    input_width = tensor_shape.Dimension(None)

  batch = input_shape[0] * block_size * block_size
  height = input_height // block_size
  width = input_width // block_size
  depth = input_shape[3]

  return [tensor_shape.TensorShape([batch, height, width, depth])]


@ops.RegisterShape("BatchToSpace")
def _BatchToSpaceShape(op):
  """Shape function for the BatchToSpace op.

  The output shape is determined by the following inputs/ attributes:

  * input: A rank-4 tensor with shape

        [B*block_size*block_size, Hp/block_size, Wp/block_size, D]

    Note that the batch size of the input tensor must be divisible by
    `block_size * block_size`.
  * crops: A 2-by-2 matrix, specified as follows:

        crops = [[crop_top, crop_bottom], [crop_left, crop_right]].

  * block_size: an int.

  Its output is also a rank-4 tensor with shape [B, H, W, D], where:

      H = Hp - crop_top - crop_bottom
      W = Wp - crop_left - crop_right

  Args:
    op: A BatchToSpace op.

  Returns:
    A single-element list containing the shape of the output.

  Raises:
    ValueError: If the shapes of the inputs are not as expected.
    IndexError: If block_size*block_size does not divide the input batch size.
  """
  # Check that the input tensor is 4-D.
  try:
    input_shape = op.inputs[0].get_shape().with_rank(4)
  except ValueError:
    raise ValueError("tf.batch_to_space() requires 4-D input tensor.")

  # Check that the crops tensor is a matrix with shape [2, 2].
  try:
    crops_shape = op.inputs[1].get_shape().with_rank(2)
  except ValueError:
    raise ValueError(
        "tf.space_to_batch() requires 2-D crops tensor.")

  if crops_shape[0] != 2 or crops_shape[1] != 2:
    raise ValueError(
        "tf.space_to_batch() requires input crops with shape [2, 2].")

  crops = tensor_util.constant_value(op.inputs[1])
  if (crops is not None and
      (crops[0, 0] < 0 or crops[0, 1] < 0 or
       crops[1, 0] < 0 or crops[1, 1] < 0)):
    raise ValueError("crops cannot be negative.")

  block_size = op.get_attr("block_size")
  if block_size <= 1:
    raise ValueError("Attribute block_size has to be > 1.")

  input_batch = input_shape[0]
  if input_batch % (block_size * block_size) > 0:
    raise IndexError("input batch must be divisible by block_size*block_size.")
  batch = input_batch // (block_size * block_size)

  if crops is not None:
    height = input_shape[1] * block_size - crops[0, 0] - crops[0, 1]
    width = input_shape[2] * block_size - crops[1, 0] - crops[1, 1]
    if height <= 0 or width <= 0:
      raise ValueError("Output height or width is not positive.")
  else:
    height = tensor_shape.Dimension(None)
    width = tensor_shape.Dimension(None)
  depth = input_shape[3]

  return [tensor_shape.TensorShape([batch, height, width, depth])]


@ops.RegisterShape("SpaceToDepth")
def _SpaceToDepthShape(op):
  """Shape function for the SpaceToDepth op.

  This op takes two inputs:

  * input: a tensor of shape like that [B, H, W, D]
  * block_size: an int.

  Its output is the same-rank tensor but with changed
  dimensions like that: [B, H/block_size, W/block_size, D*block_size*block_size]

  Args:
    op: A SpaceToDepth op.

  Returns:
    A single-element list containing the shape of the output.

  Raises:
    ValueError: If the shapes of input are not as expected.
    IndexError: If block_size does not divide W or H.
  """
  # Check that the input tensor is of 4 dimensions.
  try:
    input_shape = op.inputs[0].get_shape().with_rank(4)
  except ValueError:
    raise ValueError(
        "tf.space_to_depth() requires tensors with exactly 4 dimensions.")

  block_size = op.get_attr("block_size")
  if block_size <= 1:
    raise ValueError("Attribute block_size has to be > 1.")

  input_height = input_shape[1]
  input_width = input_shape[2]

  if (input_width % block_size > 0) or (input_height % block_size > 0):
    raise IndexError(
        "block_size needs to divide both width and height.")

  width = input_width // block_size
  height = input_height // block_size
  new_depth = input_shape[3] * block_size * block_size

  return [tensor_shape.TensorShape(
      [input_shape[0], height, width, new_depth])]


@ops.RegisterShape("DepthToSpace")
def _DepthToSpaceShape(op):
  """Shape function for the DepthToSpace op.

  This op takes two inputs:

  * input: a tensor of shape like that [B, H, W, D]
  * block_size: an int.

  Its output is the same-rank tensor but with changed
  dimensions like that:
      [B, H*block_size, W*block_size, D/(block_size*block_size)]

  Args:
    op: A DepthToSpace op.

  Returns:
    A single-element list containing the shape of the output.

  Raises:
    ValueError: If the shapes of input are not as expected.
    IndexError: If block_size*block_size does not divide D.
  """
  # Check that the input tensor is of 4 dimensions.
  try:
    input_shape = op.inputs[0].get_shape().with_rank(4)
  except ValueError:
    raise ValueError(
        "tf.depth_to_space() requires tensors with exactly 4 dimensions.")

  block_size = op.get_attr("block_size")
  if block_size <= 1:
    raise ValueError("Attribute block_size has to be > 1.")

  input_height = input_shape[1]
  input_width = input_shape[2]
  input_depth = input_shape[3]

  width = input_width * block_size
  height = input_height * block_size

  if input_depth % (block_size * block_size) > 0:
    raise IndexError(
        "block_size*block_size needs to divide the input depth.")

  new_depth = input_depth // (block_size * block_size)
  return [tensor_shape.TensorShape(
      [input_shape[0], height, width, new_depth])]


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
  """Shape function for the OneHot op.

  It closely follows the code in the .cc implementation.

  Args:
    op: A OneHot Operation.

  Returns:
    A single-element list containing the shape of the output.

  Raises:
    ValueError: if axis < -1.
  """
  indices_shape = op.inputs[0].get_shape()
  indices_dims = indices_shape.ndims
  depth = tensor_util.constant_value(op.inputs[1])
  axis = op.get_attr("axis")

  if axis < -1:
    raise ValueError("axis must be >= -1")

  new_shape = None
  if indices_dims is not None:
    new_shape = indices_shape.as_list()
    new_shape.insert(axis % (indices_dims + 1), depth)

  return [tensor_shape.TensorShape(new_shape)]


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
