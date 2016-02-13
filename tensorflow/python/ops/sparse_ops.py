# Copyright 2015 Google Inc. All Rights Reserved.
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

# pylint: disable=g-short-docstring-punctuation
"""## Sparse Tensor Representation

Tensorflow supports a `SparseTensor` representation for data that is sparse
in multiple dimensions. Contrast this representation with `IndexedSlices`,
which is efficient for representing tensors that are sparse in their first
dimension, and dense along all other dimensions.

@@SparseTensor
@@SparseTensorValue

## Sparse to Dense Conversion

@@sparse_to_dense
@@sparse_tensor_to_dense
@@sparse_to_indicator

## Manipulation

@@sparse_concat
@@sparse_reorder
@@sparse_split
@@sparse_retain
@@sparse_fill_empty_rows
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_sparse_ops import *

# pylint: enable=wildcard-import
# pylint: disable=protected-access


def sparse_concat(concat_dim, sp_inputs, name=None):
  """Concatenates a list of `SparseTensor` along the specified dimension.

  Concatenation is with respect to the dense versions of each sparse input.
  It is assumed that each inputs is a `SparseTensor` whose elements are ordered
  along increasing dimension number.

  All inputs' shapes must match, except for the concat dimension.  The
  `indices`, `values`, and `shapes` lists must have the same length.

  The output shape is identical to the inputs', except along the concat
  dimension, where it is the sum of the inputs' sizes along that dimension.

  The output elements will be resorted to preserve the sort order along
  increasing dimension number.

  This op runs in `O(M log M)` time, where `M` is the total number of non-empty
  values across all inputs. This is due to the need for an internal sort in
  order to concatenate efficiently across an arbitrary dimension.

  For example, if `concat_dim = 1` and the inputs are

      sp_inputs[0]: shape = [2, 3]
      [0, 2]: "a"
      [1, 0]: "b"
      [1, 1]: "c"

      sp_inputs[1]: shape = [2, 4]
      [0, 1]: "d"
      [0, 2]: "e"

  then the output will be

      shape = [2, 7]
      [0, 2]: "a"
      [0, 4]: "d"
      [0, 5]: "e"
      [1, 0]: "b"
      [1, 1]: "c"

  Graphically this is equivalent to doing

      [    a] concat [  d e  ] = [    a   d e  ]
      [b c  ]        [       ]   [b c          ]

  Args:
    concat_dim: Dimension to concatenate along.
    sp_inputs: List of `SparseTensor` to concatenate.
    name: A name prefix for the returned tensors (optional).

  Returns:
    A `SparseTensor` with the concatenated output.

  Raises:
    TypeError: If `sp_inputs` is not a list of `SparseTensor`.
  """
  if not isinstance(sp_inputs, list):
    raise TypeError("Inputs must be a list")
  if not all(isinstance(sp_input, ops.SparseTensor) for sp_input in sp_inputs):
    raise TypeError("All inputs must be SparseTensors")

  if len(sp_inputs) == 1:  # Degenerate case of one tensor.
    return sp_inputs[0]

  inds = [sp_input.indices for sp_input in sp_inputs]
  vals = [sp_input.values for sp_input in sp_inputs]
  shapes = [sp_input.shape for sp_input in sp_inputs]

  output_ind, output_val, output_shape = (
      gen_sparse_ops._sparse_concat(inds,
                                    vals,
                                    shapes,
                                    concat_dim,
                                    name=name))

  return ops.SparseTensor(output_ind, output_val, output_shape)


@ops.RegisterShape("SparseConcat")
def _SparseConcatShape(op):
  """Shape function for SparseConcat op."""
  num_inputs = int(op.get_attr("N"))

  # TF flattens and concatenates all list inputs, so reconstruct the lists here.
  ind_shapes = [ind.get_shape().with_rank(2) for ind in op.inputs[0:num_inputs]]
  val_shapes = [val.get_shape().with_rank(1)
                for val in op.inputs[num_inputs:2 * num_inputs]]
  shape_shapes = [shape.get_shape().with_rank(1)
                  for shape in op.inputs[2 * num_inputs:]]

  output_ind_rows = tensor_shape.Dimension(0)
  output_ind_cols = tensor_shape.Dimension(None)
  output_val_elems = tensor_shape.Dimension(0)
  output_shape_shape = tensor_shape.TensorShape(None)

  for i in xrange(num_inputs):
    num_elems_i = ind_shapes[i][0].merge_with(val_shapes[i][0])
    output_ind_rows += num_elems_i
    output_ind_cols = output_ind_cols.merge_with(ind_shapes[i][1])
    output_val_elems += num_elems_i
    output_shape_shape = output_shape_shape.merge_with(shape_shapes[i])

  output_ind_shape = tensor_shape.matrix(output_ind_rows, output_ind_cols)
  output_val_shape = tensor_shape.vector(output_val_elems)

  return [output_ind_shape, output_val_shape, output_shape_shape]


def sparse_reorder(sp_input, name=None):
  """Reorders a `SparseTensor` into the canonical, row-major ordering.

  Note that by convention, all sparse ops preserve the canonical ordering
  along increasing dimension number. The only time ordering can be violated
  is during manual manipulation of the indices and values to add entries.

  Reordering does not affect the shape of the `SparseTensor`.

  For example, if `sp_input` has shape `[4, 5]` and `indices` / `values`:

      [0, 3]: b
      [0, 1]: a
      [3, 1]: d
      [2, 0]: c

  then the output will be a `SparseTensor` of shape `[4, 5]` and
  `indices` / `values`:

      [0, 1]: a
      [0, 3]: b
      [2, 0]: c
      [3, 1]: d

  Args:
    sp_input: The input `SparseTensor`.
    name: A name prefix for the returned tensors (optional)

  Returns:
    A `SparseTensor` with the same shape and non-empty values, but in
    canonical ordering.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  if not isinstance(sp_input, ops.SparseTensor):
    raise TypeError("Input must be a SparseTensor")

  reordered_ind, reordered_val = (
      gen_sparse_ops._sparse_reorder(sp_input.indices,
                                     sp_input.values,
                                     sp_input.shape,
                                     name=name))

  return ops.SparseTensor(reordered_ind, reordered_val,
                          array_ops.identity(sp_input.shape))


@ops.RegisterShape("SparseReorder")
def _SparseReorderShape(op):
  """Shape function for SparseReorder op."""
  input_indices_shape = op.inputs[0].get_shape().with_rank(2)
  input_values_shape = op.inputs[1].get_shape().with_rank(1)
  unused_shape_shape = op.inputs[2].get_shape().with_rank(1)

  return [input_indices_shape, input_values_shape]


def sparse_split(split_dim, num_split, sp_input, name=None):
  """Split a `SparseTensor` into `num_split` tensors along `split_dim`.

  If the `sp_input.shape[split_dim]` is not an integer multiple of `num_split`
  each slice starting from 0:`shape[split_dim] % num_split` gets extra one
  dimension. For example, if `split_dim = 1` and `num_split = 2` and the
  input is:

      input_tensor = shape = [2, 7]
      [    a   d e  ]
      [b c          ]

  Graphically the output tensors are:

      output_tensor[0] =
      [    a ]
      [b c   ]

      output_tensor[1] =
      [ d e  ]
      [      ]

  Args:
    split_dim: A 0-D `int32` `Tensor`. The dimension along which to split.
    num_split: A Python integer. The number of ways to split.
    sp_input: The `SparseTensor` to split.
    name: A name for the operation (optional).

  Returns:
    `num_split` `SparseTensor` objects resulting from splitting `value`.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  if not isinstance(sp_input, ops.SparseTensor):
    raise TypeError("Input must be a SparseTensor")

  output_inds, output_vals, output_shapes = (
      gen_sparse_ops._sparse_split(split_dim,
                                   sp_input.indices,
                                   sp_input.values,
                                   sp_input.shape,
                                   num_split,
                                   name=name))
  sparse_tensors = []
  for i in range(0, num_split):
    sparse_tensors.append(ops.SparseTensor(output_inds[i], output_vals[i],
                                           output_shapes[i]))
  return sparse_tensors


# pylint: disable=invalid-name
@ops.RegisterShape("SparseSplit")
def _SparseSplitShape(op):
  """Shape function for SparseSplit op."""
  num_split = int(op.get_attr("num_split"))
  input_shape_shape = op.inputs[3].get_shape()
  dim = input_shape_shape.num_elements()
  output_indices_shape = tensor_shape.TensorShape([None, dim])
  output_values_shape = tensor_shape.unknown_shape(1)
  output_indices_shape = [output_indices_shape] * num_split
  output_values_shape = [output_values_shape] * num_split
  output_shape_shape = [input_shape_shape] * num_split
  return output_indices_shape + output_values_shape + output_shape_shape
# pylint: enable=invalid-name


@ops.RegisterShape("SparseToDense")
def _SparseToDenseShape(op):
  input_shape = tensor_util.constant_value(op.inputs[1])
  if input_shape is not None:
    if np.ndim(input_shape) > 1:
      raise ValueError("Input shape should be a vector")
    return [tensor_shape.TensorShape(input_shape)]
  else:
    input_shape_shape = op.inputs[1].get_shape().with_rank(1)
    return [tensor_shape.unknown_shape(ndims=input_shape_shape[0].value)]


def sparse_to_dense(sparse_indices,
                    output_shape,
                    sparse_values,
                    default_value=0,
                    validate_indices=True,
                    name=None):
  """Converts a sparse representation into a dense tensor.

  Builds an array `dense` with shape `output_shape` such that

  ```python
  # If sparse_indices is scalar
  dense[i] = (i == sparse_indices ? sparse_values : default_value)

  # If sparse_indices is a vector, then for each i
  dense[sparse_indices[i]] = sparse_values[i]

  # If sparse_indices is an n by d matrix, then for each i in [0, n)
  dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
  ```

  All other values in `dense` are set to `default_value`.  If `sparse_values`
  is a scalar, all sparse indices are set to this single value.

  Indices should be sorted in lexicographic order, and indices must not
  contain any repeats. If `validate_indices` is True, these properties
  are checked during execution.

  Args:
    sparse_indices: A 0-D, 1-D, or 2-D `Tensor` of type `int32` or `int64`.
      `sparse_indices[i]` contains the complete index where `sparse_values[i]`
      will be placed.
    output_shape: A 1-D `Tensor` of the same type as `sparse_indices`.  Shape
      of the dense output tensor.
    sparse_values: A 0-D or 1-D `Tensor`.  Values corresponding to each row of
      `sparse_indices`, or a scalar value to be used for all sparse indices.
    default_value: A 0-D `Tensor` of the same type as `sparse_values`.  Value
      to set for indices not specified in `sparse_indices`.  Defaults to zero.
    validate_indices: A boolean value.  If True, indices are checked to make
      sure they are sorted in lexicographic order and that there are no repeats.
    name: A name for the operation (optional).

  Returns:
    Dense `Tensor` of shape `output_shape`.  Has the same type as
    `sparse_values`.
  """
  return gen_sparse_ops._sparse_to_dense(sparse_indices,
                                         output_shape,
                                         sparse_values,
                                         default_value=default_value,
                                         validate_indices=validate_indices,
                                         name=name)


def sparse_tensor_to_dense(sp_input,
                           default_value=0,
                           validate_indices=True,
                           name=None):
  """Converts a `SparseTensor` into a dense tensor.

  This op is a convenience wrapper around `sparse_to_dense` for `SparseTensor`s.

  For example, if `sp_input` has shape `[3, 5]` and non-empty string values:

      [0, 1]: a
      [0, 3]: b
      [2, 0]: c

  and `default_value` is `x`, then the output will be a dense `[3, 5]`
  string tensor with values:

      [[x a x b x]
       [x x x x x]
       [c x x x x]]

  Indices must be without repeats.  This is only
  tested if validate_indices is True.

  Args:
    sp_input: The input `SparseTensor`.
    default_value: Scalar value to set for indices not specified in
      `sp_input`.  Defaults to zero.
    validate_indices: A boolean value.  If `True`, indices are checked to make
      sure they are sorted in lexicographic order and that there are no repeats.
    name: A name prefix for the returned tensors (optional).

  Returns:
    A dense tensor with shape `sp_input.shape` and values specified by
    the non-empty values in `sp_input`. Indices not in `sp_input` are assigned
    `default_value`.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  if not isinstance(sp_input, ops.SparseTensor):
    raise TypeError("Input must be a SparseTensor")

  return sparse_to_dense(sp_input.indices,
                         sp_input.shape,
                         sp_input.values,
                         default_value=default_value,
                         validate_indices=validate_indices,
                         name=name)


def sparse_to_indicator(sp_input, vocab_size, name=None):
  """Converts a `SparseTensor` of ids into a dense bool indicator tensor.

  The last dimension of `sp_input` is discarded and replaced with the values of
  `sp_input`.  If `sp_input.shape = [D0, D1, ..., Dn, K]`, then
  `output.shape = [D0, D1, ..., Dn, vocab_size]`, where

      output[d_0, d_1, ..., d_n, sp_input[d_0, d_1, ..., d_n, k]] = True

  and False elsewhere in `output`.

  For example, if `sp_input.shape = [2, 3, 4]` with non-empty values:

      [0, 0, 0]: 0
      [0, 1, 0]: 10
      [1, 0, 3]: 103
      [1, 1, 2]: 150
      [1, 1, 3]: 149
      [1, 1, 4]: 150
      [1, 2, 1]: 121

  and `vocab_size = 200`, then the output will be a `[2, 3, 200]` dense bool
  tensor with False everywhere except at positions

      (0, 0, 0), (0, 1, 10), (1, 0, 103), (1, 1, 149), (1, 1, 150),
      (1, 2, 121).

  Note that repeats are allowed in the input SparseTensor.
  This op is useful for converting `SparseTensor`s into dense formats for
  compatibility with ops that expect dense tensors.

  The input `SparseTensor` must be in row-major order.

  Args:
    sp_input: A `SparseTensor` of type `int32` or `int64`.
    vocab_size: The new size of the last dimension, with
      `all(0 <= sp_input.values < vocab_size)`.
    name: A name prefix for the returned tensors (optional)

  Returns:
    A dense bool indicator tensor representing the indices with specified value.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  if not isinstance(sp_input, ops.SparseTensor):
    raise TypeError("Input must be a SparseTensor")

  with ops.op_scope([sp_input], name, "SparseToIndicator") as name:
    indices_shape = array_ops.shape(sp_input.indices)
    num_entries = indices_shape[0]
    rank = indices_shape[1]

    ids = sp_input.values
    if ids.dtype != dtypes.int64:
      ids = math_ops.cast(ids, dtypes.int64)

    # Slice off the last dimension of indices, then then tack on the ids
    indices_columns_to_preserve = array_ops.slice(
        sp_input.indices, [0, 0], array_ops.pack([-1, rank - 1]))
    new_indices = array_ops.concat(1, [indices_columns_to_preserve,
                                       array_ops.reshape(ids, [-1, 1])])

    new_values = array_ops.fill(array_ops.expand_dims(num_entries, 0), True)
    new_shape = array_ops.concat(0, [array_ops.slice(
        sp_input.shape, [0], array_ops.expand_dims(rank - 1, 0)), [vocab_size]])

    sp_new = ops.SparseTensor(new_indices, new_values, new_shape)

    # validate_indices may be False because we allow duplicates in new_indices:
    # repeated indices are allowed when creating an indicator matrix.
    return sparse_tensor_to_dense(
        sp_new, default_value=False, validate_indices=False, name=name)


def sparse_retain(sp_input, to_retain):
  """Retains specified non-empty values within a `SparseTensor`.

  For example, if `sp_input` has shape `[4, 5]` and 4 non-empty string values:

      [0, 1]: a
      [0, 3]: b
      [2, 0]: c
      [3, 1]: d

  and `to_retain = [True, False, False, True]`, then the output will
  be a `SparseTensor` of shape `[4, 5]` with 2 non-empty values:

      [0, 1]: a
      [3, 1]: d

  Args:
    sp_input: The input `SparseTensor` with `N` non-empty elements.
    to_retain: A bool vector of length `N` with `M` true values.

  Returns:
    A `SparseTensor` with the same shape as the input and `M` non-empty
    elements corresponding to the true positions in `to_retain`.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  if not isinstance(sp_input, ops.SparseTensor):
    raise TypeError("Input must be a SparseTensor")

  to_retain = ops.convert_to_tensor(to_retain)

  # Shape checking, if shape is known at graph construction time
  retain_shape = to_retain.get_shape()
  retain_shape.assert_has_rank(1)
  sp_input.values.get_shape()[0].merge_with(retain_shape[0])

  where_true = array_ops.reshape(array_ops.where(to_retain), [-1])
  new_indices = array_ops.gather(sp_input.indices, where_true)
  new_values = array_ops.gather(sp_input.values, where_true)
  return ops.SparseTensor(new_indices, new_values,
                          array_ops.identity(sp_input.shape))


def sparse_fill_empty_rows(sp_input, default_value, name=None):
  """Fills empty rows in the input 2-D `SparseTensor` with a default value.

  This op adds entries with the specified `default_value` at index
  `[row, 0]` for any row in the input that does not already have a value.

  For example, suppose `sp_input` has shape `[5, 6]` and non-empty values:

      [0, 1]: a
      [0, 3]: b
      [2, 0]: c
      [3, 1]: d

  Rows 1 and 4 are empty, so the output will be of shape `[5, 6]` with values:

      [0, 1]: a
      [0, 3]: b
      [1, 0]: default_value
      [2, 0]: c
      [3, 1]: d
      [4, 0]: default_value

  Note that the input may have empty columns at the end, with no effect on
  this op.

  The output `SparseTensor` will be in row-major order and will have the
  same shape as the input.

  This op also returns an indicator vector such that

      empty_row_indicator[i] = True iff row i was an empty row.

  Args:
    sp_input: A `SparseTensor` with shape `[N, M]`.
    default_value: The value to fill for empty rows, with the same type as
      `sp_input.`
    name: A name prefix for the returned tensors (optional)

  Returns:
    sp_ordered_output: A `SparseTensor` with shape `[N, M]`, and with all empty
      rows filled in with `default_value`.
    empty_row_indicator: A bool vector of length `N` indicating whether each
      input row was empty.

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  if not isinstance(sp_input, ops.SparseTensor):
    raise TypeError("Input must be a SparseTensor")

  with ops.op_scope([sp_input], name, "SparseFillEmptyRows"):
    default_value = ops.convert_to_tensor(default_value,
                                          dtype=sp_input.values.dtype)

    num_rows = math_ops.cast(sp_input.shape[0], dtypes.int32)
    all_row_indices = math_ops.cast(math_ops.range(num_rows), dtypes.int64)
    empty_row_indices, _ = array_ops.list_diff(all_row_indices,
                                               sp_input.indices[:, 0])
    empty_row_indicator = sparse_to_dense(
        empty_row_indices, array_ops.expand_dims(sp_input.shape[0], -1), True,
        False)

    empty_row_indices_as_column = array_ops.reshape(empty_row_indices, [-1, 1])
    additional_indices = array_ops.concat(
        1, [empty_row_indices_as_column,
            array_ops.zeros_like(empty_row_indices_as_column)])
    additional_values = array_ops.fill(
        array_ops.shape(empty_row_indices), default_value)

    all_indices_unordered = array_ops.concat(0, [sp_input.indices,
                                                 additional_indices])
    all_values_unordered = array_ops.concat(0, [sp_input.values,
                                                additional_values])
    sp_unordered_output = ops.SparseTensor(all_indices_unordered,
                                           all_values_unordered, sp_input.shape)
    sp_ordered_output = sparse_reorder(sp_unordered_output)

    return sp_ordered_output, empty_row_indicator


def serialize_sparse(sp_input, name=None):
  """Serialize a `SparseTensor` into a string 3-vector (1-D `Tensor`) object.

  Args:
    sp_input: The input `SparseTensor`.
    name: A name prefix for the returned tensors (optional).

  Returns:
    A string 3-vector (1D `Tensor`), with each column representing the
    serialized `SparseTensor`'s indices, values, and shape (respectively).

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  if not isinstance(sp_input, ops.SparseTensor):
    raise TypeError("Input must be a SparseTensor.")

  return gen_sparse_ops._serialize_sparse(
      sp_input.indices,
      sp_input.values,
      sp_input.shape,
      name=name)


@ops.RegisterShape("SerializeSparse")
def _SerializeSparseShape(op):  # pylint: disable=invalid-name
  """Shape function for SerializeSparse op."""
  op.inputs[0].get_shape().with_rank(2)
  op.inputs[1].get_shape().with_rank(1)
  op.inputs[2].get_shape().with_rank(1)

  return [tensor_shape.vector(3)]


def serialize_many_sparse(sp_input, name=None):
  """Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` string `Tensor`.

  The `SparseTensor` must have rank `R` greater than 1, and the first dimension
  is treated as the minibatch dimension.  Elements of the `SparseTensor`
  must be sorted in increasing order of this first dimension.  The serialized
  `SparseTensor` objects going into each row of the output `Tensor` will have
  rank `R-1`.

  The minibatch size `N` is extracted from `sparse_shape[0]`.

  Args:
    sp_input: The input rank `R` `SparseTensor`.
    name: A name prefix for the returned tensors (optional).

  Returns:
    A string matrix (2-D `Tensor`) with `N` rows and `3` columns.
    Each column represents serialized `SparseTensor`'s indices, values, and
    shape (respectively).

  Raises:
    TypeError: If `sp_input` is not a `SparseTensor`.
  """
  if not isinstance(sp_input, ops.SparseTensor):
    raise TypeError("Input must be a SparseTensor.")

  return gen_sparse_ops._serialize_many_sparse(
      sp_input.indices,
      sp_input.values,
      sp_input.shape,
      name=name)


@ops.RegisterShape("SerializeManySparse")
def _SerializeManySparseShape(op):  # pylint: disable=invalid-name
  """Shape function for SerializeSparse op."""
  op.inputs[0].get_shape().with_rank(2)
  op.inputs[1].get_shape().with_rank(1)
  op.inputs[2].get_shape().with_rank(1)

  return [tensor_shape.matrix(None, 3)]


def deserialize_many_sparse(serialized_sparse, dtype, name=None):
  """Deserialize and concatenate `SparseTensors` from a serialized minibatch.

  The input `serialized_sparse` must be a string matrix of shape `[N x 3]` where
  `N` is the minibatch size and the rows correspond to packed outputs of
  `serialize_sparse`.  The ranks of the original `SparseTensor` objects
  must all match.  When the final `SparseTensor` is created, it has rank one
  higher than the ranks of the incoming `SparseTensor` objects (they have been
  concatenated along a new row dimension).

  The output `SparseTensor` object's shape values for all dimensions but the
  first are the max across the input `SparseTensor` objects' shape values
  for the corresponding dimensions.  Its first shape value is `N`, the minibatch
  size.

  The input `SparseTensor` objects' indices are assumed ordered in
  standard lexicographic order.  If this is not the case, after this
  step run `sparse_reorder` to restore index ordering.

  For example, if the serialized input is a `[2, 3]` matrix representing two
  original `SparseTensor` objects:

      index = [ 0]
              [10]
              [20]
      values = [1, 2, 3]
      shape = [50]

  and

      index = [ 2]
              [10]
      values = [4, 5]
      shape = [30]

  then the final deserialized `SparseTensor` will be:

      index = [0  0]
              [0 10]
              [0 20]
              [1  2]
              [1 10]
      values = [1, 2, 3, 4, 5]
      shape = [2 50]

  Args:
    serialized_sparse: 2-D `Tensor` of type `string` of shape `[N, 3]`.
      The serialized and packed `SparseTensor' objects.
    dtype: The `dtype` of the serialized `SparseTensor` objects.
    name: A name prefix for the returned tensors (optional)

  Returns:
    A `SparseTensor` representing the deserialized `SparseTensor`s,
    concatenated along the `SparseTensor`s' first dimension.

    All of the serialized `SparseTensor`s must have had the same rank and type.
  """
  output_indices, output_values, output_shape = (
      gen_sparse_ops._deserialize_many_sparse(
          serialized_sparse, dtype, name=name))

  return ops.SparseTensor(output_indices, output_values, output_shape)


@ops.RegisterShape("DeserializeManySparse")
def _DeserializeSparseShape(op):  # pylint: disable=invalid-name
  """Shape function for DeserializeManySparse op."""
  serialized_sparse_shape = op.inputs[0].get_shape().with_rank(2)
  serialized_sparse_shape.merge_with(
      tensor_shape.TensorShape([None, 3]))

  return [tensor_shape.matrix(None, None),
          tensor_shape.vector(None),
          tensor_shape.vector(None)]
