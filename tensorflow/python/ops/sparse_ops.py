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
@@sparse_retain
@@sparse_fill_empty_rows
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import types
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
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
      gen_sparse_ops._sparse_concat(
          inds,
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

  For example, if sp_input has shape `[4, 5]` and `indices` / `values`:

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
      gen_sparse_ops._sparse_reorder(
          sp_input.indices,
          sp_input.values,
          sp_input.shape,
          name=name))

  return ops.SparseTensor(
      reordered_ind, reordered_val, array_ops.identity(sp_input.shape))


@ops.RegisterShape("SparseReorder")
def _SparseReorderShape(op):
  """Shape function for SparseReorder op."""
  input_indices_shape = op.inputs[0].get_shape().with_rank(2)
  input_values_shape = op.inputs[1].get_shape().with_rank(1)
  unused_shape_shape = op.inputs[2].get_shape().with_rank(1)

  return [input_indices_shape, input_values_shape]


@ops.RegisterShape("SparseToDense")
def _SparseToDenseShape(op):
  input_shape = tensor_util.ConstantValue(op.inputs[1])
  if input_shape is not None:
    if np.ndim(input_shape) > 1:
      raise ValueError("Input shape should be a vector")
    return [tensor_shape.TensorShape(input_shape.tolist())]
  else:
    input_shape_shape = op.inputs[1].get_shape().with_rank_at_most(1)
    return [tensor_shape.unknown_shape(ndims=input_shape_shape.num_elements())]


def sparse_tensor_to_dense(sp_input, default_value, name=None):
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

  Args:
    sp_input: The input `SparseTensor`.
    default_value: Scalar value to set for indices not specified in
      `sp_input`.
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

  return gen_sparse_ops.sparse_to_dense(
      sp_input.indices,
      sp_input.shape,
      sp_input.values,
      default_value,
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
      [1, 1, 2]: 112
      [1, 1, 3]: 113
      [1, 2, 1]: 121

  and `vocab_size = 200`, then the output will be a `[2, 3, 200]` dense bool
  tensor with False everywhere except at positions

      (0, 0, 0), (0, 1, 10), (1, 0, 103), (1, 1, 112), (1, 1, 113), (1, 2, 121).

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
    if ids.dtype != types.int64:
      ids = math_ops.cast(ids, types.int64)

    # Slice off the last dimension of indices, then then tack on the ids
    indices_columns_to_preserve = array_ops.slice(
        sp_input.indices, [0, 0], array_ops.pack([-1, rank - 1]))
    new_indices = array_ops.concat(
        1, [indices_columns_to_preserve, array_ops.reshape(ids, [-1, 1])])

    new_values = array_ops.fill(array_ops.expand_dims(num_entries, 0), True)
    new_shape = array_ops.concat(
        0, [array_ops.slice(sp_input.shape, [0],
                            array_ops.expand_dims(rank - 1, 0)), [vocab_size]])

    sp_new = ops.SparseTensor(new_indices, new_values, new_shape)

    return sparse_tensor_to_dense(sp_new, False, name=name)


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
  return ops.SparseTensor(
      new_indices, new_values, array_ops.identity(sp_input.shape))


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
    default_value = ops.convert_to_tensor(
        default_value, dtype=sp_input.values.dtype)

    num_rows = math_ops.cast(sp_input.shape[0], types.int32)
    all_row_indices = math_ops.cast(math_ops.range(num_rows), types.int64)
    empty_row_indices, _ = array_ops.list_diff(
        all_row_indices, sp_input.indices[:, 0])
    empty_row_indicator = gen_sparse_ops.sparse_to_dense(
        empty_row_indices, array_ops.expand_dims(sp_input.shape[0], -1), True,
        False)

    empty_row_indices_as_column = array_ops.reshape(empty_row_indices, [-1, 1])
    additional_indices = array_ops.concat(
        1,
        [empty_row_indices_as_column,
         array_ops.zeros_like(empty_row_indices_as_column)])
    additional_values = array_ops.fill(array_ops.shape(empty_row_indices),
                                       default_value)

    all_indices_unordered = array_ops.concat(
        0, [sp_input.indices, additional_indices])
    all_values_unordered = array_ops.concat(
        0, [sp_input.values, additional_values])
    sp_unordered_output = ops.SparseTensor(
        all_indices_unordered, all_values_unordered, sp_input.shape)
    sp_ordered_output = sparse_reorder(sp_unordered_output)

    return sp_ordered_output, empty_row_indicator
