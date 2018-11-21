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
"""Ops to convert between RaggedTensors and other tensor types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util


#===============================================================================
# RaggedTensor <-> Tensor conversion
#===============================================================================
def from_tensor(tensor, lengths=None, padding=None, ragged_rank=1, name=None):
  """Converts a `Tensor` into a `RaggedTensor`.

  The set of absent/default values may be specified using a vector of lengths
  or a padding value (but not both).  If `lengths` is specified, then the
  output tensor will satisfy `output[row] = tensor[row][:lengths[row]]`.
  If `padding` is specified, then any row *suffix* consisting entirely of
  `padding` will be excluded from the returned `RaggedTensor`.  If neither
  `lengths` nor `padding` is specified, then the returned `RaggedTensor` will
  have no absent/default values.

  Examples:

  ```python
  >>> dt = tf.constant([[5, 7, 0], [0, 3, 0], [6, 0, 0]])
  >>> ragged.from_tensor(dt).eval().tolist()
  [[5, 7, 0], [0, 3, 0], [6, 0, 0]]
  >>> ragged.from_tensor(dt, lengths=[2, 0, 3]).eval().tolist()
  [[5, 7], [], [6, 0, 0]]
  >>> ragged.from_tensor(dt, padding=0).eval().tolist()
  [[5, 7], [0, 3], [6]]
  ```

  Args:
    tensor: The `Tensor` to convert.  Must have rank `ragged_rank + 1` or
      higher.
    lengths: An optional set of row lengths, specified using a 1-D integer
      `Tensor` whose length is equal to `tensor.shape[0]` (the number of rows in
      `tensor`).  If specified, then `output[row]` will contain
      `tensor[row][:lengths[row]]`.  Negative lengths are treated as zero.
    padding: An optional padding value.  If specified, then any row suffix
      consisting entirely of `padding` will be excluded from the returned
      RaggedTensor.  `padding` is a `Tensor` with the same dtype as `tensor`
      and with `shape=tensor.shape[ragged_rank + 1:]`.
    ragged_rank: Integer specifying the ragged rank for the returned
      `RaggedTensor`.  Must be greater than zero.
    name: A name prefix for the returned tensors (optional).

  Returns:
    A `RaggedTensor` with the specified `ragged_rank`.  The shape of the
    returned ragged tensor is compatible with the shape of `tensor`.
  Raises:
    ValueError: If both `lengths` and `padding` are specified.
  """
  if lengths is not None and padding is not None:
    raise ValueError('Specify lengths or padding, but not both')
  if not isinstance(ragged_rank, int):
    raise TypeError('ragged_rank expected int, got %r' % ragged_rank)
  if ragged_rank <= 0:
    raise ValueError('ragged_rank must be greater than 0; got %s' % ragged_rank)

  with ops.name_scope(name, 'RaggedFromTensor', [tensor, lengths, padding]):
    tensor = ops.convert_to_tensor(tensor, name='tensor')
    tensor.shape.with_rank_at_least(ragged_rank + 1)
    input_shape = array_ops.shape(tensor, out_type=dtypes.int64)
    ncols = input_shape[1]

    # Handle ragged_rank>1 via recursion:
    # If the output should have multiple ragged dimensions, then first
    # flatten the tensor to eliminate all but the last ragged dimension,
    # and recursively convert that flattened tensor.  Then add on the splits
    # for the dimensions that we flattened out.
    if ragged_rank > 1:
      # Flatten `tensor` to eliminate all but the last ragged dimension.
      new_shape = array_ops.concat(
          [constant_op.constant([-1], dtypes.int64), input_shape[ragged_rank:]],
          axis=0)
      flattened = array_ops.reshape(tensor, new_shape)
      # Recursively convert the flattened tensor.
      values = from_tensor(flattened, lengths, padding)
      # The total number of elements in each  dimension.  E.g., if
      # input_shape=[3, 4, 5, 6], then dim[2] has 3*4*5 elements in total.
      dim_size = math_ops.cumprod(input_shape)
      # Construct splits tensors for the dimensions that were flattened.
      new_splits = [
          math_ops.range(0, dim_size[dim - 1] + 1) * input_shape[dim]
          for dim in range(1, ragged_rank)
      ]
      return ragged_factory_ops.from_nested_row_splits(values, new_splits)

    # If padding was specified, then use it to find row lengths.
    if padding is not None:
      padding = ops.convert_to_tensor(
          padding, name='padding', dtype=tensor.dtype)
      padding.shape.assert_is_compatible_with(tensor.shape[2:])

      # Find places where the padding is equal to the tensor.  (This will
      # broadcast `padding` across the outermost 2 dimensions of `tensor`,
      # so `has_default_value.shape = tensor.shape`.)
      has_default_value = math_ops.equal(padding, tensor)

      # If the padding isn't a scalar, then require that all values in the
      # padding match each item in the tensor.  After this block of code,
      # `has_default.shape = tensor.shape[:2]`.  (Unfortunately, we can't just
      # use reduce_all for both cases, becaue when you pass an empty `axis`
      # list to reduce_all, it reduces all axes; but we want it to reduce no
      # axes -- i.e., to be a no-op.)
      tensor_rank = array_ops.rank(tensor)
      reduce_axis = math_ops.range(2, tensor_rank)
      has_default = control_flow_ops.cond(
          tensor_rank > 2,
          lambda: math_ops.reduce_all(has_default_value, axis=reduce_axis),
          lambda: has_default_value)
      has_default.set_shape(tensor_shape.TensorShape([None, None]))
      has_default.set_shape(tensor.shape[:2])

      # Use has_default it to find the length of each row: for each non-default
      # item in a row, calculate the length that the row needs to have to
      # include that item; and then take the max of those values (across each
      # row).
      has_nondefault = math_ops.logical_not(has_default)
      has_nondefault = math_ops.cast(has_nondefault, dtypes.int64)
      length_for_nondefault_value = (
          has_nondefault * array_ops.expand_dims(
              math_ops.range(1, ncols + 1), 0))
      lengths = math_ops.reduce_max(length_for_nondefault_value, axis=1)

    # If we have lengths (either directly supplied, or computed from paddings),
    # then use those to construct splits; and then use masking to get the
    # corresponding values.
    if lengths is not None:
      lengths = ragged_util.convert_to_int_tensor(lengths, 'lengths',
                                                  dtypes.int64)
      lengths.shape.assert_has_rank(1)
      lengths = math_ops.minimum(lengths, ncols)
      lengths = math_ops.maximum(lengths, 0)
      limits = math_ops.cumsum(lengths)
      splits = array_ops.concat(
          [array_ops.zeros([1], dtypes.int64), limits], axis=0)
      mask = array_ops.sequence_mask(lengths, maxlen=ncols)
      values = array_ops.boolean_mask(tensor, mask)
      return ragged_factory_ops.from_row_splits(values, splits)

    # If neither padding nor lengths were specified, then create a splits
    # vector that contains no default values, and reshape the input tensor
    # to form the values for the RaggedTensor.
    nrows = input_shape[0]
    nvals = nrows * ncols
    splits = math_ops.range(nrows + 1) * ncols
    values_shape = array_ops.concat([[nvals], input_shape[2:]], axis=0)
    values = array_ops.reshape(tensor, values_shape)
    return ragged_factory_ops.from_row_splits(values, splits)


def to_tensor(rt_input, default_value=None, name=None):
  """Converts a `RaggedTensor` into a `Tensor`.

  Example:

  ```python
  >>> rt = ragged.constant([[9, 8, 7], [], [6, 5], [4]])
  >>> print ragged.to_tensor(rt).eval()
  [[9 8 7]
   [0 0 0]
   [6 5 0]
   [4 0 0]]
  ```

  Args:
    rt_input: The input `RaggedTensor`.
    default_value: Value to set for indices not specified in `rt_input`.
      Defaults to zero.  `default_value` must be broadcastable to
      `rt_input.shape[rt_input.ragged_rank + 1:]`.
    name: A name prefix for the returned tensors (optional).

  Returns:
    A `Tensor` with shape `ragged.bounding_shape(rt_input)` and the
    values specified by the non-empty values in `rt_input`.  Empty values are
    assigned `default_value`.
  """
  with ops.name_scope(name, 'RaggedToTensor', [rt_input, default_value]):
    rt_input = ragged_factory_ops.convert_to_tensor_or_ragged_tensor(
        rt_input, name='rt_input')
    if not ragged_tensor.is_ragged(rt_input):
      return rt_input  # already dense
    if default_value is not None:
      default_value = ops.convert_to_tensor(
          default_value, name='default_value', dtype=rt_input.dtype)

    # If ragged_rank > 1, then recursively convert the ragged values into a
    # `Tensor` before we proceed.
    values = rt_input.values
    if ragged_tensor.is_ragged(values):
      values = to_tensor(values, default_value)

    # Tile the default value, if necessary.
    if default_value is not None:
      if values.shape.ndims is not None:
        default_value.shape.with_rank_at_most(values.shape.ndims - 1)
      if (values.shape.ndims is None or default_value.shape.ndims is None or
          values.shape.ndims != default_value.shape.ndims + 1):
        value_shape = array_ops.shape(values)[1:]
        default_value = array_ops.broadcast_to(default_value, value_shape)
      default_value.shape.assert_is_compatible_with(values.shape[1:])

    # Get the expected dense shape ([nrows, ncols] + value_shape).
    rt_row_lengths = [rt_input.row_splits[1:] - rt_input.row_splits[:-1]]
    nrows = array_ops.shape(rt_input.row_splits, out_type=dtypes.int64)[0] - 1
    ncols = math_ops.maximum(math_ops.reduce_max(rt_row_lengths), 0)
    values_shape = array_ops.shape(values, out_type=dtypes.int64)
    value_shape = values_shape[1:]
    nvals = values_shape[0]

    # Build a default value if none was supplied.
    if default_value is None:
      default_value = array_ops.zeros(value_shape, dtype=values.dtype)
    default_value.shape.assert_is_compatible_with(values.shape[1:])
    default_value.set_shape(values.shape[1:])

    # Get the row start indices, and expand to shape=[nrows, 1].
    starts = array_ops.expand_dims(rt_input.row_splits[:-1], 1)

    # Get the row limit indices, and expand to shape=[nrows, 1].
    limits = array_ops.expand_dims(rt_input.row_splits[1:], 1)

    # Get the column indices, and expand to shape=[1, ncols].
    columns = array_ops.expand_dims(math_ops.range(0, ncols), 0)

    # Build a list containing the values plus the default value.  We will use
    # tf.gather to collect values from this list for the `Tensor` (using
    # nvals as the index for the default value).
    values_and_default = array_ops.concat(
        [values, array_ops.stack([default_value])], axis=0)

    # Construct a matrix "indices" pointing into values_and_default.  I.e.,
    # output[r, c] = values_and_default[indices[r, c].
    nondefault_index = starts + columns
    has_value = nondefault_index < limits
    default_index = array_ops.fill(array_ops.stack([nrows, ncols]), nvals)
    indices = array_ops.where(has_value, nondefault_index, default_index)

    # Gather the results into a `Tensor`.
    return array_ops.gather(values_and_default, indices)


#===============================================================================
# RaggedTensor <-> SparseTensor conversion
#===============================================================================
def to_sparse(rt_input, name=None):
  """Converts a `RaggedTensor` into a sparse tensor.

  Example:

  ```python
  >>> rt = ragged.constant([[1, 2, 3], [4], [], [5, 6]])
  >>> ragged.to_sparse(rt).eval()
  SparseTensorValue(indices=[[0, 0], [0, 1], [0, 2], [1, 0], [3, 0], [3, 1]],
                    values=[1, 2, 3, 4, 5, 6],
                    dense_shape=[4, 3])
  ```

  Args:
    rt_input: The input `RaggedTensor`.
    name: A name prefix for the returned tensors (optional).

  Returns:
    A SparseTensor with the same values as `rt_input`.
  """
  if not ragged_tensor.is_ragged(rt_input):
    raise TypeError('Expected RaggedTensor, got %s' % type(rt_input).__name__)
  with ops.name_scope(name, 'RaggedToSparse', [rt_input]):
    rt_input = ragged_factory_ops.convert_to_tensor_or_ragged_tensor(
        rt_input, name='rt_input')
    result = gen_ragged_conversion_ops.ragged_tensor_to_sparse(
        rt_input.nested_row_splits, rt_input.inner_values, name=name)
    return sparse_tensor.SparseTensor(
        result.sparse_indices, result.sparse_values, result.sparse_dense_shape)


@ops.RegisterGradient('RaggedTensorToSparse')
def _ragged_tensor_to_sparse_gradient(op, unused_sparse_indices_grad,
                                      sparse_values_grad,
                                      unused_sparse_shape_grad):
  """Gradient for ragged.to_sparse."""
  op_inputs_nested_row_splits = op.inputs[:-1]
  op_inputs_inner_values = op.inputs[-1]

  # No gradient for the RaggedTensor's nested_row_splits.
  nested_row_splits_gradient = [None] * len(op_inputs_nested_row_splits)

  # Gradient for the RaggedTensor's inner_values is formed by reshaping
  # the gradient for the SparseTensor's values.
  inner_values_shape = array_ops.shape(op_inputs_inner_values)
  inner_values_gradient = array_ops.reshape(sparse_values_grad,
                                            inner_values_shape)

  return nested_row_splits_gradient + [inner_values_gradient]


def from_sparse(st_input, name=None):
  """Converts a 2D `SparseTensor` to a `RaggedTensor`.

  Each row of the `output` `RaggedTensor` will contain the explicit values from
  the same row in `st_input`.  `st_input` must be ragged-right.  If not it is
  not ragged-right, then an error will be generated.

  Example:

  ```python
  >>> st = SparseTensor(indices=[[0, 1], [0, 2], [0, 3], [1, 0], [3, 0]],
  ...                   values=[1, 2, 3, 4, 5],
  ...                   dense_shape=[4, 3])
  >>> ragged.from_sparse(st).eval().tolist()
  [[1, 2, 3], [4], [], [5]]
  ```

  Currently, only two-dimensional `SparseTensors` are supported.

  Args:
    st_input: The sparse tensor to convert.  Must have rank 2.
    name: A name prefix for the returned tensors (optional).

  Returns:
    A `RaggedTensor` with the same values as `st_input`.
    `output.ragged_rank = rank(st_input) - 1`.
    `output.shape = [st_input.dense_shape[0], None]`.
  Raises:
    ValueError: If the number of dimensions in `st_input` is not known
      statically, or is not two.
  """
  if not sparse_tensor.is_sparse(st_input):
    raise TypeError('Expected SparseTensor, got %s' % type(st_input).__name__)
  with ops.name_scope(name, 'RaggedFromSparse', [st_input]):
    st_input = sparse_tensor.convert_to_tensor_or_sparse_tensor(
        st_input, name='rt_input')

    static_rank_from_dense_shape = (
        None if st_input.dense_shape.shape.ndims is None
        else st_input.dense_shape.shape.dims[0].value)
    static_rank_from_indices = (
        None if st_input.indices.shape.ndims is None
        else st_input.indices.shape.dims[1].value)

    if static_rank_from_dense_shape != 2 and static_rank_from_indices != 2:
      raise ValueError('rank(st_input) must be 2')

    with ops.control_dependencies(
        _assert_sparse_indices_are_ragged_right(st_input.indices)):
      # Treat sparse row indices as segment ids to generate a splits tensor that
      # we can pair with the sparse tensor values.  (Ignore sparse column
      # indices.)
      segment_ids = st_input.indices[:, 0]
      num_segments = st_input.dense_shape[0]
      return ragged_factory_ops.from_value_rowids(st_input.values, segment_ids,
                                                  num_segments)


def _assert_sparse_indices_are_ragged_right(indices):
  """Checks that the given SparseTensor.indices tensor is ragged-right.

  Example: `indices = [[0, 0], [0, 1], [2, 0], [3, 1]]` is not ragged right
  because the entry `[3, 1]` skips a cell.

  Args:
    indices: The SparseTensor indices to check.

  Returns:
    A list of control dependency op tensors.
  """
  index_prefix = indices[:, :-1]
  index_suffix = indices[:, -1]

  # Check whether each index is starting a new row in the innermost dimension
  # (prefix[i] != prefix[i-1]) or continuing a row (prefix[i] == prefix[i-1]).
  # (Note: this skips the first index; we will check that separately below.)
  index_prefix_changed = math_ops.reduce_any(
      math_ops.not_equal(index_prefix[1:], index_prefix[:-1]), axis=1)

  # Check two cases:
  #   * For indices that start a new row: index_suffix[i] must be zero.
  #   * For indices that continue a row: index_suffix[i] must be equal to
  #     index_suffix[i-1]+1.
  index_ok = array_ops.where(
      index_prefix_changed, math_ops.equal(index_suffix[1:], 0),
      math_ops.equal(index_suffix[1:], index_suffix[:-1] + 1))

  # Also check that the very first index didn't skip any cells.  The first
  # index starts a new row (by definition), so its suffix should be zero.
  sparse_indices_are_ragged_right = math_ops.logical_and(
      math_ops.reduce_all(math_ops.equal(index_suffix[:1], 0)),
      math_ops.reduce_all(index_ok))

  message = [
      'SparseTensor is not right-ragged',
      'SparseTensor.indices =', indices
  ]
  return [control_flow_ops.Assert(sparse_indices_are_ragged_right, message)]
