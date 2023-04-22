# Lint as python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""StructuredTensor array ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Sequence

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.structured.structured_tensor import StructuredTensor
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch


@dispatch.dispatch_for_types(array_ops.expand_dims, StructuredTensor)
@deprecation.deprecated_args(None, 'Use the `axis` argument instead', 'dim')
def expand_dims(input, axis=None, name=None, dim=None):  # pylint: disable=redefined-builtin
  """Creates a StructuredTensor with a length 1 axis inserted at index `axis`.

  This is an implementation of tf.expand_dims for StructuredTensor. Note
  that the `axis` must be less than or equal to rank.

  >>> st = StructuredTensor.from_pyval([[{"x": 1}, {"x": 2}], [{"x": 3}]])
  >>> tf.expand_dims(st, 0).to_pyval()
  [[[{'x': 1}, {'x': 2}], [{'x': 3}]]]
  >>> tf.expand_dims(st, 1).to_pyval()
  [[[{'x': 1}, {'x': 2}]], [[{'x': 3}]]]
  >>> tf.expand_dims(st, 2).to_pyval()
  [[[{'x': 1}], [{'x': 2}]], [[{'x': 3}]]]
  >>> tf.expand_dims(st, -1).to_pyval()  # -1 is the same as 2
  [[[{'x': 1}], [{'x': 2}]], [[{'x': 3}]]]

  Args:
    input: the original StructuredTensor.
    axis: the axis to insert the dimension: `-(rank + 1) <= axis <= rank`
    name: the name of the op.
    dim: deprecated: use axis.

  Returns:
    a new structured tensor with larger rank.

  Raises:
    an error if `axis < -(rank + 1)` or `rank < axis`.
  """
  axis = deprecation.deprecated_argument_lookup('axis', axis, 'dim', dim)
  return _expand_dims_impl(input, axis, name=name)


@dispatch.dispatch_for_types(array_ops.expand_dims_v2, StructuredTensor)
def expand_dims_v2(input, axis, name=None):  # pylint: disable=redefined-builtin
  """Creates a StructuredTensor with a length 1 axis inserted at index `axis`.

  This is an implementation of tf.expand_dims for StructuredTensor. Note
  that the `axis` must be less than or equal to rank.

  >>> st = StructuredTensor.from_pyval([[{"x": 1}, {"x": 2}], [{"x": 3}]])
  >>> tf.expand_dims(st, 0).to_pyval()
  [[[{'x': 1}, {'x': 2}], [{'x': 3}]]]
  >>> tf.expand_dims(st, 1).to_pyval()
  [[[{'x': 1}, {'x': 2}]], [[{'x': 3}]]]
  >>> tf.expand_dims(st, 2).to_pyval()
  [[[{'x': 1}], [{'x': 2}]], [[{'x': 3}]]]
  >>> tf.expand_dims(st, -1).to_pyval()  # -1 is the same as 2
  [[[{'x': 1}], [{'x': 2}]], [[{'x': 3}]]]

  Args:
    input: the original StructuredTensor.
    axis: the axis to insert the dimension: `-(rank + 1) <= axis <= rank`
    name: the name of the op.

  Returns:
    a new structured tensor with larger rank.

  Raises:
    an error if `axis < -(rank + 1)` or `rank < axis`.
  """
  return _expand_dims_impl(input, axis, name=name)


@dispatch.dispatch_for_types(array_ops.gather, StructuredTensor)
def gather(params,
           indices,
           validate_indices=None,
           name=None,
           axis=None,
           batch_dims=0):
  """tf.gather for structured tensors.

  Does not support (yet) checks on illegal axis values, et cetera.

  Indices must be a ragged or dense tensor.
  Args:
    params: a structured tensor to be gathered
    indices: a ragged tensor or tensor to gather by.
    validate_indices: whether to validate the indices
    name: the name of the op(s).
    axis: the axis in params to gather on.
    batch_dims: the number of batch dimensions.

  Returns:
    the params reorganized according to indices.
  """
  if name is None:
    name = 'gather'
  with ops.name_scope(name):
    if axis is None:
      axis = batch_dims
    axis = array_ops.get_positive_axis(axis, params.shape.rank,
                                       ndims_name='params.shape.rank')
    indices = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        indices, name='indices')

    def leaf_op(p):
      return array_ops.gather(
          p,
          indices,
          validate_indices=validate_indices,
          axis=axis,
          batch_dims=batch_dims,
          name=None)

    return _extend_op_single(params, leaf_op)


@dispatch.dispatch_for_types(array_ops.concat, StructuredTensor)
def concat(values, axis, name: str = 'concat'):
  """tf.concat for structured tensors.

  Does not support (yet) checks on illegal axis values, et cetera.

  Args:
    values: a sequence of StructuredTensors.
    axis: an axis to concatenate upon.
    name: the name of the op(s).

  Returns:
    the params reorganized according to indices.
  """
  if name is None:
    name = 'concat'
  _assert_concat_compatible_structured_tensors(values)
  def leaf_op(values):
    return array_ops.concat(values, axis)
  # TODO(martinz): handle axis when it is a tensor.
  axis = array_ops.get_positive_axis(axis, values[0].rank)
  with ops.name_scope(name, 'StructuredConcat', values):
    return _extend_op(values, leaf_op)


@dispatch.dispatch_for_types(random_ops.random_shuffle, StructuredTensor)
def random_shuffle(value, seed=None, name=None):
  """Shuffle a structured tensor on the zeroth axis.

  Args:
    value: a structured tensor of rank at least one.
    seed: the seed for shuffling.
    name: the name for shuffle.

  Returns:
    The shuffled structured tensor.
  """
  with ops.name_scope(name, 'shuffle', [value, seed]):
    if value.rank == 0:
      raise ValueError('Cannot shuffle a scalar StructuredTensor')
    first_dimension = value.nrows()
    index = random_ops.random_shuffle(math_ops.range(first_dimension),
                                      seed=seed)
    return gather(value, index, axis=0)


@dispatch.dispatch_for_types(array_ops.size_v2, StructuredTensor)
def size_v2(input, out_type=dtypes.int32, name=None):
  # pylint: disable=redefined-builtin
  """Returns the size of a tensor."""
  return size(input, name=name, out_type=out_type)


# pylint: disable=protected-access
@dispatch.dispatch_for_types(array_ops.size, StructuredTensor)
def size(input, name=None, out_type=dtypes.int32):
  # pylint: disable=redefined-builtin
  """Returns the size of a tensor."""
  with ops.name_scope(name, 'size', [input]) as name:
    if not input._row_partitions:
      if input._nrows is not None:
        return math_ops.cast(input._nrows, out_type)  # vector.
      else:
        return math_ops.cast(1, out_type)  # scalar.
    # 2D and up.
    last_row_partition = input._row_partitions[-1]
    return last_row_partition.nvals(out_type)


# pylint: disable=protected-access
@dispatch.dispatch_for_types(array_ops.zeros_like, StructuredTensor)
def zeros_like(tensor, dtype=None, name=None, optimize=True):
  """Implementation of zeros_like for StructuredTensor for TF v1."""
  del optimize
  return zeros_like_v2(tensor, dtype=dtype, name=name)


# pylint: disable=protected-access
@dispatch.dispatch_for_types(array_ops.zeros_like_v2, StructuredTensor)
def zeros_like_v2(input, dtype=None, name=None):  # pylint: disable=redefined-builtin
  """Replace every object with a zero.

  Example:
  >>> st = StructuredTensor.from_pyval([{"x":[3]}, {"x":[4,5]}])
  >>> tf.zeros_like(st)
  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([0.0, 0.0], dtype=float32)>
  >>> st = StructuredTensor.from_pyval([[{"x":[3]}], [{"x":[4,5]}, {"x":[]}]])
  >>> tf.zeros_like(st, dtype=tf.int32)
  <tf.RaggedTensor [[0], [0, 0]]>

  Args:
    input: a structured tensor.
    dtype: the dtype of the resulting zeros. (default is tf.float32)
    name: a name for the op.
  Returns:
    a tensor of zeros of the same shape.
  """
  if dtype is None:
    dtype = dtypes.float32
  with ops.name_scope(name, 'zeros_like', [input]) as name:
    if not input._row_partitions:
      if input._nrows is not None:
        return array_ops.zeros([input._nrows], dtype)  # vector.
      else:
        return array_ops.zeros([], dtype)  # scalar.
    # 2D and up.
    last_row_partition = input._row_partitions[-1]

    result = ragged_tensor.RaggedTensor._from_nested_row_partitions(
        array_ops.zeros(last_row_partition.nvals(), dtype=dtype),
        input._row_partitions)
    return result


# pylint: disable=protected-access
@dispatch.dispatch_for_types(array_ops.ones_like, StructuredTensor)
def ones_like(tensor, dtype=None, name=None, optimize=True):
  """Implementation of zeros_like for StructuredTensor for TF v1."""
  del optimize
  return ones_like_v2(tensor, dtype=dtype, name=name)


# pylint: disable=protected-access
@dispatch.dispatch_for_types(array_ops.ones_like_v2, StructuredTensor)
def ones_like_v2(input, dtype=None, name=None):  # pylint: disable=redefined-builtin
  """Replace every object with a zero.

  Example:
  >>> st = StructuredTensor.from_pyval([{"x":[3]}, {"x":[4,5]}])
  >>> tf.ones_like(st)
  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([1.0, 1.0], dtype=float32)>
  >>> st = StructuredTensor.from_pyval([[{"x":[3]}], [{"x":[4,5]}, {"x":[]}]])
  >>> tf.ones_like(st, dtype=tf.int32)
  <tf.RaggedTensor [[1], [1, 1]]>

  Args:
    input: a structured tensor.
    dtype: the dtype of the resulting zeros. (default is tf.float32)
    name: a name for the op.
  Returns:
    a tensor of zeros of the same shape.
  """
  if dtype is None:
    dtype = dtypes.float32
  with ops.name_scope(name, 'ones_like', [input]) as name:
    if not input._row_partitions:
      if input._nrows is not None:
        return array_ops.ones([input._nrows], dtype)  # vector.
      else:
        return array_ops.ones([], dtype)  # scalar.
    # 2D and up.
    last_row_partition = input._row_partitions[-1]

    result = ragged_tensor.RaggedTensor._from_nested_row_partitions(
        array_ops.ones(last_row_partition.nvals(), dtype=dtype),
        input._row_partitions)
    return result


@dispatch.dispatch_for_types(array_ops.rank, StructuredTensor)
def rank(input, name=None):
  # pylint: disable=redefined-builtin
  """Returns the rank of a tensor."""
  with ops.name_scope(name, 'rank', [input]) as name:
    return constant_op.constant(input.rank, dtype=dtypes.int32)


def _expand_dims_impl(st, axis, name=None):  # pylint: disable=redefined-builtin
  """Creates a StructuredTensor with a length 1 axis inserted at index `axis`.

  This is an implementation of tf.expand_dims for StructuredTensor. Note
  that the `axis` must be less than or equal to rank.

  >>> st = StructuredTensor.from_pyval([[{"x": 1}, {"x": 2}], [{"x": 3}]])
  >>> tf.expand_dims(st, 0).to_pyval()
  [[[{'x': 1}, {'x': 2}], [{'x': 3}]]]
  >>> tf.expand_dims(st, 1).to_pyval()
  [[[{'x': 1}, {'x': 2}]], [[{'x': 3}]]]
  >>> tf.expand_dims(st, 2).to_pyval()
  [[[{'x': 1}], [{'x': 2}]], [[{'x': 3}]]]
  >>> tf.expand_dims(st, -1).to_pyval()  # -1 is the same as 2
  [[[{'x': 1}], [{'x': 2}]], [[{'x': 3}]]]

  Args:
    st: the original StructuredTensor.
    axis: the axis to insert the dimension: `-(rank + 1) <= axis <= rank`
    name: the name of the op.

  Returns:
    a new structured tensor with larger rank.

  Raises:
    an error if `axis < -(rank + 1)` or `rank < axis`.
  """
  axis = array_ops.get_positive_axis(
      axis, st.rank + 1, axis_name='axis', ndims_name='rank(st)')
  with ops.name_scope(name, 'ExpandDims', [st, axis]):
    new_fields = {
        k: array_ops.expand_dims(v, axis) for (k, v) in st._fields.items()
    }
    new_shape = st.shape[:axis] + (1,) + st.shape[axis:]
    new_row_partitions = _expand_st_row_partitions(st, axis)
    new_nrows = st.nrows() if (axis > 0) else 1
    return StructuredTensor.from_fields(
        new_fields,
        shape=new_shape,
        row_partitions=new_row_partitions,
        nrows=new_nrows)


def _expand_st_row_partitions(st, axis):
  """Create the row_partitions for expand_dims."""
  if axis == 0:
    if st.shape.rank == 0:
      return ()
    nvals = st.nrows()
    new_partition = RowPartition.from_uniform_row_length(
        nvals, nvals, nrows=1, validate=False)
    return (new_partition,) + st.row_partitions
  elif axis == st.rank:
    nvals = (
        st.row_partitions[axis - 2].nvals() if (axis - 2 >= 0) else st.nrows())
    return st.row_partitions + (RowPartition.from_uniform_row_length(
        1, nvals, nrows=nvals, validate=False),)
  else:
    nvals = (
        st.row_partitions[axis - 1].nrows() if (axis - 1 >= 0) else st.nrows())
    return st.row_partitions[:axis - 1] + (RowPartition.from_uniform_row_length(
        1, nvals, nrows=nvals, validate=False),) + st.row_partitions[axis - 1:]


# TODO(martinz): consider allowing values to be nested.
def _extend_op(values, leaf_op, empty_st_op=None):
  """Extend an op from RaggedTensor and Tensor to StructuredTensor.

  Visits all children of the structured tensor, and children of children,
  applying leaf_op whenever it reaches a leaf, and empty_st_op whenever
  it reaches an internal node without children.

  Args:
    values: a list of structured tensors, ragged tensors, or tensors. All must
      have the same type. If they are structured tensors, they must have the
      same paths.
    leaf_op: an op for handling non-structured tensor.
    empty_st_op: op to create a structured tensor without fields.

  Returns:
    the result of the extended op (a StructuredTensor, RaggedTensor, or Tensor)

  Raises:
    ValueError:
      If values is not a Sequence or is empty.
  """
  if not isinstance(values, Sequence):
    raise ValueError('Expected a list')

  if not values:
    raise ValueError('List cannot be empty')

  if empty_st_op is None:
    empty_st_op = empty_st_op_like_zeros(leaf_op)
  # Use the structure of the first StructuredTensor. They are all assumed to
  # be the same.
  value = values[0]

  if isinstance(value, StructuredTensor):
    # TODO(martinz): Calling empty_st_op may add unnecessary ops. Revisit later.
    empty_result = empty_st_op(values)
    if not value.field_names():
      return empty_result
    new_fields = {}
    for k in value.field_names():
      new_fields[k] = _extend_op([v.field_value(k) for v in values], leaf_op,
                                 empty_st_op)
    return StructuredTensor.from_fields(new_fields, shape=empty_result.shape)
  else:
    return leaf_op(values)


def _extend_op_single(value, leaf_op, empty_st_op=None):
  """Extend an op to a value instead of a list of values."""

  def to_list_op(element_op):
    if element_op is None:
      return None

    def list_op(values):
      [value] = values
      return element_op(value)

    return list_op

  return _extend_op([value], to_list_op(leaf_op), to_list_op(empty_st_op))


def empty_st_op_like_zeros(leaf_op):

  def empty_st_op(values):
    as_zeros = [
        zeros_like_v2(value, dtype=dtypes.int32) for value in values
    ]
    result = leaf_op(as_zeros)
    return _structured_tensor_like(result)

  return empty_st_op


def _structured_tensor_from_dense_tensor(t):
  """Create a structured tensor with the shape of a dense tensor."""
  # Note: If a tensor will have rank 0,
  # it either has a fully defined shape or has unknown rank.
  if t.shape.is_fully_defined():
    return StructuredTensor.from_fields({}, shape=t.shape)
  elif t.shape.rank is None:
    raise ValueError("Can't build StructuredTensor w/ unknown rank")
  elif t.shape.rank == 1:
    return StructuredTensor.from_fields({}, shape=t.shape,
                                        nrows=array_ops.shape(t)[0])
  else:
    rt = ragged_tensor.RaggedTensor.from_tensor(t)
    return _structured_tensor_from_row_partitions(t.shape,
                                                  rt._nested_row_partitions)


def _structured_tensor_from_row_partitions(shape, row_partitions):
  return StructuredTensor.from_fields({},
                                      shape=shape,
                                      row_partitions=row_partitions)


# pylint: disable=protected_access
def _all_nested_row_partitions(rt):
  """Returns all nested row partitions in rt, including for dense dimensions."""
  if isinstance(rt, ops.Tensor):
    if rt.shape.rank <= 1:
      return ()
    else:
      rt2 = ragged_tensor.RaggedTensor.from_tensor(rt)
      return rt2._nested_row_partitions
  else:
    tail_partitions = _all_nested_row_partitions(rt.flat_values)
    head_partitions = rt._nested_row_partitions  # pylint: disable=protected_access
    return head_partitions + tail_partitions


def _structured_tensor_like(t):
  """Create a StructuredTensor with the shape of a (composite) tensor."""
  if isinstance(t, ops.Tensor):
    return _structured_tensor_from_dense_tensor(t)
  if ragged_tensor.is_ragged(t):
    return StructuredTensor.from_fields(
        {}, shape=t.get_shape(), row_partitions=_all_nested_row_partitions(t))
  # here, it is a StructuredTensor
  return StructuredTensor.from_fields({},
                                      shape=t.shape,
                                      row_partitions=t.row_partitions,
                                      nrows=t.nrows())


def _get_all_paths(st):
  """Get all the paths from a StructuredTensor."""
  fields = st.field_names()
  all_paths = {()}
  for k in fields:
    v = st.field_value(k)
    if isinstance(v, StructuredTensor):
      all_paths = all_paths.union([(k,) + p for p in _get_all_paths(v)])
    else:
      all_paths.add((k,))
  return all_paths


def _get_all_ranks(st):
  """Get ranks of all submessages of a StructuredTensor."""
  fields = st.field_names()
  all_ranks = {(): st.rank}
  for k in fields:
    v = st.field_value(k)
    if isinstance(v, StructuredTensor):
      for (k2, v2) in _get_all_ranks(v).items():
        all_ranks[(k,) + k2] = v2
  return all_ranks


def _assert_all_paths_match(values):
  """Raises an error if the paths are not identical."""
  paths = [_get_all_paths(st) for st in values]
  path_diff = set()
  for other_paths in paths[1:]:
    path_diff = path_diff.union(paths[0].symmetric_difference(other_paths))
  if path_diff:
    raise ValueError(
        'Some paths are present in some, but not all, structured tensors: %r' %
        (path_diff,))


def _assert_all_ranks_match(values):
  """Raises an error if the ranks of submessages are not identical."""
  ranks = [_get_all_ranks(st) for st in values]
  for other_ranks in ranks[1:]:
    if other_ranks != ranks[0]:
      # TODO(martinz): If this becomes common, we can provide more detail.
      # e.g.: which path is inconsistent.
      raise ValueError('Ranks of sub-message do not match')


def _assert_concat_compatible_structured_tensors(values):
  """Sometimes raises an error if concat doesn't make sense statically on values.

  values must be a sequence, and each element in values must be a structured
  tensor, and must have the same paths. Additionally, each path that is a
  submessage must have the same rank.

  These constraints are sufficient for concat on the fields to be the same
  as concat on structured tensors. This is meant to capture scenarios like
  paths that are not in the first structured tensor, but are in later
  structured tensors, which will just be ignored by the recursive algorithm.

  If the rank of a submessage was different for two structured tensors,
  then that is also a non-sensical merge.

  Note that all of these checks are static, as paths and submessage ranks
  are known.

  Args:
    values: a Sequence of StructuredTensors.

  Raises:
    ValueError: if there is any inconsistency as described above.
  """
  if not isinstance(values, Sequence):
    raise ValueError('values must be a list of StructuredTensors (not a list)')
  if not values:
    raise ValueError('values must not be an empty list')
  for st in values:
    if not isinstance(st, StructuredTensor):
      raise ValueError('values must be a list of StructuredTensors')
  _assert_all_paths_match(values)
  _assert_all_ranks_match(values)
