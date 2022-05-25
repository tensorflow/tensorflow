# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Shapes & broadcasting for RaggedTensors.

TODO(martinz): make this suitable for output for tf.shape
TODO(martinz): replace ragged_tensor_shape with this.
"""


import abc
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.ragged.row_partition import RowPartitionSpec
from tensorflow.python.types import core
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


# TODO(martinz): allow inner_shape to be a fully defined TensorShape.
# A "fully defined TensorShape" means one where the rank and all dimensions are
# known.
# Allowing inner_shape might mean allowing inner_shape to be initialized by
# a fully defined TensorShape, or it might mean that you can actually store
# TensorShape in the inner_shape field. This could conceivably construct
# a DynamicRaggedShape that was dtype agnostic.
#
# TODO(martinz): unify the impl of the determination of index type across
#     RowPartition and DynamicRaggedShape.
@tf_export("experimental.DynamicRaggedShape")
class DynamicRaggedShape(extension_type.ExtensionType):
  """The shape of a ragged or dense tensor.

  Ragged shapes are encoded using two fields:

  * `inner_shape`: An integer vector giving the shape of a dense tensor.
  * `row_partitions`: A list of `RowPartition` objects, describing how
    that flat shape should be partitioned to add ragged axes.

  If a DynamicRaggedShape is the shape of a RaggedTensor rt, then:
  1. row_partitions = rt._nested_row_partitions
     (and thus len(row_partitions) > 0)
  2. inner_shape is the shape of rt.flat_values

  If a DynamicRaggedShape is the shape of a dense tensor t, then:
  1. row_partitions = []
  2. inner_shape is the shape of t.

  Examples:

  The following table gives a few examples (where `RP(lengths)` is short
  for `RowPartition.from_lengths(lengths)`):

  Row Partitions              | Inner Shape  | Example Tensor
  --------------------------- | ------------ | ----------------------------
  []                          | [2, 3]       | `[[1, 2, 3], [4, 5, 6]]`
  [RP([2, 0, 3])]             | [5]          | `[[1, 2], [], [3, 4, 5]]`
  [RP([2, 1])]                | [3, 2]       | `[[[1, 2], [3, 4]], [[5, 6]]]`
  [RP([2, 1]), RP([2, 1, 2])] | [5]          | `[[[1, 2], [3]], [[4, 5]]]`
  """
  _row_partitions: Tuple[RowPartition, ...]
  _inner_shape: ops.Tensor
  _static_inner_shape: tensor_shape.TensorShape

  def __init__(self,
               row_partitions: Sequence[RowPartition],
               inner_shape: core.TensorLike,
               dtype: Optional[dtypes.DType] = None,
               validate: bool = False,
               static_inner_shape: ... = None):
    """Core constructor for a DynamicRaggedShape.

    Create a DynamicRaggedShape. This can be used to construct a
    DynamicRaggedShape representing a ragged or dense shape. If row_partitions
    is an empty list, then this is equivalent to a dense shape.

    If row_partitions is specified, then the num_row_partitions will be equal
    to len(row_partitions). There are several checks made.
    Specifically:
    1. Consecutive row_partitions must have consistent nvals and nrows.
    2. The last row_partitions must have nvals equal to the first element of
       inner_shape.

    The inner_shape is converted to a tensor.
    All row_partitions and the inner_shape are converted to the same dtype
    (int64 or int32).

    Args:
      row_partitions: the row_partitions of the shape.
      inner_shape: if len(row_partitions) > 0, the shape of the flat_values.
        Otherwise, the shape of the tensor.
      dtype: tf.int64, tf.int32, or None representing the preferred dtype.
      validate: if true, dynamic validation is applied to the shape.
      static_inner_shape: if len(row_partitions) > 0, the static shape of the
        flat_values. Otherwise, the static shape of the tensor.
        Should be convertible to a TensorShape.
    """
    if not isinstance(row_partitions, Iterable):
      raise TypeError(
          "row_partitions should be a list of row partitions. Instead, got " +
          str(row_partitions))
    for x in row_partitions:
      if not isinstance(x, RowPartition):
        raise TypeError("row_partitions contains " + str(x) +
                        " which is not a RowPartition")
    dtype = _find_dtype_iterable(row_partitions, dtype)
    dtype = _find_dtype(inner_shape, dtype)
    if (isinstance(inner_shape, np.ndarray) and
        inner_shape.dtype == np.int32 and dtype is None):
      dtype = dtypes.int32
    dtype = _find_dtype(dtypes.int64, dtype)

    row_partitions = tuple([rp.with_dtype(dtype) for rp in row_partitions])
    self._row_partitions = row_partitions
    self._inner_shape = ops.convert_to_tensor(
        inner_shape, dtype_hint=dtype, name="inner_dim_sizes")
    if self._inner_shape.dtype != dtype:
      self._inner_shape = math_ops.cast(self._inner_shape, dtype)

    checks = []
    # Validate shapes.
    if self._row_partitions:
      for axis, rp in enumerate(self._row_partitions):
        if axis > 0:
          previous_row_partition = self._row_partitions[axis - 1]
          msg = ("RowPartitions in DynamicRaggedShape do not align "
                 f"between {axis - 1} and {axis}")
          static_nrows = rp.static_nrows
          static_nvals = previous_row_partition.static_nvals
          if (static_nrows is not None) and (static_nvals is not None):
            if static_nrows != static_nvals:
              raise ValueError(msg)
            else:
              continue
          if validate:
            checks.append(
                check_ops.assert_equal(
                    previous_row_partition.nvals(),
                    rp.nrows(),
                    message=msg))

    self._inner_shape.shape.assert_has_rank(1)

    self._static_inner_shape = tensor_util.constant_value_as_shape(
        self._inner_shape)
    if static_inner_shape is not None:
      self._static_inner_shape = self._static_inner_shape.merge_with(
          static_inner_shape)

    if row_partitions:
      last_row_partition = row_partitions[-1]
      static_nvals = last_row_partition.static_nvals
      static_inner_shape_nvals = tensor_shape.dimension_value(
          self._static_inner_shape[0])
      if static_nvals is not None and static_inner_shape_nvals is not None:
        if static_nvals != static_inner_shape_nvals:
          raise ValueError("Last row partition does not match inner_shape.")
      elif validate:
        checks.append(
            check_ops.assert_equal(
                last_row_partition.nvals(),
                self._inner_shape[0],
                message="Last row partition does not match inner_shape."))
    if checks:
      self._inner_shape = control_flow_ops.with_dependencies(
          checks, self._inner_shape, name="inner_shape_validated")
      self._row_partitions = [
          rp._with_dependencies(checks) for rp in self._row_partitions  # pylint: disable=protected-access
      ]

  @classmethod
  def from_lengths(cls,
                   lengths: Sequence[Union[Sequence[int], int]],
                   num_row_partitions=None,
                   dtype=dtypes.int64):
    """Creates a shape with the given lengths and num_row_partitions.

    The lengths can either be a nonnegative int or a list of nonnegative ints.

    If num_row_partitions is None, then the minimal num_row_partitions is used.

    For example, [2, (3, 2)] is the shape of [[0, 0, 0], [0, 0]], and
    [2, 2] is the shape of [[0, 0], [0, 0]]

    This chooses the minimal num_row_partitions required (including zero).

    The following table gives a few examples (where `RP(lengths)` is short
    for `RowPartition.from_lengths(lengths)`):

    For example:
    from_lengths           | row_partitions            | inner_shape
    ---------------------- | --------------------------| -------------
    []                     | []                        | []
    [2, (3, 2)]            | [RP([3, 2])]              | [5]
    [2, 2]                 | []                        | [2, 2]
    [2, (3, 2), 7]         | [RP([3, 2])]              | [5, 7]
    [2, (2, 2), 3]         | [RP([2, 2])]              | [4, 3]
    [2, 2, 3]              | []                        | [2, 2, 3]
    [2, (2, 1), (2, 0, 3)] | [RP(2, 1), RP([2, 0, 3])] | [5]

    If we want the row partitions to end with uniform row partitions, then
    we can set num_row_partitions.

    For example,
    below URP(3, 12) is RowPartition.from_uniform_row_length(3, 12)

    from_lengths   | num_row_partitions | row_partitions           | inner_shape
    ---------------| -------------------|--------------------------|------------
    [2, (3, 2), 2] | 2                  | [RP([3, 2]), URP(2, 10)] | [10]
    [2, 2]         | 1                  | [URP(2, 4)]              | [4]
    [2, 2, 3]      | 0                  | []                       | [2, 2, 3]
    [2, 2, 3]      | 1                  | [URP(2, 4)]              | [4, 3]
    [2, 2, 3]      | 2                  | [URP(2, 4), URP(3, 12)]  | [12]



    Representing the shapes from init():

    from_lengths             | Tensor Example
    ------------------------ | ------------------------------
    `[2, 3]`                 | `[[1, 2, 3], [4, 5, 6]]`
    `[3, (2, 0, 3)]`         | `[[1, 2], [], [3, 4, 5]]`
    `[2, (2, 1), 2]`         | `[[[1, 2], [3, 4]], [[5, 6]]]`
    `[2, (2, 1), (2, 1, 2)]` | `[[[1, 2], [3]], [[4, 5]]]`

    Args:
      lengths: the lengths of sublists along each axis.
      num_row_partitions: the num_row_partitions of the result or None
      indicating the minimum number of row_partitions.
      dtype: the dtype of the shape (tf.int32 or tf.int64).

    Returns:
      a new DynamicRaggedShape
    """
    if not isinstance(lengths, list):
      raise ValueError("lengths should be a list")
    for x in lengths:
      if not _is_int_or_tuple_of_ints(x):
        raise ValueError(
            "element of lengths should be int or tuple of ints: instead %r" %
            (x,))

    if num_row_partitions is None:
      # Calculate the minimal num_row_partitions.
      is_list = [not isinstance(x, int) for x in lengths]
      if any(is_list):
        # Last index when not a list.
        num_row_partitions = len(is_list) - is_list[-1::-1].index(True) - 1
      else:
        num_row_partitions = 0

    if not isinstance(num_row_partitions, int):
      raise ValueError("num_row_partitions should be an int or None")

    if not lengths:
      if num_row_partitions > 0:
        raise ValueError("num_row_partitions==0 for a scalar shape")
      return DynamicRaggedShape([], [], dtype=dtype)

    if not num_row_partitions < len(lengths):
      raise ValueError(
          "num_row_partitions should be less than `len(lengths)` "
          "if shape is not scalar."
      )

    if num_row_partitions > 0:
      (row_partitions, nvals) = _to_row_partitions_and_nvals_from_lengths(
          lengths[:num_row_partitions + 1])
      inner_shape = [nvals] + lengths[num_row_partitions + 1:]
      return DynamicRaggedShape(
          row_partitions, inner_shape, dtype=dtype)
    else:
      return DynamicRaggedShape([], lengths, dtype=dtype)

  @classmethod
  def from_row_partitions(cls, row_partitions, dtype=None):
    """Create a shape from row_partitions.

    Args:
      row_partitions: a nonempty list of RowPartition objects.
      dtype: the dtype to use, or None to use the row_partitions dtype.

    Returns:
      a DynamicRaggedShape with inner_rank==1.
    """
    if not row_partitions:
      raise ValueError("row_partitions cannot be empty")
    inner_shape = [row_partitions[-1].nvals()]
    return DynamicRaggedShape(
        row_partitions, inner_shape, dtype=dtype)

  @classmethod
  def _from_inner_shape(cls, inner_shape, dtype=None):
    """Create a shape from inner_shape, where num_row_partitions == 0."""
    return DynamicRaggedShape([], inner_shape, dtype=dtype)

  # pylint: disable=protected-access
  @classmethod
  def from_tensor(cls, t, dtype=None):
    """Constructs a ragged shape for a potentially ragged tensor."""
    if ragged_tensor.is_ragged(t):
      return DynamicRaggedShape(
          t._nested_row_partitions, _flat_values_shape(t), dtype=dtype)
    else:
      return DynamicRaggedShape._from_inner_shape(
          array_ops.shape(t), dtype=dtype)

  @property
  def row_partitions(self):
    """The row_partitions of the shape."""
    return self._row_partitions

  @property
  def num_row_partitions(self):
    """The number of row_partitions of the shape."""
    return len(self._row_partitions)

  @property
  def dtype(self):
    """The dtype of the shape -- one of tf.int32 or tf.int64."""
    return self._inner_shape.dtype

  def _static_inner_shape_as_list(self, truncate_first):
    """Returns the lengths of the inner shape (if rank known), or [...]."""
    if self._static_inner_shape.rank is None:
      return [...]
    result = self._static_inner_shape.as_list()
    if truncate_first:
      return result[1:]
    return result

  def static_lengths(self, ragged_lengths=True):
    """Returns a list of statically known axis lengths.

    This represents what values are known. For each row partition, it presents
    either the uniform row length (if statically known),
    the list of row lengths, or none if it is not statically known.
    For the inner shape, if the rank is known, then each dimension is reported
    if known, and None otherwise. If the rank of the inner shape is not known,
    then the returned list ends with an ellipsis.

    Args:
      ragged_lengths: If false, returns None for all ragged dimensions.

    Returns:
      A Sequence[Union[Sequence[int],int, None]] of lengths, with a possible
      Ellipsis at the end.
    """
    if self.num_row_partitions == 0:
      return self._static_inner_shape_as_list(False)
    first_dim = self.row_partitions[0].static_nrows
    if isinstance(first_dim, tensor_shape.Dimension):
      first_dim = first_dim.value
    rp_dims = [first_dim]
    for rp in self.row_partitions:
      if rp.is_uniform():
        rp_dims.append(rp.static_uniform_row_length)
      elif ragged_lengths:
        const_vals = tensor_util.constant_value(rp.row_lengths())
        if const_vals is None:
          rp_dims.append(None)
        else:
          rp_dims.append(tuple(const_vals.tolist()))
      else:
        rp_dims.append(None)

    return rp_dims + self._static_inner_shape_as_list(True)

  def __repr__(self):
    lengths = _list_with_ellipsis_to_str(self.static_lengths())
    return ("<DynamicRaggedShape "
            "lengths=%s num_row_partitions=%r>" %
            (lengths, self.num_row_partitions))

  def _to_tensor_shape(self) -> tensor_shape.TensorShape:
    """Returns a TensorShape representation of the shape."""
    lengths = self.static_lengths(ragged_lengths=False)
    if not lengths:
      return tensor_shape.TensorShape(())
    if lengths[-1] == Ellipsis:
      return tensor_shape.TensorShape(None)
    return tensor_shape.TensorShape(lengths)

  def _slice_shape(self, start, stop):
    """Returns a shape self[start:stop].

    If start == 0, then this truncates dimensions after stop.
    If start != 0, then this will return a shape with num_row_partitions == 0.

    See __getitem__.

    Args:
      start: the first dimension. 0 <= start <= rank
      stop: the last dimension (exclusive). 0 <= stop <= rank
    """
    if stop <= start:
      return DynamicRaggedShape._from_inner_shape([])
    elif start == 0:
      if stop <= self.num_row_partitions:
        if stop == 1:
          return DynamicRaggedShape._from_inner_shape(
              [self.row_partitions[0].nrows()])
        new_row_partitions = self.row_partitions[:stop - 1]
        new_inner_shape = [new_row_partitions[-1].nvals()]
        return DynamicRaggedShape(new_row_partitions, new_inner_shape)
      else:
        if self.rank <= stop:
          return self
        if self.num_row_partitions == 0:
          return DynamicRaggedShape._from_inner_shape(self.inner_shape[:stop])
        else:
          new_inner_shape = self.inner_shape[:stop - self.num_row_partitions]
        return DynamicRaggedShape(
            self.row_partitions, new_inner_shape)
    else:
      if stop < self.rank:
        partial = self._slice_shape(0, stop)
      else:
        partial = self
      for x in self.row_partitions:
        if not x.is_uniform():
          raise ValueError("All relevant dimensions must be uniform")

      return DynamicRaggedShape._from_inner_shape(
          partial._with_num_row_partitions(0).inner_shape[start:])

  def _dimension(self, index):
    """Return a dimension, if the dimension is not ragged (see __getitem__)."""
    rank = self.rank
    if not isinstance(index, int):
      raise TypeError("index should be an int")
    if (self.num_row_partitions == 0 or index > self.num_row_partitions + 1):
      # If num_row_partitions > 0 and index <= num_row_partitions + 1, then
      # we are safe.
      if rank is None:
        raise ValueError(
            "Rank must be known to use __getitem__ on a large index.")
      if index >= rank:
        raise IndexError("Index is too big: " + str(index) + ">=" + str(rank))
    if index < 0:
      raise IndexError("Index must be non-negative: " + str(index))
    elif not self.is_uniform(index):
      raise ValueError("Index " + str(index) + " is not uniform")
    elif index == 0 and self.num_row_partitions > 0:
      static_nrows = self.row_partitions[0].static_nrows
      if static_nrows is not None:
        return constant_op.constant(static_nrows, dtype=self.dtype)
      return self.row_partitions[0].nrows()
    elif self.num_row_partitions == 0:
      static_result = tensor_shape.dimension_value(
          self._static_inner_shape[index])
      if static_result is not None:
        return constant_op.constant(static_result, dtype=self.dtype)
      return self.inner_shape[index]
    elif index > self.num_row_partitions:
      static_result = tensor_shape.dimension_value(
          self._static_inner_shape[index - self.num_row_partitions])
      if static_result is not None:
        return constant_op.constant(static_result, dtype=self.dtype)

      return self.inner_shape[index - self.num_row_partitions]
    else:
      return self.row_partitions[index - 1].uniform_row_length()

  def __getitem__(self, index):
    """Returns a dimension or a slice of the shape.

    Ragged shapes can have ragged dimensions that depend upon other dimensions.
    Therefore, if you ask for a dimension that is ragged, this function returns
    a ValueError. For similar reasons, if a slice is selected that includes
    a ragged dimension without including the zero dimension, then this fails.

    Any slice that does not start at zero will return a shape
    with num_row_partitions == 0.

    Args:
      index: the index: can be an int or a slice.

    Raises:
      IndexError: if the index is not in range.
      ValueError: if the rank is unknown, or a ragged rank is requested
      incorrectly.
    """
    rank = self.rank
    if isinstance(index, slice):

      if (index.step is not None) and (index.step != 1):
        raise IndexError("Cannot stride through a shape")
      start = index.start
      stop = index.stop
      if start is None:
        start = 0
      start = _fix_slice_index(start, rank, self.num_row_partitions)
      if stop is None:
        if rank is None:
          raise ValueError(
              "Rank must be known to use __getitem__ without a stop.")
        stop = rank
      stop = _fix_slice_index(stop, rank, self.num_row_partitions)
      return self._slice_shape(start, stop)
    elif isinstance(index, int):
      if index < 0:
        if rank is None:
          raise ValueError(
              "Rank must be known to use __getitem__ with a negative index.")
        return self._dimension(rank + index)
      return self._dimension(index)
    else:
      raise TypeError("Argument is not an int or a slice")

  def _num_elements(self):
    """Number of elements in a shape.

    Returns:
      The number of elements in the shape.

    """
    return math_ops.reduce_prod(self.inner_shape)

  def _num_slices_in_dimension(self, axis):
    """The total size of a dimension (like nvals).

    Effectively, this is self[:axis+1]._num_elements()

    Example:
    shape = DynamicRaggedShape._from_inner_shape([2, 3, 4])
    shape._num_slices_in_dimension(0) = 2
    shape._num_slices_in_dimension(1) = 6
    shape._num_slices_in_dimension(2) = 24
    shape._num_slices_in_dimension(-1) = 24
    shape._num_slices_in_dimension(-2) = 6
    shape._num_slices_in_dimension(-2) = 2

    Args:
      axis: the last axis to include in the number of elements. If negative,
        then axis = axis + rank.

    Returns:
      The number of elements in the shape.
    """
    if not isinstance(axis, int):
      raise TypeError("axis must be an integer")
    if axis < 0:
      rank = self.rank
      if rank is None:
        raise ValueError(
            "You can't use negative values if the rank is undefined")
      axis = axis + rank
    if axis == 0:
      return self._dimension(0)
    if axis <= self.num_row_partitions:
      return self.row_partitions[axis - 1].nvals()
    # If self.num_row_partitions = 1, and
    # self.inner_shape=[3,5,6], and axis=2, then you want:
    # 15 = 3 * 5 = math_ops.reduce_prod(self.inner_shape[:2])
    # 2 = axis - (self.num_row_partitions - 1)
    # If num_row_partitions=0, and
    # self.inner_shape=[3,5,6] and axis=2, then you want:
    # 90 = 3 * 5 * 6 = math_ops.reduce_prod(self.inner_shape[:3])
    # 3 = axis - (self.num_row_partitions - 1)
    remainder = axis - (self.num_row_partitions - 1)
    return _reduce_prod_patch(self.inner_shape[:remainder])

  def is_uniform(self, axis):
    """Returns true if the indicated dimension is uniform."""
    if not isinstance(axis, int):
      raise TypeError("axis must be an integer")
    rank = self.rank
    if axis < 0:
      raise IndexError("Negative axis values are not supported")
    elif rank is not None and axis >= rank:
      raise IndexError("Expected axis=%s < rank=%s" % (axis, rank))
    else:
      return ((axis == 0 or axis > len(self._row_partitions))  # pylint:disable=superfluous-parens
              or self._row_partitions[axis - 1].is_uniform())

  @property
  def rank(self):
    """The number of dimensions in this shape, or None if unknown."""
    inner_rank = self.inner_rank
    if inner_rank is None:
      return None
    else:
      return self.num_row_partitions + inner_rank

  @property
  def inner_shape(self):
    """The inner dimension sizes for this shape.

    Returns:
      A 1-D integer `Tensor`.
    """
    return self._inner_shape

  @property
  def inner_rank(self):
    """The rank of inner_shape."""
    return tensor_shape.dimension_value(self._inner_shape.shape[0])

  def _alt_inner_shape(self, new_inner_rank):
    """Get an alternative inner shape with higher or lower rank.

    For the rank of the inner shape to be be higher, the last few ragged
    dimensions must have uniform_row_length.

    Args:
      new_inner_rank: the new rank of the inner_shape

    Returns:
       A new inner_shape of rank new_inner_rank.
    """
    if new_inner_rank == 0:
      raise ValueError("new_inner_rank cannot be zero")
    elif self.inner_rank == 0:
      raise ValueError("old inner_rank cannot be zero")
    elif new_inner_rank == self.inner_rank:
      return self.inner_shape
    elif new_inner_rank < self.inner_rank:
      if self._static_inner_shape.is_fully_defined():
        return _alt_inner_shape_from_tensor_shape(self._static_inner_shape,
                                                  self.dtype, new_inner_rank)
      first_dimension = self._num_slices_in_dimension(-new_inner_rank)
      if new_inner_rank == 1:
        return array_ops.expand_dims(first_dimension, 0)
      remaining_dimensions = self.inner_shape[1 - new_inner_rank:]
      return array_ops.concat(
          [array_ops.expand_dims(first_dimension, 0), remaining_dimensions],
          axis=0)
    else:
      assert new_inner_rank > self.inner_rank
      new_dimensions = new_inner_rank - self.inner_rank
      if any(
          [not x.is_uniform() for x in self.row_partitions[-new_dimensions:]]):
        raise ValueError("Cannot get an inner shape over a ragged dimension")
      first_dimension = self._num_slices_in_dimension(-new_inner_rank)
      new_dimensions = new_inner_rank - self.inner_rank
      new_dims = [first_dimension] + [
          x.uniform_row_length() for x in self.row_partitions[-new_dimensions:]
      ]
      return array_ops.concat([array_ops.stack(new_dims), self.inner_shape[1:]],
                              axis=0)

  def _inner_shape_dim(self, dimension):
    """Returns an int or a tensor representing _inner_shape[dimension]."""
    result = tensor_shape.dimension_value(self._static_inner_shape[dimension])
    return self._inner_shape[dimension] if result is None else result

  def _with_inner_rank(self, inner_rank):
    """Returns the same shape but a different inner_rank.

    All dimensions that are to be represented in the inner_shape must be dense.
    See inner_rank.

    Args:
      inner_rank: the new inner_rank of the shape.

    Returns:
      the same shape but a different inner_rank

    Raises:
      ValueError if the new dense rank is invalid, or the old rank is unknown.
    """
    rank = self.rank
    if rank is None:
      raise ValueError("Rank must be known to adjust inner_rank")
    elif rank < 2:
      if inner_rank == rank:
        return self
      raise ValueError("Cannot change inner_rank if rank < 2")
    else:
      # When self.rank is not None:
      # self.rank = self.inner_rank + self.num_row_partitions
      new_num_row_partitions = rank - inner_rank
      return self._with_num_row_partitions(new_num_row_partitions)

  def _with_num_row_partitions(self, num_row_partitions):
    """Creates an identical shape with the given num_row_partitions.

    Note that the shape must be statically refactorable to this rank.
    In particular:
    * rank must be known.
    * num_row_partitions must be a nonnegative int.
    * num_row_partitions must be less than the rank of the shape
    * num_row_partitions must be greater or equal to the index of any ragged
    dimension.

    Note that if the num_row_partitions is the same, self is returned.

    Args:
      num_row_partitions: the target num_row_partitions (must be a nonnegative
        int).

    Returns:
      a shape with a (possibly) different num_row_partitions.

    Raises:
      ValueError: if the rank is unknown, the argument is not a nonnegative int,
        or there is a dimension that is nonuniform.
    """
    rank = self.rank
    if rank is None:
      raise ValueError("Rank must be known to adjust num_row_partitions")
    if not isinstance(num_row_partitions, int):
      raise ValueError("num_row_partitions must be an int")
    if num_row_partitions < 0:
      raise ValueError("num_row_partitions must be nonnegative")
    if num_row_partitions == self.num_row_partitions:
      return self
    if num_row_partitions >= rank:
      raise ValueError("num_row_partitions must be less than rank")
    if num_row_partitions > self.num_row_partitions:
      num_row_partitions_diff = num_row_partitions - self.num_row_partitions
      new_inner_rank = self.rank - num_row_partitions
      nvals = self._inner_shape_dim(0)
      more_rp = []
      for i in range(num_row_partitions_diff):
        nrows = nvals
        row_length = self._inner_shape_dim(i + 1)
        nvals = nrows * row_length
        rp = RowPartition.from_uniform_row_length(
            row_length, nrows=nrows, dtype=self.dtype)
        more_rp.append(rp)
      alt_inner = self._alt_inner_shape(new_inner_rank)
      return DynamicRaggedShape(
          list(self.row_partitions) + more_rp, alt_inner)
    else:
      assert num_row_partitions < self.num_row_partitions
      return DynamicRaggedShape(
          self.row_partitions[:num_row_partitions],
          self._alt_inner_shape(self.rank - num_row_partitions))

  def with_dtype(self, dtype):
    """Change the dtype of the shape."""
    if dtype == self.dtype:
      return self
    else:
      return DynamicRaggedShape(
          self.row_partitions, self.inner_shape, dtype=dtype)

  def _as_row_partitions(self):
    """Returns row partitions representing this shape.

    In order to represent a shape as row partitions, the rank of the shape
    must be known, and the shape must have rank at least one.

    Returns:
      A list of RowPartition objects.
    Raises:
      ValueError, if the shape cannot be represented by RowPartitions.
    """
    rank = self.rank
    if rank is None:
      raise ValueError("rank must be known for _as_row_partitions")
    elif rank < 1:
      raise ValueError("rank must be >= 1 for _as_row_partitions")
    fully_ragged = self._with_num_row_partitions(rank - 1)
    return fully_ragged.row_partitions

  def _validate_flat_values_dynamically(self, flat_values):
    """Test if flat_values have the right nvals dynamically."""
    if self.row_partitions:
      assert_op = check_ops.assert_equal(
          self.row_partitions[-1].nvals(),
          array_ops.shape(flat_values, out_type=self.dtype)[0],
          message="Last row partition does not match flat_values.")
      return control_flow_ops.with_dependencies([assert_op], flat_values)
    return flat_values

  def _validate_flat_values(self, flat_values):
    """Test if flat_values have the right nvals."""
    if not isinstance(flat_values, ops.Tensor):
      return flat_values
    if self.row_partitions:
      last_row_partition = self.row_partitions[-1]
      flat_values_shape = flat_values.shape
      if flat_values_shape is None:
        return self._validate_flat_values_dynamically(flat_values)
      first_dim_flat_values = flat_values_shape[0]
      if isinstance(first_dim_flat_values, tensor_shape.Dimension):
        first_dim_flat_values = first_dim_flat_values.value
      if first_dim_flat_values is None:
        return self._validate_flat_values_dynamically(flat_values)
      static_nvals = last_row_partition.static_nvals
      if static_nvals is None:
        return self._validate_flat_values_dynamically(flat_values)
      if first_dim_flat_values != static_nvals:
        raise ValueError("Last row partition does not match flat_values.")
    return flat_values

  def _add_row_partitions(self, flat_values, validate=False):
    """Add row partitions to flat_values, if necessary.

    If the shape is truly ragged, then this adds the row_partitions.

    The the shape is dense, then this just returns flat_values.

    Args:
      flat_values: the flat_values of a ragged tensor with this shape, or a
        dense tensor with this shape.
      validate: validate the flat_values have the right first dimension.

    Returns:
      flat_values reshaped to have row_partitions.
    """
    if self.row_partitions:
      if validate:
        flat_values = self._validate_flat_values(flat_values)
      return ragged_tensor.RaggedTensor._from_nested_row_partitions(
          flat_values, self.row_partitions, validate=False)
    else:
      return flat_values

  class Spec:
    """A Spec for DynamicRaggedShape: similar to a static shape."""

    def __init__(self, row_partitions: Tuple[RowPartitionSpec, ...],
                 static_inner_shape: tensor_shape.TensorShape,
                 dtype: dtypes.DType):
      """Create a Spec given row partitions, a static inner shape, and a dtype.

      Args:
        row_partitions: A sequence of `RowPartitionSpec`s describing how the
            ragged shape is partitioned.
        static_inner_shape: The static shape of the flat_values.
        dtype: The DType used to encode the shape (tf.int64 or tf.int32).
      """
      # Independent validation and coercion of each argument.
      if not isinstance(row_partitions, Iterable):
        raise TypeError("row_partitions should be an Iterable")

      row_partitions = tuple(row_partitions)

      static_inner_shape = tensor_shape.as_shape(static_inner_shape)

      dtype = dtypes.as_dtype(dtype)

      if not all(isinstance(rp, RowPartitionSpec) for rp in row_partitions):
        raise TypeError(
            "row_partitions should be an Iterable of RowPartitionSpecs")

      if dtype != dtypes.int32 and dtype != dtypes.int64:
        raise ValueError("dtype must be tf.int32 or tf.int64")

      # All fields are now typechecked and internally consistent.
      for spec in row_partitions:
        if spec.dtype != dtype:
          raise ValueError(
              f"dtype of {spec!r} is {spec.dtype!r}: expected {dtype!r}")

      row_partitions = tuple(row_partitions)

      inner_rank = static_inner_shape.rank

      if inner_rank == 0:
        if row_partitions:
          raise ValueError(
              "If row_partitions are provided, must have inner_rank > 0")
      else:
        num_slices_in_dimension = []   # type: Sequence[tensor_shape.Dimension]

        # We first attempt to calculate num_slices_in_dimension through a
        # forward pass, using nrows[k] = nrows[k-1] * uniform_row_length
        # and other tricks.
        for i in range(len(row_partitions)):
          rp = row_partitions[i]
          result = tensor_shape.Dimension(rp.nrows)
          if i > 0:
            previous_rp = row_partitions[i - 1]
            result = result.merge_with(previous_rp.nvals)
            result = result.merge_with(num_slices_in_dimension[-1] *
                                       previous_rp.uniform_row_length)
          num_slices_in_dimension.append(result)
        # In the last step of the forward pass,
        # we combine nvals and the first dimension in static_inner_shape.
        if row_partitions:
          last_rp = row_partitions[-1]
          result = (num_slices_in_dimension[-1] *
                    last_rp.uniform_row_length).merge_with(last_rp.nvals)
          if inner_rank is not None:
            result = result.merge_with(
                tensor_shape.dimension_at_index(static_inner_shape, 0))
            static_inner_shape = result + static_inner_shape[1:]
          num_slices_in_dimension.append(result)

        # Now, we start a backward pass.
        for i in range(len(num_slices_in_dimension) - 1, 0, -1):
          num_slices_in_dimension[i - 1] = num_slices_in_dimension[
              i - 1].merge_with(
                  _safe_floor_div(num_slices_in_dimension[i],
                                  row_partitions[i - 1].uniform_row_length))

        # Finally, we construct the partitions.
        row_partitions = [
            RowPartitionSpec(  # pylint: disable=g-complex-comprehension
                nrows=num_slices_in_dimension[i].value,
                uniform_row_length=rp.uniform_row_length,
                nvals=num_slices_in_dimension[i + 1].value,
                dtype=rp.dtype) for i, rp in enumerate(row_partitions)
        ]

      self._static_inner_shape = static_inner_shape
      self._inner_shape = tensor_spec.TensorSpec(
          [inner_rank], dtype=dtype)
      self._row_partitions = row_partitions

    def __repr__(self):
      return (
          f"DynamicRaggedShape.Spec(row_partitions={self._row_partitions!r}, " +
          f"static_inner_shape={self._static_inner_shape!r}, " +
          f"dtype={self.dtype!r})")

    @classmethod
    def from_value(cls, value: Any) -> "DynamicRaggedShape.Spec":
      """Create a Spec from a DynamicRaggedShape."""
      # super().from_value(...) creates an object, but there is no validation.
      # No methods can be trusted on the object, just the properties.
      initial = super(DynamicRaggedShape.Spec, cls).from_value(value)

      # However, since value is a DynamicRaggedShape, we
      # can guarantee that initial._inner_shape.shape.rank == 1

      # Moreover, if inner_shape.shape[0] is not None, then
      # static_inner_shape.rank is not None.

      return DynamicRaggedShape.Spec(
          row_partitions=initial._row_partitions,
          static_inner_shape=initial._static_inner_shape,
          dtype=initial._inner_shape.dtype)

    # TODO(martinz): it is unclear what the default uniformity of RowPartitions
    # should be, so I am moving this to experimental until we figure it out.
    # Also, while I have specified this is meant to represent a shape of a
    # proper Tensor instead of a RaggedTensor, this is also subject to
    # interpretation.
    @classmethod
    def _from_tensor_shape(cls,
                           shape: Any,
                           num_row_partitions: int,
                           dtype: dtypes.DType) -> "DynamicRaggedShape.Spec":
      """Creates a `DynamicRaggedShape.Spec` corresponding to a `tf.TensorShape`.

      It is assumed that this is a `tf.TensorShape` coming from a
      `tf.TensorSpec`, not from `RaggedTensor.shape`.

      In addition to the shape, we need to know the number of row partitions,
      and the dtype used in the shape (tf.int32 or tf.int64).

      Within the dimensions that are partitioned, all dimensions are assumed
      to be uniform.

      Args:
        shape: a TensorShape.
        num_row_partitions: the ragged rank of the RaggedShape.
        dtype: the dtype of the shape (not the tensor); tf.int64 or tf.int32.

      Returns:
        a DynamicRaggedShape.Spec representing a TensorShape.
      """
      if dtype != dtypes.int32 and dtype != dtypes.int64:
        raise ValueError("dtype must be tf.int32 or tf.int64")

      shape = tensor_shape.as_shape(shape)
      if shape.rank is None:
        row_partitions = [
            RowPartitionSpec(dtype=dtype) for _ in range(num_row_partitions)
        ]
        return DynamicRaggedShape.Spec(
            row_partitions=row_partitions,
            static_inner_shape=tensor_shape.TensorShape(None),
            dtype=dtype)

      if shape.rank <= 1:
        # Create a scalar or vector shape.
        if num_row_partitions:
          raise ValueError("num_row_partitions should be zero " +
                           "if shape is a scalar or vector.")
        return DynamicRaggedShape.Spec(
            row_partitions=[], static_inner_shape=shape, dtype=dtype)

      if shape.rank <= num_row_partitions:
        raise ValueError("num_row_partitions must be less than rank")

      num_elements_so_far = tensor_shape.dimension_value(shape[0])
      rp_specs = []
      for i in range(num_row_partitions):
        current_dim = tensor_shape.dimension_value(shape[i + 1])
        if current_dim is None or num_elements_so_far is None:
          nvals = None
        else:
          nvals = num_elements_so_far * current_dim
        rp_specs.append(RowPartitionSpec(
            nrows=num_elements_so_far,
            nvals=nvals,
            uniform_row_length=current_dim,
            dtype=dtype))
        num_elements_so_far = nvals

      static_inner_shape = tensor_shape.TensorShape(
          [num_elements_so_far]) + shape[num_row_partitions + 1:]
      return DynamicRaggedShape.Spec(
          row_partitions=rp_specs,
          static_inner_shape=static_inner_shape,
          dtype=dtype)

    @classmethod
    def _from_spec(
        cls,
        spec: Union["DynamicRaggedShape.Spec", ragged_tensor.RaggedTensorSpec,
                    tensor_spec.TensorSpec],
        dtype: dtypes.DType = dtypes.int64) -> "DynamicRaggedShape.Spec":
      """Create a TypeSpec for the shape of an object with a given TypeSpec.

      I.e., if `x_spec = tf.type_spec_from_value(x)`, then
      `DynamicRaggedShape.from_spec(x_spec)` returns a TypeSpec compatible with
      `tf.type_spec_from_value(tf.shape(x))`.

      >>> rt = tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
      >>> rt_spec = tf.type_spec_from_value(rt)
      >>> rt_shape = DynamicRaggedShape.from_tensor(rt)

      >>> shape_spec_1 = tf.type_spec_from_value(rt_shape)
      >>> shape_spec_2 = DynamicRaggedShape.Spec._from_spec(rt_spec)
      >>> assert shape_spec_1.is_compatible_with(shape_spec_2)

      Args:
        spec: a Spec of a Tensor or RaggedTensor.
        dtype: the default dtype (if necessary).

      Returns:
        A Spec of the shape of a Tensor or RaggedTensor.

      """
      # TODO(martinz): Add StructuredTensor.Spec when its easy.
      if isinstance(spec, DynamicRaggedShape.Spec):
        return spec
      elif isinstance(spec, ragged_tensor.RaggedTensorSpec):
        return cls._from_tensor_shape(spec.shape,
                                      spec.ragged_rank,
                                      spec.row_splits_dtype)
      elif isinstance(spec, tensor_spec.TensorSpec):
        return cls._from_tensor_shape(shape=spec.shape,
                                      num_row_partitions=0,
                                      dtype=dtype)

    @property
    def dtype(self) -> dtypes.DType:
      return self._inner_shape.dtype

    @property
    def inner_rank(self) -> Optional[int]:
      if self._static_inner_shape.rank is not None:
        return self._static_inner_shape.rank
      if self._inner_shape.shape.rank is None:
        return None
      return tensor_shape.dimension_value(self._inner_shape.shape[0])

    @property
    def num_row_partitions(self) -> int:
      return len(self._row_partitions)

    @property
    def rank(self) -> Optional[int]:
      inner_rank = self.inner_rank
      return None if inner_rank is None else inner_rank + self.num_row_partitions

    def _dimension(self, index: int) -> Optional[int]:
      """Get the size of dimension index, if known statically."""
      if index == 0:
        if self._row_partitions:
          return self._row_partitions[0].nrows
        elif self.inner_rank is None:
          return None
        elif self.inner_rank == 0:
          raise ValueError("Index out of range: 0.")
        else:
          return tensor_shape.dimension_value(self._static_inner_shape[0])
      if index <= len(self._row_partitions):
        return self._row_partitions[index - 1].uniform_row_length

      relative_index = index - self.num_row_partitions

      if self.inner_rank is None:
        return None
      elif self.inner_rank <= relative_index:
        raise ValueError(f"Index out of range: {index}.")
      else:
        return tensor_shape.dimension_value(
            self._static_inner_shape[relative_index])

    def _num_slices_in_dimension(self, axis: int) -> Optional[int]:
      """The total size of a dimension (like nvals).

      This is a static version of DynamicRaggedShape._num_slices_in_dimension()

      Example:

      ```
      shape = DynamicRaggedShape.Spec(
        _row_partitions=[
          RowPartitionSpec(nrows=3, nvals=14, dtype=tf.int32)
          RowPartitionSpec(nrows=14, nvals=25, dtype=tf.int32)

        ],
        _static_inner_shape=tf.TensorShape([25, 3, 4]),
        _inner_shape=tf.TensorSpec(tf.TensorShape([3]), dtype=tf.int32))
      shape._num_slices_in_dimension(0) = 3
      shape._num_slices_in_dimension(1) = 14
      shape._num_slices_in_dimension(2) = 25
      shape._num_slices_in_dimension(3) = 3
      shape._num_slices_in_dimension(4) = 4
      shape._num_slices_in_dimension(-2) = 3
      ```

      Args:
        axis: the last dimension to include.

      Returns:
        the number of values in a dimension.
      """
      if not isinstance(axis, int):
        raise TypeError("axis must be an integer")
      axis = array_ops.get_positive_axis(axis, self.rank, ndims_name="rank")

      if axis == 0:
        return self._dimension(0)
      if axis <= self.num_row_partitions:
        # TODO(martinz): use nvals OR nrows, whichever is defined.
        return self._row_partitions[axis - 1].nvals
      remainder = axis - (self.num_row_partitions - 1)
      head_inner_shape = self._static_inner_shape[:remainder]
      return head_inner_shape.num_elements()

    def with_dtype(self, dtype: dtypes.DType) -> "DynamicRaggedShape.Spec":
      """Return the same spec, but with a different DType."""
      new_rp_specs = [rp.with_dtype(dtype) for rp in self._row_partitions]
      return DynamicRaggedShape.Spec(
          row_partitions=new_rp_specs,
          static_inner_shape=self._static_inner_shape,
          dtype=dtype)

    def _merge_with(
        self,
        other: "DynamicRaggedShape.Spec") -> "DynamicRaggedShape.Spec":
      """Merges all information between two specs.

      Specs are expected to represent the same information modulo
      num_row_partitons.

      If the specs are of different ranks, then fail.

      Args:
        other: another Spec of the same rank.

      Returns:
        a Spec with the union of information.
      """
      max_num_row_partitions = max(self.num_row_partitions,
                                   other.num_row_partitions)
      a = self._with_num_row_partitions(max_num_row_partitions)
      b = other._with_num_row_partitions(max_num_row_partitions)

      new_rp = [
          a._merge_with(b)
          for (a, b) in zip(a._row_partitions, b._row_partitions)
      ]

      new_static_inner_shape = a._static_inner_shape.merge_with(
          b._static_inner_shape)

      dtype = b.dtype if (a.dtype == dtypes.int32) else dtypes.int64

      return DynamicRaggedShape.Spec(
          new_rp, new_static_inner_shape, dtype=dtype)

    def _with_num_row_partitions(
        self,
        new_num_row_partitions: int) -> "DynamicRaggedShape.Spec":
      """Change the number of row partitions in the spec."""
      rank = self.rank
      if rank is None:
        raise ValueError(
            "Changing num_row_partitions with unknown rank unsupported")
      if new_num_row_partitions > max(rank - 1, 0):
        raise ValueError("Number of row partitions too large")
      if new_num_row_partitions < 0:
        raise ValueError("Number of row partitions negative")
      if self.num_row_partitions == new_num_row_partitions:
        return self
      elif self.num_row_partitions < new_num_row_partitions:
        # TODO(martinz): Consider swapping.
        rp_delta = new_num_row_partitions - self.num_row_partitions
        tail_shape = DynamicRaggedShape.Spec._from_tensor_shape(
            self._static_inner_shape, rp_delta, self.dtype)
        return DynamicRaggedShape.Spec(
            row_partitions=self._row_partitions + tail_shape._row_partitions,
            static_inner_shape=tail_shape._static_inner_shape,
            dtype=self.dtype)
      else:
        assert self.num_row_partitions > new_num_row_partitions
        new_row_partitions = self._row_partitions[:new_num_row_partitions]
        last_row_partition = new_row_partitions[-1]
        old_row_partitions = self._row_partitions[new_num_row_partitions:]
        new_static_inner_shape = (
            tensor_shape.TensorShape(
                [last_row_partition.nvals] +
                [x.uniform_row_length for x in old_row_partitions]) +
            self._static_inner_shape[1:])
        return DynamicRaggedShape.Spec(
            new_row_partitions, new_static_inner_shape, self.dtype)

    def _set_rank_if_unknown(self, new_rank: int) -> "DynamicRaggedShape.Spec":
      """Ensures this has a known rank at least new_rank."""
      if new_rank is None:
        raise TypeError("new_rank is None, but expected int")
      if new_rank < 0:
        raise ValueError("Rank must be non-negative")
      current_rank = self.rank
      if current_rank is not None and current_rank < new_rank:
        raise ValueError(
            "Rank is {current_rank}, expected at least {new_rank}.".format(
                current_rank=current_rank, new_rank=new_rank))

      if current_rank is not None:
        return self

      if self._row_partitions:
        new_inner_rank = max(new_rank - self.num_row_partitions, 1)
        first_dim = self._row_partitions[-1].nvals
        static_inner_shape = tensor_shape.TensorShape(
            [first_dim] + [None] * (new_inner_rank - 1))
      else:
        static_inner_shape = tensor_shape.TensorShape([None] * new_rank)

      return DynamicRaggedShape.Spec(
          row_partitions=self._row_partitions,
          static_inner_shape=static_inner_shape,
          dtype=self.dtype)

    def _truncate(self, new_rank: int) -> "DynamicRaggedShape.Spec":
      """Truncate a ragged shape spec.

      For example, if the original spec s was for a shape:
      [3, [4, 1], 2, 7]

      Then truncate_dynamic_ragged_shape_spec(s, 3) is a spec for:
      [3, [4, 1], 2]

      Args:
        new_rank: the new rank

      Returns:
        A truncated DynamicRaggedShape.Spec.
      """
      if self.rank is None:
        return self._set_rank_if_unknown(new_rank)._truncate(new_rank)

      if new_rank == 0:
        return DynamicRaggedShape.Spec._from_tensor_shape([], 0, self.dtype)

      if new_rank == 1:
        vector_size = self._dimension(0)
        return DynamicRaggedShape.Spec._from_tensor_shape([vector_size], 0,
                                                          self.dtype)

      if new_rank < self.num_row_partitions + 1:
        new_row_partitions = self._row_partitions[:new_rank - 1]
        new_static_inner_shape = tensor_shape.TensorShape(
            [new_row_partitions[-1].nvals])
        return DynamicRaggedShape.Spec(
            row_partitions=new_row_partitions,
            static_inner_shape=new_static_inner_shape,
            dtype=self.dtype)
      else:
        remainder = new_rank - self.num_row_partitions
        new_static_inner_shape = self._static_inner_shape[:remainder]
        return DynamicRaggedShape.Spec(
            row_partitions=self._row_partitions,
            static_inner_shape=new_static_inner_shape,
            dtype=self.dtype)

    def _to_tensor_shape(self):
      """Get a tensor shape corresponding to this type."""
      alt = self
      if alt._static_inner_shape.rank is None:
        return tensor_shape.TensorShape(None)
      if alt._static_inner_shape.rank == 0:
        assert not alt._row_partitions
        return alt._static_inner_shape
      prefix = [alt._dimension(0)]
      prefix.extend([rp.uniform_row_length for rp in alt._row_partitions])
      suffix = alt._static_inner_shape[1:]
      return tensor_shape.TensorShape(prefix) + suffix


def broadcast_dynamic_shape(shape_x: DynamicRaggedShape,
                            shape_y: DynamicRaggedShape) -> DynamicRaggedShape:
  """Returns the shape formed by broadcasting two shapes to be compatible.

  1. If shape_x and shape_y both have row_partitions, then fail if their dtypes
     don't match.
  2. If neither has row_partitions and they have different dtypes,
     go with int64.
  3. If one has row_partitions, go with that dtype.

  Args:
    shape_x: A `DynamicRaggedShape`
    shape_y: A `DynamicRaggedShape`

  Returns:
    A `DynamicRaggedShape`.
  Raises:
    ValueError: If `shape_x` and `shape_y` are not broadcast-compatible.
  """
  if not isinstance(shape_x, DynamicRaggedShape):
    raise TypeError("shape_x must be a DynamicRaggedShape")
  if not isinstance(shape_y, DynamicRaggedShape):
    raise TypeError("shape_y must be a DynamicRaggedShape")

  return broadcast_dynamic_shape_extended(shape_x, shape_y)[0]


def broadcast_to(rt_input, shape: DynamicRaggedShape):
  """Broadcasts a potentially ragged tensor to a ragged shape.

  Tiles `rt_input` as necessary to match the given shape.

  Behavior is undefined if `rt_input` is not broadcast-compatible with `shape`.

  Args:
    rt_input: The potentially ragged tensor to broadcast.
    shape: A `DynamicRaggedShape`

  Returns:
    A potentially ragged tensor whose values are taken from
    `rt_input`, and whose shape matches `shape`.
  """
  if not isinstance(shape, DynamicRaggedShape):
    raise TypeError("shape must be a DynamicRaggedShape")
  rt_input = ragged_tensor.convert_to_tensor_or_ragged_tensor(rt_input)
  origin_shape = None
  if ragged_tensor.is_ragged(rt_input):
    if shape.num_row_partitions != 0:
      if rt_input.row_splits.dtype != shape.dtype:
        raise ValueError("Cannot coerce row_splits.dtype")
    else:
      shape = shape.with_dtype(rt_input.row_splits.dtype)
    origin_shape = DynamicRaggedShape.from_tensor(rt_input)
  else:
    if shape.num_row_partitions != 0:
      origin_shape = DynamicRaggedShape.from_tensor(rt_input, dtype=shape.dtype)
    else:
      origin_shape = DynamicRaggedShape.from_tensor(rt_input,
                                                    dtype=dtypes.int64)
      shape = shape.with_dtype(dtype=dtypes.int64)

  broadcaster = _get_broadcaster(origin_shape, shape)
  return broadcaster.broadcast(rt_input)


def broadcast_dynamic_shape_extended(
    a: DynamicRaggedShape, b: DynamicRaggedShape
):  #  -> Tuple[DynamicRaggedShape, _Broadcaster, _Broadcaster]
  """Gets the smallest shape to which a and b can broadcast.

  In order to create the smallest shape, one must also do most of the
  work to figure out how to transform from the shapes given. Thus, in addition
  to returning the shape, it also creates transformations from the
  original shapes to the result.

  This is the equivalent of:

  c = broadcast_dynamic_shape(a, b)
  ac = get_broadcaster(a, c)
  bc = get_broadcaster(b, c)
  return (c, ac, bc)

  Args:
    a: a DynamicRaggedShape
    b: a DynamicRaggedShape

  Returns:
    A triple of a shape and two broadcasters.
  """
  if a.row_partitions and b.row_partitions:
    if a.dtype != b.dtype:
      raise ValueError("Dtypes don't match")
  elif a.dtype != b.dtype:
    if a.row_partitions:
      b = b.with_dtype(a.dtype)
    elif b.row_partitions:
      a = a.with_dtype(b.dtype)
    else:
      a = a.with_dtype(dtypes.int64)
      b = b.with_dtype(dtypes.int64)

  if (a.rank is None or b.rank is None):
    raise ValueError("Unable to broadcast: unknown rank")
  elif a.rank == 0:
    return (b, _Broadcaster(a, b, []), _get_identity_broadcaster(b))
  elif b.rank == 0:
    return (a, _get_identity_broadcaster(a), _Broadcaster(b, a, []))
  elif a.rank == 1 and b.rank == 1:
    [a_layer, b_layer,
     target] = _broadcast_dynamic_shape_one_layer(a.inner_shape, b.inner_shape)
    target_shape = DynamicRaggedShape._from_inner_shape(target)  # pylint: disable=protected-access
    return (target_shape, _Broadcaster(a, target_shape, [a_layer]),
            _Broadcaster(b, target_shape, [b_layer]))

  if a.rank > b.rank:
    (c, bc, ac) = _broadcast_dynamic_shape_extended_helper(b, a)  # pylint: disable=arguments-out-of-order

    return (c, ac, bc)

  return _broadcast_dynamic_shape_extended_helper(a, b)


def _row_partitions_identical(shape_a, shape_b):
  """Returns True iff all row_partitions in shapes are identical."""
  return ((shape_a.num_row_partitions == shape_b.num_row_partitions) and all(
      a is b for a, b in zip(shape_a.row_partitions, shape_b.row_partitions)))


# TODO(martinz): Preserve shapes better (see CL/414806185)
@dispatch.dispatch_for_binary_elementwise_apis(ragged_tensor.RaggedOrDense,
                                               ragged_tensor.RaggedOrDense)
def ragged_binary_elementwise_op_impl(op, x, y):
  """Binary elementwise api handler for RaggedTensors."""
  x_is_ragged = ragged_tensor.is_ragged(x)
  y_is_ragged = ragged_tensor.is_ragged(y)

  # Convert args to tensors.
  x = ragged_tensor.convert_to_tensor_or_ragged_tensor(
      x, preferred_dtype=(y.dtype if y_is_ragged else None))
  y = ragged_tensor.convert_to_tensor_or_ragged_tensor(
      y, preferred_dtype=x.dtype)

  if x_is_ragged and y_is_ragged:
    x, y = ragged_tensor.match_row_splits_dtypes(x, y)

  if ((x_is_ragged and y_is_ragged) or
      (x_is_ragged and x.flat_values.shape.ndims <= y.shape.ndims) or
      (y_is_ragged and y.flat_values.shape.ndims <= x.shape.ndims)):
    shape_x = DynamicRaggedShape.from_tensor(x)
    shape_y = DynamicRaggedShape.from_tensor(y)
    if shape_x.dtype != shape_y.dtype:
      if not x_is_ragged:
        shape_x = shape_x.with_dtype(shape_y.dtype)
      elif not y_is_ragged:
        shape_y = shape_y.with_dtype(shape_x.dtype)

    if _row_partitions_identical(shape_x, shape_y):
      # At this point, both x and y must be ragged.
      return shape_x._add_row_partitions(  # pylint: disable=protected-access
          op(x.flat_values, y.flat_values), validate=False)

    (shape_z, bcast_xz,
     bcast_yz) = broadcast_dynamic_shape_extended(shape_x, shape_y)
    x_new_flat = bcast_xz.broadcast_flat_values(x, inner_dimensions=False)
    y_new_flat = bcast_yz.broadcast_flat_values(y, inner_dimensions=False)
    z_flat = op(x_new_flat, y_new_flat)
    return shape_z._add_row_partitions(z_flat, validate=True)  # pylint: disable=protected-access

  x_values = x.flat_values if ragged_tensor.is_ragged(x) else x
  y_values = y.flat_values if ragged_tensor.is_ragged(y) else y
  mapped_values = op(x_values, y_values)
  if isinstance(mapped_values, bool):
    return mapped_values  # Special case for tensor_equals.
  if ragged_tensor.is_ragged(x):
    return x.with_flat_values(mapped_values)
  else:
    return y.with_flat_values(mapped_values)


@dispatch.dispatch_for_binary_elementwise_assert_apis(
    ragged_tensor.RaggedOrDense, ragged_tensor.RaggedOrDense)
def ragged_binary_elementwise_assert_op_impl(op, x, y):
  """Binary elementwise assert api handler for RaggedTensors.

  This handles binary assert operations for ragged tensors. Compared with
  `ragged_binary_elementwise_op_impl`, this handler does not compute a ragged
  tensor as output. Instead, it applies the assert operation `op` to input
  tensors based on their ragged shapes and flat_values, and returns the result
  of the assertion operation.

  Args:
    op: a binary assert operation on Tensors.
    x: something that can be coerced to a Tensor or RaggedTensor.
    y: something that can be coerced to a Tensor or RaggedTensor.

  Returns:
    the result of the assertion operation.

  """
  x_is_ragged = ragged_tensor.is_ragged(x)
  y_is_ragged = ragged_tensor.is_ragged(y)

  # Convert args to tensors.
  x = ragged_tensor.convert_to_tensor_or_ragged_tensor(
      x, preferred_dtype=(y.dtype if y_is_ragged else None))
  y = ragged_tensor.convert_to_tensor_or_ragged_tensor(
      y, preferred_dtype=x.dtype)

  if x_is_ragged and y_is_ragged:
    x, y = ragged_tensor.match_row_splits_dtypes(x, y)

  if ((x_is_ragged and y_is_ragged) or
      (x_is_ragged and x.flat_values.shape.ndims <= y.shape.ndims) or
      (y_is_ragged and y.flat_values.shape.ndims <= x.shape.ndims)):
    shape_x = DynamicRaggedShape.from_tensor(x)
    shape_y = DynamicRaggedShape.from_tensor(y)
    if shape_x.dtype != shape_y.dtype:
      if not x_is_ragged:
        shape_x = shape_x.with_dtype(shape_y.dtype)
      elif not y_is_ragged:
        shape_y = shape_y.with_dtype(shape_x.dtype)

    if _row_partitions_identical(shape_x, shape_y):
      # At this point, both x and y must be ragged.
      return op(x.flat_values, y.flat_values)

    (_, bcast_xz, bcast_yz) = broadcast_dynamic_shape_extended(shape_x, shape_y)
    x_new_flat = bcast_xz.broadcast_flat_values(x, inner_dimensions=False)
    y_new_flat = bcast_yz.broadcast_flat_values(y, inner_dimensions=False)
    return op(x_new_flat, y_new_flat)

  x_values = x.flat_values if ragged_tensor.is_ragged(x) else x
  y_values = y.flat_values if ragged_tensor.is_ragged(y) else y
  return op(x_values, y_values)


def _find_dtype_helper(value, preferred):
  """Helper for _find_dtype."""
  if preferred is not None:
    return preferred
  elif isinstance(value, RowPartition):
    return value.dtype
  elif isinstance(value, dtypes.DType):
    return value
  elif isinstance(value, int):
    return None
  elif isinstance(value, list):
    return None
  elif isinstance(value, tuple):
    return None
  elif isinstance(value, core.Tensor):
    return value.dtype
  return value.dtype


def _find_dtype(value, preferred):
  """Returns the preferred dtype of value or preferred if preferred != None.

  This is used as an operator to pass over multiple objects in decreasing order
  of priority until there is a preferred dtype for one. For example, if you were
  adding three tensor-ish things (some tensors, some lists), and needed a
  preferred dtype, you could use this as:

  def adding(a, b, c, dtype = None):
    dtype = _find_dtype(a, dtype)
    dtype = _find_dtype(b, dtype)
    dtype = _find_dtype(c, dtype)
    if dtype is None:
      dtype = tf.float32
    ...Code continues here...

  Args:
    value: a list, value, RowPartition, or tensor.
    preferred: a given dtype. If not None, this will be returned.

  Returns:
    an optional dtype.
  """
  result = _find_dtype_helper(value, preferred)
  if (result == dtypes.int64 or result == dtypes.int32 or result is None):
    return result
  raise ValueError("Illegal dtype: " + str(result))


def _find_dtype_iterable(
    iterable: Iterable[Any],
    dtype: Optional[dtypes.DType]) -> Optional[dtypes.DType]:
  """Find the preferred dtype of a list of objects.

  This will go over the iterable, and use the first object with a preferred
  dtype. The dtype passed has highest priority if it is not None.

  Args:
    iterable: an iterable with things that might have a dtype.
    dtype: an overriding dtype, or None.

  Returns:
    an optional dtype.
  """
  if dtype is not None:
    return dtype
  for x in iterable:
    dtype = _find_dtype(x, dtype)
  return dtype


class _LayerBroadcaster(abc.ABC):
  """A broadcaster of a single layer.

  Although this class does not literally contain a gather_index, the reference
  implementation is defined through a gather_index. Thus, any subclasses should
  first define the gather_index property. Other functions can be overridden
  for optimization, but it should not change the behavior.
  """

  @property
  @abc.abstractmethod
  def gather_index(self):
    """Returns a 1D tensor.

    The size of the 1D tensor is equal to the destination size.

    The ith element of the result is the index of the source of the ith element.
    """
    pass

  @property
  def dtype(self):
    """Returns the dtype of the broadcast."""
    return self.gather_index.dtype

  @abc.abstractmethod
  def with_dtype(self, dtype):
    """Returns an identical _LayerBroadcaster with a different dtype."""
    pass

  def __repr__(self):
    return str(self.gather_index)

  @classmethod
  def from_gather_index(cls, gather_index):
    """Create a broadcaster from a gather_index."""
    return _GatherLayerBroadcaster(gather_index)

  @classmethod
  def first_layer(cls, nrows_source, nrows_target):
    """Create a broadcaster from a gather_index."""
    gather_index = _first_layer_gather_index(nrows_source, nrows_target)
    return _LayerBroadcaster.from_gather_index(gather_index)

  @classmethod
  def get_singleton_broadcaster(cls, target_size):
    """Broadcast from 1 element to target_size elements."""
    return _LayerBroadcaster.from_gather_index(
        array_ops.zeros(target_size, dtype=target_size.dtype))

  @abc.abstractmethod
  def with_dependencies(self, checks):
    """Add dependencies to a _LayerBroadcaster.

    Args:
      checks: a list of ops that need to be run before any tensors from the
        Broadcaster are used.

    Returns:
      a copy of this _LayerBroadcaster with dependencies added.
    """
    pass

  @classmethod
  def get_identity_broadcaster(cls, nvals, dtype=None):
    """Create an identity broadcaster.

    TODO(martinz): an identity broadcaster can be far more efficient than a
    generic broadcaster. Add an optimized implementation.
    Args:
      nvals: the number of values for the broadcaster.
      dtype: the dtype of the broadcaster, or None to use the dtype of nvals.
    Returns:
      an identity broadcaster from [0....nvals-1] to [0...nvals-1]
    """
    return _GatherLayerBroadcaster(math_ops.range(nvals, dtype=dtype))

  def broadcast_tensor(self, tensor):
    """Broadcast from a dense tensor.

    It is assumed that the first axis of the dense tensor is indexed by the
    source shape, and at the end, the first axis of the dense tensor is
    indexed by the destination shape.

    Args:
      tensor: a dense tensor.

    Returns:
      A dense tensor.
    """
    return array_ops.gather(tensor, self.gather_index)

  def dest_nrows(self):
    """Return the number of rows in the resulting gather, or None if tiling."""
    return math_ops.cast(
        array_ops.shape(self.gather_index)[0], dtype=self.dtype)

  def broadcast_row_partition(self, rp):
    """Return a new shape where the rows are broadcasted.

        *--self--->*
        |          |
        rp       result
        |          |
        V          V
        *--------->*

    This is equivalent to:
      return RowPartition.from_row_lengths(self.broadcast(rp.row_lengths()))

    However, if the shape has uniform row length, then that property is
    maintained.

    Args:
      rp: a row partition.

    Returns:
      a RowPartition representing a broadcast version of this row partition.
    """
    if not rp.is_uniform():
      return RowPartition.from_row_lengths(
          self.broadcast_tensor(rp.row_lengths()))
    else:
      return RowPartition.from_uniform_row_length(
          rp.uniform_row_length(),
          nvals=rp.uniform_row_length() * self.dest_nrows(),
          nrows=self.dest_nrows())

  def next_layer(self, original_rp, broadcast_rp):
    r"""Create the next layer gather_index whether or not a broadcast happens.

       *---------self------->*
       |                     |
    original_rp           broadcast_rp
       |                     |
      \|/                   \|/
       *--next_broadcaster-->*
    Args:
      original_rp: the original row partition.
      broadcast_rp: the target row partition.

    Returns:
      the gather_index for next_broadcaster.

    """
    gather_index = _next_layer_gather_index(self, original_rp, broadcast_rp)
    return _LayerBroadcaster.from_gather_index(gather_index)


class _GatherLayerBroadcaster(_LayerBroadcaster):
  """Implements _LayerBroadcaster with an explicit gather_index.

  For example, suppose that the source shape is:
  [*],[*,*]
  And the target shape is:
  [*],[*,*],[*],[*,*]
  Then, this can be represented with a map:
  [0,1,2,0,1,2]

  """

  def __init__(self, gather_index):
    gather_index = ops.convert_to_tensor(gather_index)
    if (gather_index.dtype != dtypes.int64 and
        gather_index.dtype != dtypes.int32):
      raise ValueError("gather_index must be int64 or int32")
    self._gather_index = gather_index

  @property
  def gather_index(self):
    return self._gather_index

  def with_dtype(self, dtype):
    return _GatherLayerBroadcaster(math_ops.cast(self._gather_index, dtype))

  def with_dependencies(self, checks):
    new_gather_index = control_flow_ops.with_dependencies(
        checks, self._gather_index)
    return _GatherLayerBroadcaster(new_gather_index)


class _Broadcaster:
  """A _Broadcaster represents a transformation from one shape to another.

  It provides a transform for each axis of the source shape to the
  corresponding axis of the destination shape.

  """

  def __init__(self,
               source_shape,
               target_shape,
               layer_broadcasters,
               dtype=None):
    """Create a broadcaster.

    Do not call directly.
    The source_shape, target_shape, and layer_broadcasters are converted
    to have the same dtype.

    Note: source_shape.rank and target_shape.rank must be known.
    Args:
      source_shape: the source DynamicRaggedShape
      target_shape: the target DynamicRaggedShape
      layer_broadcasters: List[_LayerBroadcaster] of length source_shape.rank.
      dtype: the preferred dtype of the broadcaster.

    Raises:
      TypeError: if the input types don't match.
    """
    if not isinstance(source_shape, DynamicRaggedShape):
      raise TypeError("source_shape is not a DynamicRaggedShape")
    if not isinstance(target_shape, DynamicRaggedShape):
      raise TypeError("target_shape is not a DynamicRaggedShape")
    if not isinstance(layer_broadcasters, list):
      raise TypeError("layer_broadcasters not a list: " +
                      str(layer_broadcasters))
    for bc in layer_broadcasters:
      if not isinstance(bc, _LayerBroadcaster):
        raise TypeError("Not a LayerBroadcaster: " + str(bc))

    dtype = _find_dtype(source_shape, dtype)
    dtype = _find_dtype(target_shape, dtype)
    dtype = _find_dtype_iterable(layer_broadcasters, dtype)
    dtype = _find_dtype(dtypes.int64, dtype)
    self._source_shape = source_shape.with_dtype(dtype)
    self._target_shape = target_shape.with_dtype(dtype)
    self._layer_broadcasters = [x.with_dtype(dtype) for x in layer_broadcasters]

  def __repr__(self):
    return ("{src_shape:" + str(self._source_shape) + ", target_shape:" +
            str(self._target_shape) + " layer_broadcasters: " +
            str(self._layer_broadcasters) + "}")

  def with_dtype(self, dtype):
    """Return a copy of this Broadcaster with a different dtype."""
    return _Broadcaster(self._source_shape, self._target_shape,
                        self._layer_broadcasters, dtype)

  @property
  def source_shape(self):
    return self._source_shape

  @property
  def target_shape(self):
    return self._target_shape

  @property
  def dtype(self):
    return self._source_shape.dtype

  def _target_inner_shape_int32(self):
    new_inner_shape = self.target_shape.inner_shape
    if new_inner_shape.dtype == dtypes.int64:
      new_inner_shape = math_ops.cast(new_inner_shape, dtype=dtypes.int32)
    return new_inner_shape

  # pylint:disable=protected-access
  def broadcast_flat_values(self, rt, inner_dimensions=True):
    """flat_values of a ragged tensor broadcast to target_shape.

    If inner_dimensions==True, then the result is a dense tensor with shape
    target_shape.inner_shape, the flat values of the broadcasted shape.

    If you add target_shape.row_partitions, you will get the full broadcasted
    shape.

    If inner_dimensions==False, the result is a dense tensor that satsifies
    certain properties:
    1. broadcast_to(result, target_shape.inner_shape) will give the result
       if inner_dimensions==True.
    2. Either (a) (result.rank < target_shape.inner_rank)
       or (b) (result.shape[0] == target_shape.inner_shape[0]).
    3. result.rank = min(target_shape.inner_rank, rt.rank)
    4. For i < target_shape.inner_rank - 1, and i < rt.rank,
       and if rt.shape[-i]!=1, then result.shape[-i]=target_shape[-i].
    Args:
      rt: a ragged or dense tensor.
      inner_dimensions: if true, broadcast the inner dimensions as well.

    Returns:
      a dense tensor
    """
    if ragged_tensor.is_ragged(rt):
      rt = rt.flat_values
    # If rt was a regular tensor, it is its own flat_values.
    if self.target_shape.rank == 0:
      return rt
    inner_rank = self.target_shape.inner_rank
    if inner_rank > self._source_shape.rank:
      # The dense rank is larger than the whole shape. So, we make the shape
      # dense.
      if self.source_shape.num_row_partitions > 0:
        rt = array_ops.reshape(
            rt, self.source_shape._alt_inner_shape(self.source_shape.rank))
      # rt.rank == self._source_shape.rank < inner_rank
      # Here, property 2a holds.
      if inner_dimensions:
        return array_ops.broadcast_to(rt, self._target_inner_shape_int32())
      return rt
    else:
      if self._source_shape.inner_rank != inner_rank:
        rt = array_ops.reshape(rt,
                               self._source_shape._alt_inner_shape(inner_rank))  # pylint:disable=protected-access
      # After the reshape, rt is flat_values with inner_rank.
      flat_broadcaster = self._layer_broadcasters[-inner_rank]
      rt = flat_broadcaster.broadcast_tensor(rt)
      # Here, property 2b holds.
      if inner_dimensions:
        rt = array_ops.broadcast_to(rt, self._target_inner_shape_int32())
      return rt

  def broadcast(self, rt):
    """Broadcast a tensor of source_shape to target_shape."""
    flat_values = self.broadcast_flat_values(rt)
    return self.target_shape._add_row_partitions(flat_values)  # pylint:disable=protected-access


def _get_layer_broadcasters_from_rps(zero_broadcaster, source_rps, target_rps):
  """Get LayerBroadcasters from RowPartitions.

           *--zero_broadcaster->*
           |                    |
         source_rps[0]     target_rps[0]
           |                    |
           V                    V
           *---result[1]------->*
           |                    |
         source_rps[1]     target_rps[1]
           |                    |
           V                    V
           *---result[2]------->*
                  .
                  .
                  .
           *---result[k-1]----->*
           |                    |
         source_rps[k]     target_rps[k]
           |                    |
           V                    V
           *---result[k]------->*

  Note: result[0] = zero_broadcaster

  Args:
    zero_broadcaster: a broadcaster between the source and target row
      partitions' rows, and equal to result[0].
    source_rps: source row partitions.
    target_rps: target row partitions (same length as source_rps).

  Returns:
    result: a list of LayerBroadcasters.
  """
  if not isinstance(zero_broadcaster, _LayerBroadcaster):
    raise TypeError("Not a _LayerBroadcaster: " + str(zero_broadcaster))
  assert len(source_rps) == len(target_rps)
  if not source_rps:
    return [zero_broadcaster]
  next_broadcaster = zero_broadcaster.next_layer(source_rps[0], target_rps[0])
  tail_broadcasters = _get_layer_broadcasters_from_rps(next_broadcaster,
                                                       source_rps[1:],
                                                       target_rps[1:])
  return [zero_broadcaster] + tail_broadcasters


def _get_broadcaster(source_shape, target_shape):
  """Get a _Broadcaster from source_shape to target_shape."""
  if source_shape.dtype != target_shape.dtype:
    raise ValueError("The source and target row_split dtypes should be equal")

  if (source_shape.rank is None or target_shape.rank is None):
    raise ValueError("Rank of source and target must be statically known")
  elif source_shape.rank > target_shape.rank:
    raise ValueError("Cannot broadcast to a shape with smaller rank")
  elif source_shape.rank == 0:
    return _Broadcaster(source_shape, target_shape, [])
  elif target_shape.rank == 1:
    assert source_shape.rank == 1
    layer = _LayerBroadcaster.first_layer(source_shape.inner_shape[0],
                                          target_shape.inner_shape[0])
    return _Broadcaster(source_shape, target_shape, [layer])

  assert source_shape.rank <= target_shape.rank
  assert target_shape.rank >= 2
  assert source_shape.rank >= 1

  source_rps = source_shape._as_row_partitions()  # pylint: disable=protected-access

  target_rps = target_shape._as_row_partitions()  # pylint: disable=protected-access

  assert len(target_rps) >= 1
  assert len(source_rps) <= len(target_rps)
  source_nrows = source_shape[0]
  if len(source_rps) < len(target_rps):
    # Note: this includes the case where len(source_rps)==0.
    # Here we begin at -1, one dimension before source_rps[0].
    # neg_one_source_rp  | neg_one_target_rp=target_rps[-(len(source_rps)+1)]
    # source_rps[0]      | target_rps[-len(source_rps)]
    # source_rps[1]      | target_rps[1-len(source_rps)]
    # ...                | ...
    # source_rps[-1]     | target_rps[-1]
    neg_one_source_rp = RowPartition.from_uniform_row_length(
        uniform_row_length=source_nrows, nrows=1, nvals=source_nrows)
    neg_one_target_rp = target_rps[-(len(source_rps) + 1)]
    neg_one_broadcaster = _LayerBroadcaster.get_singleton_broadcaster(
        neg_one_target_rp.nrows())
    zeroth_broadcaster = neg_one_broadcaster.next_layer(neg_one_source_rp,
                                                        neg_one_target_rp)
    target_rps_tail = target_rps[-len(source_rps):] if len(
        source_rps) >= 1 else []

    layers = _get_layer_broadcasters_from_rps(zeroth_broadcaster, source_rps,
                                              target_rps_tail)
    return _Broadcaster(source_shape, target_shape, layers)
  else:
    assert len(target_rps) == len(source_rps)
    zeroth_broadcaster = _LayerBroadcaster.first_layer(source_rps[0].nrows(),
                                                       target_rps[0].nrows())
    layers = _get_layer_broadcasters_from_rps(zeroth_broadcaster, source_rps,
                                              target_rps)

    return _Broadcaster(source_shape, target_shape, layers)


def _get_identity_broadcaster(shape):
  """Gets a Broadcaster for two identical shapes."""
  if shape.rank is None:
    raise ValueError("Shape must have a defined rank")
  layers = [
      _LayerBroadcaster.get_identity_broadcaster(
          shape._num_slices_in_dimension(i)) for i in range(shape.rank)  # pylint: disable=protected-access
  ]
  return _Broadcaster(shape, shape, layers)


def _broadcast_dynamic_shape_one_layer(a, b):
  """Broadcast two vectors, given their shapes.

  Args:
    a: the number of rows in a.
    b: the number of rows in b.

  Returns:
    (layer_a, layer_b, target_shape)
    layer_a is a _LayerBroadcaster from a to the target_shape.
    layer_b is a _LayerBroadcaster from b to the target_shape.
    target_shape is the target_shape

  Raises:
    InvalidArgumentError if the shapes are not consistent.
  """
  a_0 = a[0]
  b_0 = b[0]

  def broadcast_from_a():
    # Assumes a_0 == 1
    a_layer = array_ops.zeros(b_0, dtype=b_0.dtype)
    b_layer = math_ops.range(b_0)
    target = b
    return [a_layer, b_layer, target]

  a_static = tensor_util.constant_value(a)
  if a_static is not None and a_static[0] == 1:
    [a_gi, b_gi, target] = broadcast_from_a()
    a_layer = _LayerBroadcaster.from_gather_index(a_gi)
    b_layer = _LayerBroadcaster.from_gather_index(b_gi)
    return [a_layer, b_layer, target]

  def broadcast_from_b():
    # Assumes b_0 == 1
    a_layer = math_ops.range(a_0)
    b_layer = array_ops.zeros(a_0, dtype=a_0.dtype)
    target = a
    return [a_layer, b_layer, target]

  b_static = tensor_util.constant_value(b)
  if b_static is not None and b_static[0] == 1:
    [a_gi, b_gi, target] = broadcast_from_b()
    a_layer = _LayerBroadcaster.from_gather_index(a_gi)
    b_layer = _LayerBroadcaster.from_gather_index(b_gi)
    return [a_layer, b_layer, target]

  def broadcast_noop():
    # Assumes a_0 == 1
    a_layer = math_ops.range(a_0)
    b_layer = math_ops.range(b_0)
    target = b
    return [a_layer, b_layer, target]

  can_broadcast_from_a = math_ops.equal(a_0, 1)
  can_broadcast_from_b = math_ops.equal(b_0, 1)

  def broadcast_not_from_a():
    return control_flow_ops.cond(
        can_broadcast_from_b, true_fn=broadcast_from_b, false_fn=broadcast_noop)

  nrows_equal = math_ops.equal(a_0, b_0)
  can_broadcast = math_ops.logical_or(
      can_broadcast_from_a,
      math_ops.logical_or(can_broadcast_from_b, nrows_equal))

  check_can_broadcast = check_ops.assert_equal(
      can_broadcast, True, message="Cannot broadcast")

  results = control_flow_ops.cond(
      can_broadcast_from_a,
      true_fn=broadcast_from_a,
      false_fn=broadcast_not_from_a)

  results = [
      control_flow_ops.with_dependencies([check_can_broadcast], x)
      for x in results
  ]
  [a_gi, b_gi, target] = results
  a_layer = _LayerBroadcaster.from_gather_index(a_gi)
  b_layer = _LayerBroadcaster.from_gather_index(b_gi)
  return [a_layer, b_layer, target]


def _broadcast_dynamic_shape_first_layer(a_0, b_0):
  """Broadcast the first layer of two dynamic shapes given the dimensions.

  Args:
    a_0: the number of rows in a.
    b_0: the number of rows in b.

  Returns:
    (use_a, layer_a, layer_b)
    where use_a is true if the target provably equals a, false otherwise.
    layer_a is a _LayerBroadcaster from a to the target.
    layer_b is a _LayerBroadcaster from b to the target.
  """
  def broadcast_from_a():
    # Assumes a_0 == 1
    a_layer = array_ops.zeros(b_0, dtype=b_0.dtype)
    b_layer = math_ops.range(b_0)
    return [a_layer, b_layer]

  static_a_0 = tensor_util.constant_value(a_0)
  static_b_0 = tensor_util.constant_value(b_0)
  if static_a_0 is not None:
    if static_a_0 == static_b_0:
      id_broadcaster = _LayerBroadcaster.get_identity_broadcaster(
          static_a_0, dtype=a_0.dtype)
      return [id_broadcaster, id_broadcaster]
    elif static_a_0 == 1:
      return [
          _LayerBroadcaster.get_singleton_broadcaster(b_0),
          _LayerBroadcaster.get_identity_broadcaster(b_0)
      ]

  if static_b_0 == 1:
    return [
        _LayerBroadcaster.get_identity_broadcaster(a_0),
        _LayerBroadcaster.get_singleton_broadcaster(a_0)
    ]

  def broadcast_from_b():
    # Assumes b_0 == 1
    a_layer = math_ops.range(a_0)
    b_layer = array_ops.zeros(a_0, dtype=a_0.dtype)
    return [a_layer, b_layer]

  def broadcast_noop():
    # Assumes a_0 == b_0
    a_layer = math_ops.range(a_0)
    b_layer = math_ops.range(b_0)
    return [a_layer, b_layer]

  can_broadcast_from_a = math_ops.equal(a_0, constant_op.constant(1, a_0.dtype))
  can_broadcast_from_b = math_ops.equal(b_0, constant_op.constant(1, b_0.dtype))

  def broadcast_not_from_a():
    return control_flow_ops.cond(
        can_broadcast_from_b, true_fn=broadcast_from_b, false_fn=broadcast_noop)

  # Ideally, this would only block control flow on broadcast_noop, but
  # the control flow doesn't seem to work.
  can_broadcast = math_ops.logical_or(
      math_ops.logical_or(can_broadcast_from_a, can_broadcast_from_b),
      math_ops.equal(a_0, b_0))

  result = control_flow_ops.cond(
      can_broadcast_from_a,
      true_fn=broadcast_from_a,
      false_fn=broadcast_not_from_a)

  return [
      _LayerBroadcaster.from_gather_index(
          control_flow_ops.with_dependencies(
              [check_ops.assert_equal(can_broadcast, True)], x)) for x in result
  ]


def _broadcast_half(
    ac_0: _LayerBroadcaster,
    a_1: RowPartition) -> Tuple[_LayerBroadcaster, RowPartition]:
  """Does a NOOP broadcast of a_1.

      *-ac_0-->*
      |        |
     a_1      c_1
      |        |
      V        V
      *-ac_1-->*

  Note that by definition this cannot fail: there is always a well-defined
  NOOP broadcast. This is usually intended as half of broadcasting two shapes
  together.
  Args:
    ac_0: previous LayerBroadcaster
    a_1: previous RowPartition

  Returns:
    [ac_1, c_1] where ac_1 is the next LayerBroadcaster, and c_1 is the
    broadcast RowPartition
  """
  c_1 = ac_0.broadcast_row_partition(a_1)
  old_value_rowids = array_ops.gather(ac_0.gather_index, c_1.value_rowids())
  old_row_starts = array_ops.gather(a_1.row_splits(), old_value_rowids)
  gather_index = old_row_starts + c_1.offsets_in_rows()
  return [_LayerBroadcaster.from_gather_index(gather_index), c_1]


def _broadcast_dynamic_shape_next_layer_half_ragged(
    ac_0: _LayerBroadcaster, bc_0: _LayerBroadcaster, a_1: RowPartition,
    b_1: RowPartition
) -> Tuple[RowPartition, _LayerBroadcaster, _LayerBroadcaster]:
  r"""Broadcast target and next layer broadcaster of two dynamic shapes.

  a_1 is uniform, and b_1 is ragged.
     *--ac_0-->*<--bc_0--*
     |         |         |
    a_1       c_1       b_1
     |         |         |
     V         V         V
     *--ac_1-->*<--bc_1--*

  Args:
    ac_0: _LayerBroadcaster from a to c in the previous layer.
    bc_0: _LayerBroadcaster from b to c in the previous layer.
    a_1: a uniform RowPartition for the next layer of a.
    b_1: a ragged RowPartition for the next layer of b.

  Returns:
    (c_1, ac_1, bc_1)
    c_1: a RowPartition for the next layer of the dynamic shape.
    ac_1: _LayerBroadcaster from a to c in the next layer.
    bc_1: _LayerBroadcaster from b to c in the next layer.
  """
  if not isinstance(ac_0, _LayerBroadcaster):
    raise TypeError("ac_0 should be a _LayerBroadcaster")
  if not isinstance(bc_0, _LayerBroadcaster):
    raise TypeError("bc_0 should be a _LayerBroadcaster")
  if not isinstance(a_1, RowPartition):
    raise TypeError("a_1 should be a RowPartition")
  if not isinstance(b_1, RowPartition):
    raise TypeError("b_1 should be a RowPartition")

  assert a_1.is_uniform()
  assert not b_1.is_uniform()

  static_a_1 = tensor_util.constant_value(a_1.uniform_row_length())
  if static_a_1 == 1:
    [bc_1, c_1b] = _broadcast_half(bc_0, b_1)
    ac_1_gather_index = array_ops.gather(ac_0.gather_index, c_1b.value_rowids())
    c_1 = RowPartition.from_row_splits(c_1b.row_splits())
    ac_1 = _LayerBroadcaster.from_gather_index(ac_1_gather_index)
    bc_1 = _LayerBroadcaster.from_gather_index(bc_1.gather_index)
    return [c_1, ac_1, bc_1]

  def broadcast_noop():
    # The sides must be "equal".
    [ac_1, c_1a] = _broadcast_half(ac_0, a_1)
    [bc_1, c_1b] = _broadcast_half(bc_0, b_1)
    checks = [check_ops.assert_equal(c_1a.row_splits(), c_1b.row_splits())]
    return [
        control_flow_ops.with_dependencies(checks, x)
        for x in [a_1.row_splits(), ac_1.gather_index, bc_1.gather_index]
    ]

  def broadcast_a():
    [bc_1, c_1b] = _broadcast_half(bc_0, b_1)
    ac_1_gather_index = array_ops.gather(ac_0.gather_index, c_1b.value_rowids())
    return [
        c_1b.row_splits(),
        ac_1_gather_index,
        bc_1.gather_index,
    ]

  can_broadcast_a = math_ops.equal(a_1.uniform_row_length(), 1)

  [c_1_row_splits, ac_1_gather_index,
   bc_1_gather_index] = control_flow_ops.cond(
       can_broadcast_a, true_fn=broadcast_a, false_fn=broadcast_noop)

  c_1 = RowPartition.from_row_splits(c_1_row_splits)
  ac_1 = _LayerBroadcaster.from_gather_index(ac_1_gather_index)
  bc_1 = _LayerBroadcaster.from_gather_index(bc_1_gather_index)
  return [c_1, ac_1, bc_1]


def _broadcast_dynamic_shape_next_layer_both_uniform(
    ac_0: _LayerBroadcaster, bc_0: _LayerBroadcaster, a_1: RowPartition,
    b_1: RowPartition
) -> Tuple[RowPartition, _LayerBroadcaster, _LayerBroadcaster]:
  r"""Broadcast target and next layer broadcaster of two uniform dynamic shapes.

     *--ac_0-->*<--bc_0--*
     |         |         |
    a_1       c_1       b_1
     |         |         |
     V         V         V
     *--ac_1-->*<--bc_1--*

  Args:
    ac_0: _LayerBroadcaster from a to c in the previous layer.
    bc_0: _LayerBroadcaster from b to c in the previous layer.
    a_1: a RowPartition for the next layer of a.
    b_1: a RowPartition for the next layer of b.

  Returns:
    (c_1, ac_1, bc_1)
    c_1: a RowPartition for the next layer of the dynamic shape.
    ac_1: _LayerBroadcaster from a to c in the next layer.
    bc_1: _LayerBroadcaster from b to c in the next layer.
  """
  if not isinstance(ac_0, _LayerBroadcaster):
    raise TypeError("ac_0 should be a _LayerBroadcaster")
  if not isinstance(bc_0, _LayerBroadcaster):
    raise TypeError("bc_0 should be a _LayerBroadcaster")
  if not isinstance(a_1, RowPartition):
    raise TypeError("a_1 should be a RowPartition")
  if not isinstance(b_1, RowPartition):
    raise TypeError("b_1 should be a RowPartition")
  assert a_1.is_uniform()
  assert b_1.is_uniform()

  static_a_1 = tensor_util.constant_value(a_1.uniform_row_length())
  static_b_1 = tensor_util.constant_value(b_1.uniform_row_length())

  if static_a_1 is not None:
    if static_a_1 == static_b_1:
      # Here, this dimension is the same, but we may have to broadcast previous
      # dimensions.
      [ac_1, _] = _broadcast_half(ac_0, a_1)
      [bc_1, _] = _broadcast_half(bc_0, b_1)
      c_1 = RowPartition.from_uniform_row_length(
          static_a_1,
          nrows=ac_0.dest_nrows())
      return [c_1, ac_1, bc_1]
    elif static_a_1 == 1:
      [bc_1, c_1b] = _broadcast_half(bc_0, b_1)
      ac_1 = _LayerBroadcaster.from_gather_index(
          array_ops.gather(ac_0.gather_index, c_1b.value_rowids()))
      c_1 = RowPartition.from_uniform_row_length(
          b_1.uniform_row_length(),
          nrows=bc_0.dest_nrows())
      return [c_1, ac_1, bc_1]

  if static_b_1 == 1:
    [ac_1, c_1a] = _broadcast_half(ac_0, a_1)
    bc_1 = _LayerBroadcaster.from_gather_index(
        array_ops.gather(bc_0.gather_index, c_1a.value_rowids()))
    c_1 = RowPartition.from_uniform_row_length(
        a_1.uniform_row_length(),
        nrows=ac_0.dest_nrows())
    return [c_1, ac_1, bc_1]

  def broadcast_noop():
    # Assumes a_1.uniform_row_length() == b_1.uniform_row_length()
    # Both sides broadcast to a single shape.
    [ac_1, _] = _broadcast_half(ac_0, a_1)
    [bc_1, _] = _broadcast_half(bc_0, b_1)
    return [a_1.uniform_row_length(), ac_1.gather_index, bc_1.gather_index]

  def broadcast_a():
    [bc_1, c_1b] = _broadcast_half(bc_0, b_1)
    ac_1_gather_index = array_ops.gather(ac_0.gather_index, c_1b.value_rowids())
    return [
        b_1.uniform_row_length(),
        ac_1_gather_index,
        bc_1.gather_index,
    ]

  def broadcast_b():
    [ac_1, c_1a] = _broadcast_half(ac_0, a_1)
    bc_1_gather_index = array_ops.gather(bc_0.gather_index, c_1a.value_rowids())
    return [a_1.uniform_row_length(), ac_1.gather_index, bc_1_gather_index]

  can_broadcast_b = math_ops.equal(b_1.uniform_row_length(), 1)

  def no_broadcast_a():
    return control_flow_ops.cond(
        can_broadcast_b, true_fn=broadcast_b, false_fn=broadcast_noop)

  can_broadcast_a = math_ops.equal(a_1.uniform_row_length(), 1)

  broadcast_asserts = [
      check_ops.assert_equal(
          math_ops.logical_or(
              math_ops.logical_or(can_broadcast_a, can_broadcast_b),
              math_ops.equal(a_1.uniform_row_length(),
                             b_1.uniform_row_length())), True)
  ]

  result = control_flow_ops.cond(
      can_broadcast_a, true_fn=broadcast_a, false_fn=no_broadcast_a)

  [c_1_uniform_row_length, ac_1_gather_index, bc_1_gather_index] = [
      control_flow_ops.with_dependencies(broadcast_asserts, x) for x in result
  ]

  c_1 = RowPartition.from_uniform_row_length(
      c_1_uniform_row_length,
      nvals=c_1_uniform_row_length * ac_0.dest_nrows(),
      nrows=ac_0.dest_nrows())
  ac_1 = _LayerBroadcaster.from_gather_index(ac_1_gather_index)
  bc_1 = _LayerBroadcaster.from_gather_index(bc_1_gather_index)
  return [c_1, ac_1, bc_1]


def _broadcast_dynamic_shape_next_layer(
    ac_0: _LayerBroadcaster, bc_0: _LayerBroadcaster, a_1: RowPartition,
    b_1: RowPartition
) -> Tuple[RowPartition, _LayerBroadcaster, _LayerBroadcaster]:
  r"""Broadcast target and next layer broadcaster of two dynamic shapes.

     *--ac_0-->*<--bc_0--*
     |         |         |
    a_1       c_1       b_1
     |         |         |
     V         V         V
     *--ac_1-->*<--bc_1--*

  Args:
    ac_0: _LayerBroadcaster from a to c in the previous layer.
    bc_0: _LayerBroadcaster from b to c in the previous layer.
    a_1: a RowPartition for the next layer of a.
    b_1: a RowPartition for the next layer of b.

  Returns:
    (c_1, ac_1, bc_1)
    c_1: a RowPartition for the next layer of the dynamic shape.
    ac_1: _LayerBroadcaster from a to c in the next layer.
    bc_1: _LayerBroadcaster from b to c in the next layer.
  """
  if not isinstance(ac_0, _LayerBroadcaster):
    raise TypeError("ac_0 should be a _LayerBroadcaster")
  if not isinstance(bc_0, _LayerBroadcaster):
    raise TypeError("bc_0 should be a _LayerBroadcaster")
  if not isinstance(a_1, RowPartition):
    raise TypeError("a_1 should be a RowPartition")
  if not isinstance(b_1, RowPartition):
    raise TypeError("b_1 should be a RowPartition")

  if a_1.is_uniform():
    if b_1.is_uniform():
      return _broadcast_dynamic_shape_next_layer_both_uniform(
          ac_0, bc_0, a_1, b_1)
    else:
      return _broadcast_dynamic_shape_next_layer_half_ragged(
          ac_0, bc_0, a_1, b_1)
  else:
    if b_1.is_uniform():
      [c_1, bc_1, ac_1] = _broadcast_dynamic_shape_next_layer_half_ragged(  # pylint: disable=arguments-out-of-order
          bc_0, ac_0, b_1, a_1)
      return (c_1, ac_1, bc_1)
    else:
      # If neither shape is uniform, we cannot broadcast the dimension.
      [ac_1, c_1a] = _broadcast_half(ac_0, a_1)
      [bc_1, c_1b] = _broadcast_half(bc_0, b_1)
      check_valid = [
          check_ops.assert_equal(c_1a.row_splits(), c_1b.row_splits())
      ]
      return (c_1a._with_dependencies(check_valid),  # pylint: disable=protected-access
              ac_1.with_dependencies(check_valid),
              bc_1.with_dependencies(check_valid))


def _broadcast_dynamic_shape_from_rps(
    a_zero: _LayerBroadcaster, b_zero: _LayerBroadcaster,
    a_rps: Sequence[RowPartition], b_rps: Sequence[RowPartition]
) -> Tuple[Sequence[RowPartition], Sequence[_LayerBroadcaster],
           Sequence[_LayerBroadcaster]]:
  """Create BroadcastLayers from two shapes to a target shape.


      *--a_zero->*<-b_zero-*
      |          |         |
   a_rps[0]    c_rps[0]  b_rps[0]
      |          |         |
      V          V         V
      *--ac[1]-->*<-bc[1]--*
      |          |         |
   a_rps[1]   c_rps[0]   b_rps[1]
      |          |         |
      V          V         V
      *--ac[2]-->*<-bc[2]--*

  Note: ac[0]=a_zero, and bc[0]=b_zero.
  Args:
    a_zero: broadcaster from rows of a_rps[0] to target shape.
    b_zero: broadcaster from rows of b_rps[0] to target shape.
    a_rps: RowPartitions of first shape.
    b_rps: RowPartitions of second shape, equal in length to a_rps.

  Returns:
    (c_rps, ac, bc) where:
    c_rps: RowPartitions of target shape.
    ac: layers broadcasting from the first shape.
    bc: layers broadcasting from the second shape.
  """
  assert len(a_rps) == len(b_rps)
  if a_rps:
    (c_1, ac_1,
     bc_1) = _broadcast_dynamic_shape_next_layer(a_zero, b_zero, a_rps[0],
                                                 b_rps[0])
    (c_suffix, a_layers,
     b_layers) = _broadcast_dynamic_shape_from_rps(ac_1, bc_1, a_rps[1:],
                                                   b_rps[1:])

    return ([c_1] + c_suffix, [ac_1] + a_layers, [bc_1] + b_layers)
  else:
    return ([], [], [])


def _get_broadcast_num_row_partitions(a: DynamicRaggedShape,
                                      b: DynamicRaggedShape):
  """Returns broadcast_dynamic_shape(a, b).num_row_partitions."""
  # Assumes rank and num_row_partitions are not None.
  if (a.num_row_partitions == 0 and b.num_row_partitions == 0):
    return 0
  expanded_num_row_partitions_a = a.num_row_partitions + max(0, b.rank - a.rank)
  expanded_num_row_partitions_b = b.num_row_partitions + max(0, a.rank - b.rank)

  if a.num_row_partitions == 0:
    return expanded_num_row_partitions_b

  if b.num_row_partitions == 0:
    return expanded_num_row_partitions_a

  return max(expanded_num_row_partitions_a, expanded_num_row_partitions_b)


# pylint: disable=protected-access
def _broadcast_dynamic_shape_extended_complete(
    a: DynamicRaggedShape, b: DynamicRaggedShape, b_rps: Sequence[RowPartition],
    c_suffix: Sequence[RowPartition], ac: Sequence[_LayerBroadcaster],
    bc_suffix: Sequence[_LayerBroadcaster]
) -> Tuple[DynamicRaggedShape, _Broadcaster, _Broadcaster]:
  """Helper for broadcast_dynamic_shape_extended."""
  c_prefix = b_rps[:-len(c_suffix)]
  bc_prefix_length = b.rank - len(bc_suffix)
  bc_prefix = [
      _LayerBroadcaster.get_identity_broadcaster(b._num_slices_in_dimension(i))
      for i in range(bc_prefix_length)
  ]
  c_num_row_partitions = _get_broadcast_num_row_partitions(a, b)

  c_raw = DynamicRaggedShape.from_row_partitions(c_prefix + tuple(c_suffix))
  c = c_raw._with_num_row_partitions(c_num_row_partitions)
  return (c, _Broadcaster(a, c, ac), _Broadcaster(b, c, bc_prefix + bc_suffix))


def _broadcast_dynamic_shape_extended_helper(
    a: DynamicRaggedShape, b: DynamicRaggedShape
) -> Tuple[DynamicRaggedShape, _Broadcaster, _Broadcaster]:
  """Helper for broadcast_dynamic_shape_extended.

  Here, we force:
    a.rank <= b.rank
    2 <= b.rank
    1 <= a.rank
  Args:
    a: a DynamicRaggedShape
    b: a DynamicRaggedShape

  Returns:
    A triple of a shape and two broadcasters.
  """
  assert a.rank <= b.rank
  assert 2 <= b.rank
  assert 1 <= a.rank
  a_rps = a._as_row_partitions()  # pylint: disable=protected-access
  b_rps = b._as_row_partitions()  # pylint: disable=protected-access

  if len(a_rps) < len(b_rps):
    # Note: this includes the case where len(a_rps)==0.
    # Here we begin at -1, one dimension before a_rps[0].
    # neg_one_a_rp  | b_rps[-(len(a_rps)+1)]
    # a_rps[0]      | b_rps[-len(a_rps)]
    # a_rps[1]      | b_rps[1-len(a_rps)]
    # ...           | ...
    # a_rps[-1]     | b_rps[-1]

    a_nrows = a[0]
    a_nrows_static = tensor_util.constant_value(a_nrows)
    if a_nrows_static is not None:
      a_nrows = a_nrows_static

    neg_one_a_rp = RowPartition.from_uniform_row_length(
        uniform_row_length=a_nrows, nrows=1, nvals=a_nrows)
    neg_one_b_rp = b_rps[-(len(a_rps) + 1)]
    (neg_one_ac, neg_one_bc) = _broadcast_dynamic_shape_first_layer(
        constant_op.constant(1, dtype=b_rps[0].dtype), neg_one_b_rp.nrows())

    # The first part of the solution.
    (c_zero, ac_zero,
     bc_zero) = _broadcast_dynamic_shape_next_layer(neg_one_ac, neg_one_bc,
                                                    neg_one_a_rp, neg_one_b_rp)
    b_rps_tail = b_rps[-len(a_rps):] if len(a_rps) >= 1 else []

    (c_suffix, ac_layers,
     bc_layers) = _broadcast_dynamic_shape_from_rps(ac_zero, bc_zero, a_rps,
                                                    b_rps_tail)

    return _broadcast_dynamic_shape_extended_complete(
        a=a,
        b=b,
        b_rps=b_rps,
        c_suffix=[c_zero] + c_suffix,
        ac=[ac_zero] + ac_layers,
        bc_suffix=[neg_one_bc, bc_zero] + bc_layers)

  else:
    assert len(a_rps) == len(b_rps)
    (ac_zero,
     bc_zero) = _broadcast_dynamic_shape_first_layer(a_rps[0].nrows(),
                                                     b_rps[0].nrows())

    (c_rps, a_layers,
     b_layers) = _broadcast_dynamic_shape_from_rps(ac_zero, bc_zero, a_rps,
                                                   b_rps)
    return _broadcast_dynamic_shape_extended_complete(
        a=a,
        b=b,
        b_rps=b_rps,
        c_suffix=c_rps,
        ac=[ac_zero] + a_layers,
        bc_suffix=[bc_zero] + b_layers)


def _fix_slice_index(index, rank, num_row_partitions):
  """Slice indexes are always silently truncated."""
  if index < 0:
    if rank is None:
      raise ValueError(
          "Rank must be known to use __getitem__ on a negative index.")
    index = rank + index
  if index < 0:
    index = 0
  if (num_row_partitions > 0 and index <= num_row_partitions + 1):
    # The rank is always >= num_row_partitions + 1 if num_row_partitions > 0.
    return index
  if rank is None:
    raise ValueError("Rank must be known to use __getitem__ on a large index.")
  if index >= rank:
    index = rank
  return index


def _first_layer_gather_index(nrows_source, nrows_target):
  """Return the first layer gather_index.

  Args:
    nrows_source: the number of rows in the source.
    nrows_target: the number of rows in the target.

  Returns:
    A tensor, usable as a gather_index for a _LayerBroadcaster.
  """

  def gi_broadcast_first():
    return array_ops.zeros(nrows_target, dtype=nrows_target.dtype)

  def gi_no_broadcast_first():
    gather_index = math_ops.range(nrows_target, dtype=nrows_target.dtype)
    return gather_index

  do_broadcast = math_ops.equal(nrows_source,
                                constant_op.constant(1, nrows_source.dtype))
  nrows_equal = math_ops.equal(nrows_source, nrows_target)
  can_broadcast = check_ops.assert_equal(
      math_ops.logical_or(do_broadcast, nrows_equal),
      True,
      message="Cannot broadcast")

  gather_index = control_flow_ops.cond(
      do_broadcast, true_fn=gi_broadcast_first, false_fn=gi_no_broadcast_first)

  return control_flow_ops.with_dependencies([can_broadcast], gather_index)


def _next_layer_gather_index(bc, original_rp, broadcast_rp):
  r"""Create the next layer gather_index whether or not a broadcast happens.

     *----------bc-------->*
     |                     |
  original_rp           broadcast_rp
     |                     |
    \|/                   \|/
     *--next_broadcaster-->*

  Args:
    bc: the old broadcaster.
    original_rp: the original row partition.
    broadcast_rp: the target row partition.

  Returns:
    the gather_index for next_broadcaster.
  Raises:
    InvalidArgumentError if the shapes are incompatible.
  """
  old_value_rowids = array_ops.gather(bc.gather_index,
                                      broadcast_rp.value_rowids())

  def gi_no_broadcast():
    # TODO(martinz): decide if row_splits or row_starts should be used here.
    old_row_starts = array_ops.gather(original_rp.row_splits(),
                                      old_value_rowids)
    expected_row_lengths = array_ops.gather(
        params=original_rp.row_lengths(), indices=bc.gather_index)
    actual_row_lengths = broadcast_rp.row_lengths()
    check_valid = check_ops.assert_equal(
        expected_row_lengths, actual_row_lengths, message="Cannot broadcast")
    gather_index = old_row_starts + broadcast_rp.offsets_in_rows()
    return control_flow_ops.with_dependencies([check_valid], gather_index)

  def gi_broadcast():
    # Several optimizations can occur here.
    # old_row_starts == old_value_rowids, because:
    #   if you are broadcasting, then the source has uniform row length of 1,
    #   implying original_rp.row_splits == tf.range(orgininal_rp.nvals + 1)
    # When broadcasting, there is no need to add offsets to the
    # source, because the source has size 1.
    # Also, this is always valid, because we enforce source and destination
    # have uniform_row_length.
    return old_value_rowids

  if not original_rp.is_uniform():
    return gi_no_broadcast()

  do_broadcast = math_ops.equal(original_rp.uniform_row_length(),
                                constant_op.constant(1, original_rp.dtype))
  gather_index = control_flow_ops.cond(
      do_broadcast, true_fn=gi_broadcast, false_fn=gi_no_broadcast)

  return gather_index


def _flat_values_shape(rt):
  if isinstance(rt, ragged_tensor.RaggedTensor):
    return array_ops.shape(rt.flat_values)
  return rt.flat_values.shape


def _to_row_partitions_and_nvals_from_lengths(
    lengths: Sequence[Union[int, Sequence[int]]],
    dtype=None) -> Tuple[Sequence[RowPartition], int]:
  """Allow ragged and uniform shapes to be specified.

  For example, [2, [2,1], 2] represents a shape like:
  [[[0, 0], [0, 0]], [[0, 0]]]

  Args:
    lengths: a list of integers and lists of integers.
    dtype: dtype of the shape (tf.int32 or tf.int64)

  Returns:
    a sequence of RowPartitions, and the number of values of the last partition.
  """
  size_so_far = lengths[0]
  result = []
  for current_lengths in lengths[1:]:
    if isinstance(current_lengths, int):
      nrows = size_so_far
      nvals = current_lengths * nrows
      size_so_far = nvals
      result.append(
          RowPartition.from_uniform_row_length(
              current_lengths, nvals, nrows=nrows, dtype_hint=dtype))
    else:
      if size_so_far != len(current_lengths):
        raise ValueError("Shape not consistent.")
      result.append(
          RowPartition.from_row_lengths(current_lengths, dtype_hint=dtype))
      size_so_far = sum(current_lengths)
  return (result, size_so_far)


def _element_to_string(x):
  """element to a string within a list."""
  if x is Ellipsis:
    return "..."
  if isinstance(x, str):
    return "'" + x + "'"
  return str(x)


def _list_tail_with_ellipsis(arr):
  """Print the tail of a list where the list might have an ellipsis."""
  if not arr:
    return "]"
  else:
    return ", " + _element_to_string(arr[0]) + _list_tail_with_ellipsis(arr[1:])


def _list_with_ellipsis_to_str(arr):
  """Print a list that might have ellipsis."""
  if not arr:
    return "[]"
  return "[" + _element_to_string(arr[0]) + _list_tail_with_ellipsis(arr[1:])


def _is_int_or_tuple_of_ints(x):
  if isinstance(x, int):
    return True
  if not isinstance(x, tuple):
    return False
  for y in x:
    if not isinstance(y, int):
      return False
  return True


def _alt_inner_shape_from_tensor_shape(shape, dtype, new_inner_rank):
  """Helper for _alt_inner_shape, used directly in _with_num_row_partitions."""
  if new_inner_rank == 1:
    return constant_op.constant([shape.num_elements()], dtype=dtype)
  new_inner_rank_tail_length = new_inner_rank - 1
  inner_shape_tail = shape[-new_inner_rank_tail_length:].as_list()
  first_dim = shape[:-new_inner_rank_tail_length].num_elements()
  return constant_op.constant([first_dim] + inner_shape_tail, dtype=dtype)


def _safe_floor_div(dividend: tensor_shape.Dimension,
                    divisor: tensor_shape.Dimension) -> tensor_shape.Dimension:
  if tensor_shape.dimension_value(divisor) == 0:
    return None
  return dividend // divisor


# TODO(b/218932570)
def _reduce_prod_patch(x):
  if x.dtype == dtypes.int64:
    return math_ops.cast(
        math_ops.reduce_prod(math_ops.cast(x, dtypes.int32)), dtypes.int64)
  return math_ops.reduce_prod(x)


# Type alias for shape encoded as a DynamicRaggedShape or a Tensor.
DenseOrRaggedShape = Union[DynamicRaggedShape, core.TensorLike]
