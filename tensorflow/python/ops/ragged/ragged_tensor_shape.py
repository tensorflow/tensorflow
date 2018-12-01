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
"""Shapes & broadcasting for RaggedTensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util


class RaggedTensorDynamicShape(object):
  """A collection of tensors encoding the shape of a potentially ragged tensor.

  Each `RaggedTensorDynamicShape` consists of an ordered list of dimension
  sizes.  There are two dimension types:

    * "Uniform dimensions" are dimenisons where all slices have the same
      length.  `RaggedTensorDynamicShape` records the size of each uniform
      dimension using a single scalar integer.

    * "Ragged dimensions" are dimensions whose slices may have different
      lengths.  `RaggedTensorDynamicShape` records the size of each ragged
      dimension using an integer vector containing the slice lengths for all
      the slices across that dimension.

  Furthermore, there are two ways a dimension might be encoded:

    * "Partitioned dimensions" are dimensions that are encoded using a
      `RaggedTensor`'s `nested_row_splits`.  The outermostmost partitioned
      dimension must be uniform, and the innermost partitioned dimension must
      be ragged.

    * "Inner dimensions" are dimensions that are encoded using a
      `RaggedTensor`'s `inner_values`.  Inner dimensions are always uniform.

  The sizes of partitioned dimensions are recorded using `partitioned_dim_sizes`
  and `inner_dim_sizes`:

    * `paritioned_dim_sizes` is a list of tensors (one for each partitioned
      dimension).

      * For uniform dimensions, the tensor is an integer scalar specifying the
        size of all slices across that dimension.
      * For ragged dimensions, the tensor is an integer vector specifying the
        size of each slice across that dimension.

    * `inner_dim_sizes` is a single integer vector, where each element
      specifies the size of a single inner dimension.

  Examples:

  Tensor                         | Ragged | Partitioned Dim Sizes  | Inner Dim
                                 : Rank   :                        : Sizes
  ------------------------------ | ------ | ---------------------- | ----------
  `[[1, 2, 3], [4, 5, 6]]`       |      0 |                        | `2, 3`
  `[[1, 2], [], [3, 4, 5]]`      |      1 | `3, (2, 0, 3)`         |
  `[[[1, 2], [3, 4]], [[5, 6]]]` |      1 | `2, (2, 1)`            | 2
  `[[[1, 2], [3]], [[4, 5]]]`    |      2 | `2, (2, 1), (2, 1, 2)` |
  """

  def __init__(self, partitioned_dim_sizes, inner_dim_sizes):
    """Creates a RaggedTensorDynamicShape.

    Args:
      partitioned_dim_sizes: A `list` of 0-D or 1-D integer `Tensor`, one for
        each partitioned dimension.  If dimension `d` is uniform, then
        `partitioned_dim_sizes[d]` must be an integer scalar, specifying the
        size of all slices across dimension `d`.  If dimension `d` is ragged,
        then `partitioned_dim_sizes[d]` must be an integer vector, specifying
        the size of each slice across dimension `d`.
      inner_dim_sizes: A 1-D integer `Tensor`, whose length is equal to the
        number of inner dimensions.  `inner_dim_sizes[n]` is the size of all
        slices across the `n`th inner dimension (which is the
        `(len(partitioned_dim_sizes)+n)`th dimension in the overall tensor.
    """
    assert isinstance(partitioned_dim_sizes, (list, tuple))
    with ops.name_scope(None, 'RaggedTensorDynamicShape',
                        (partitioned_dim_sizes, inner_dim_sizes)):
      partitioned_dim_sizes = tuple(
          ragged_util.convert_to_int_tensor(
              size, dtype=dtypes.int64, name='partitioned_dimension_size')
          for size in partitioned_dim_sizes)
      inner_dim_sizes = ragged_util.convert_to_int_tensor(
          inner_dim_sizes, dtype=dtypes.int64, name='inner_dim_sizes')

      # Validate shapes.
      if partitioned_dim_sizes:
        for axis, dimension_size in enumerate(partitioned_dim_sizes):
          if dimension_size.shape.ndims is None:
            raise ValueError(
                'rank of partitioned_dim_sizes[%d] is unknown' % axis)
          dimension_size.shape.with_rank_at_most(1)
        if partitioned_dim_sizes[0].shape.ndims == 1:
          raise ValueError('outermost partitioned dimension must be uniform')
        if partitioned_dim_sizes[-1].shape.ndims == 0:
          raise ValueError('innermost partitioned dimension must be ragged')
      inner_dim_sizes.shape.assert_has_rank(1)

      self._partitioned_dim_sizes = partitioned_dim_sizes
      self._inner_dim_sizes = inner_dim_sizes

  def __repr__(self):
    return ('RaggedTensorDynamicShape'
            '(partitioned_dim_sizes=%r, inner_dim_sizes=%r)' %
            (self._partitioned_dim_sizes, self._inner_dim_sizes))

  @staticmethod
  def from_dim_sizes(dim_sizes):
    """Constructs a ragged shape from a list of dimension sizes.

    This list contains a single tensor for each dimension, where the tensor
    is a scalar if the dimension is uniform, or a vector if the dimension is
    ragged.

    Args:
      dim_sizes: List of int64 scalars or vectors.

    Returns:
      A RaggedTensorDynamicShape.
    """
    with ops.name_scope(None, 'RaggedTensorDynamicShapeFromDimensionSizes',
                        [dim_sizes]):
      dim_sizes = tuple(
          ragged_util.convert_to_int_tensor(
              size, dtype=dtypes.int64, name='dim_sizes') for size in dim_sizes)
      # Split the dimensions into partitioned & inner dimensions.
      inner_split = 0
      for dim, dim_size in enumerate(dim_sizes):
        if dim_size.shape.ndims == 1:
          inner_split = dim + 1
        elif dim_size.shape.ndims != 0:
          raise ValueError('Each dim_size must be a scalar or a vector')
      return RaggedTensorDynamicShape(dim_sizes[:inner_split],
                                      dim_sizes[inner_split:])

  @classmethod
  def from_tensor(cls, rt_input):
    """Constructs a ragged shape for a potentially ragged tensor."""
    with ops.name_scope(None, 'RaggedTensorDynamicShapeFromTensor', [rt_input]):
      rt_input = ragged_factory_ops.convert_to_tensor_or_ragged_tensor(rt_input)
      if not ragged_tensor.is_ragged(rt_input):
        return cls([], array_ops.shape(rt_input))
      else:
        partitioned_dim_sizes = ((ragged_array_ops.nrows(rt_input),) +
                                 ragged_array_ops.nested_row_lengths(rt_input))
        return RaggedTensorDynamicShape(
            partitioned_dim_sizes,
            array_ops.shape(rt_input.inner_values)[1:])

  def dimension_size(self, axis):
    """Returns the size of slices across the specified dimension."""
    if not isinstance(axis, int):
      raise TypeError('axis must be an integer')
    partitioned_ndims = len(self._partitioned_dim_sizes)
    if axis < partitioned_ndims:
      return self._partitioned_dim_sizes[axis]
    else:
      return self._inner_dim_sizes[axis - partitioned_ndims]

  def is_ragged(self, axis):
    """Returns true if the indicated dimension is ragged."""
    if not isinstance(axis, int):
      raise TypeError('axis must be an integer')
    rank = self.rank
    if axis < 0:
      raise ValueError('Negative axis values are not supported')
    elif rank is not None and axis >= rank:
      raise ValueError('Expected axis=%s < rank=%s' % (axis, rank))
    else:
      return (axis > 0 and axis < len(self._partitioned_dim_sizes) and
              self._partitioned_dim_sizes[axis].shape.ndims == 1)

  @property
  def rank(self):
    """The number of dimensions in this shape, or None if unknown."""
    inner_ndims = self._inner_dim_sizes.shape[0].value
    if inner_ndims is None:
      return None
    else:
      return len(self._partitioned_dim_sizes) + inner_ndims

  @property
  def partitioned_dim_sizes(self):
    """The partitioned dimension sizes for this shape.

    Returns:
      A `list` of 0-D or 1-D integer `Tensor`.
    """
    return self._partitioned_dim_sizes

  @property
  def inner_dim_sizes(self):
    """The inner dimension sizes for this shape.

    Returns:
      A 1-D integer `Tensor`.
    """
    return self._inner_dim_sizes

  @property
  def num_partitioned_dimensions(self):
    """The number of partitioned dimensions in this shape."""
    return len(self._partitioned_dim_sizes)

  @property
  def num_inner_dimensions(self):
    """The number of inner dimensions, or `None` if not statically known."""
    return self._inner_dim_sizes.shape[0].value

  def broadcast_to_rank(self, rank):
    """Adds leading size-1 dimensions to broadcast `self` to the given rank.

    E.g., if `shape1` is `[3, (D2), 4]`, then `shape1.broadcast_to_rank(5)`
    is `[1, 1, 3, (D2), 4]`.

    Args:
      rank: The rank for the returned shape.

    Returns:
      A RaggedTensorDynamicShape with `rank` dimensions, whose inner dimensions
      have the same size as `self` and whose outer dimensions have size `1`.

    Raises:
      ValueError: If `self.rank` is unknown or greater than `rank`.
    """
    if self.rank is None:
      raise ValueError('Unable to broadcast: self.rank is unknown')
    dims_to_add = rank - self.rank
    if dims_to_add < 0:
      raise ValueError('Unable to broadcast: rank=%d must be greater than '
                       'self.rank=%d.' % (rank, self.rank))
    elif dims_to_add == 0:
      return self
    elif self._partitioned_dim_sizes:
      partitioned_dims = (1,) * dims_to_add + self._partitioned_dim_sizes
      return RaggedTensorDynamicShape(partitioned_dims, self._inner_dim_sizes)
    else:
      inner_dims = array_ops.concat(
          [array_ops.ones([dims_to_add], dtypes.int64), self.inner_dim_sizes],
          axis=0)
      return RaggedTensorDynamicShape([], inner_dims)

  def broadcast_dimension(self, axis, lengths):
    """Returns a shape that is broadcast-compatible with self & lengths.

    * If dimension[axis] is uniform and lengths is a scalar, the check
      that either lengths==1 or axis==1 or lengths==axis, and tile
      dimension[axis] with tf.where(lengths==axis, 1, axis) repeats.

    * If dimension[axis] is uniform and lengths is a vector, then check
      that dimension[axis]==1, and raggedly tile dimension[axis] with
      lengths repeats.  (we can skip tiling if we statically know that
      slice_lengths == 1??)

    * If dimension[axis] is ragged and lengths is a scalar, then check
      that lengths==1.

    * If dimension[axis] is ragged and lengths is a vector, then check
      that self.dimension_size(axis) == lengths.

    Args:
      axis: `int`.  The dimension to broadcast.
      lengths: 0-D or 1-D integer `Tensor`.

    Returns:
      A `RaggedTensorDynamicShape`.
    """
    lengths = ragged_util.convert_to_int_tensor(
        lengths, name='lengths', dtype=dtypes.int64)
    # Check whether lengths is a scalar (for uniform dimensions) or
    # vector (for ragged dimensions).
    if lengths.shape.ndims is None:
      raise ValueError('lengths must have a known rank.')
    elif lengths.shape.ndims > 1:
      raise ValueError('lengths must be a scalar or vector')
    else:
      lengths_is_scalar = (lengths.shape.ndims == 0)

    # Verify that the shapes are compatible.
    if self.is_ragged(axis):
      if lengths_is_scalar:
        condition = math_ops.equal(lengths, 1)
      else:
        condition = math_ops.reduce_all(
            math_ops.equal(lengths, self.dimension_size(axis)))
    else:
      axis_dim_size = self.dimension_size(axis)
      if lengths_is_scalar:
        condition = (
            math_ops.equal(lengths, 1) | math_ops.equal(axis_dim_size, 1)
            | math_ops.equal(axis_dim_size, lengths))
      else:
        condition = math_ops.equal(axis_dim_size, 1)
    broadcast_err = [
        'Unable to broadcast: dimension size mismatch in dimension', axis,
        'lengths=', lengths, 'dim_size=',
        self.dimension_size(axis)
    ]
    broadcast_check = control_flow_ops.Assert(
        condition, data=broadcast_err, summarize=10)

    with ops.control_dependencies([broadcast_check]):
      # Partitioned dimensions:
      if axis < self.num_partitioned_dimensions:
        if self.is_ragged(axis):
          # Use an identity op to make sure the check actually gets run.
          return RaggedTensorDynamicShape(
              self._partitioned_dim_sizes,
              array_ops.identity(self.inner_dim_sizes))
        else:
          return self._broadcast_uniform_partitioned_dimension(axis, lengths)

      # Inner dimensions:
      else:
        if lengths_is_scalar:
          return self._broadcast_inner_dimension_to_uniform(axis, lengths)
        else:
          if axis == 0:
            raise ValueError('Unable to broadcast: '
                             'outermost dimension must be uniform.')
          return self._broadcast_inner_dimension_to_ragged(axis, lengths)

  def num_slices_in_dimension(self, axis):
    """Returns the total number of slices across the indicated dimension."""
    if axis < 0:
      return constant_op.constant(1, dtype=dtypes.int64)
    elif self.is_ragged(axis):
      return math_ops.reduce_sum(self._partitioned_dim_sizes[axis])
    else:
      return self.dimension_size(axis) * self.num_slices_in_dimension(axis - 1)

  def _broadcast_uniform_partitioned_dimension(self, axis, lengths):
    """Broadcasts the partitioned dimension `axis` to match `lengths`."""
    axis_dim_size = self.dimension_size(axis)
    partitioned_sizes = list(self._partitioned_dim_sizes[:axis])

    if lengths.shape.ndims == 0:
      lengths = array_ops.where(
          math_ops.equal(axis_dim_size, 1), lengths, axis_dim_size)
      repeats = array_ops.where(math_ops.equal(axis_dim_size, 1), lengths, 1)
      splits = array_ops.stack([0, self.num_slices_in_dimension(axis)])
    else:
      splits = math_ops.range(
          array_ops.size(lengths, out_type=dtypes.int64) + 1)
      repeats = lengths

    partitioned_sizes.append(lengths)

    for dim_size in self._partitioned_dim_sizes[axis + 1:]:
      if dim_size.shape.ndims == 0:
        partitioned_sizes.append(dim_size)
        splits *= dim_size
      else:
        partitioned_sizes.append(
            ragged_util.repeat_ranges(dim_size, splits, repeats))
        splits = array_ops.gather(
            ragged_util.lengths_to_splits(dim_size), splits)
    inner_sizes = self._inner_dim_sizes
    return RaggedTensorDynamicShape(partitioned_sizes, inner_sizes)

  def _broadcast_inner_dimension_to_uniform(self, axis, length):
    """Broadcasts the inner dimension `axis` to match `lengths`."""
    dim_size = self.dimension_size(axis)
    axis_in_inner_dims = axis - self.num_partitioned_dimensions
    partitioned_sizes = self._partitioned_dim_sizes
    inner_sizes = array_ops.concat([
        self._inner_dim_sizes[:axis_in_inner_dims],
        [array_ops.where(math_ops.equal(dim_size, 1), length, dim_size)],
        self._inner_dim_sizes[axis_in_inner_dims + 1:]
    ],
                                   axis=0)
    return RaggedTensorDynamicShape(partitioned_sizes, inner_sizes)

  def _broadcast_inner_dimension_to_ragged(self, axis, lengths):
    axis_in_inner_dims = axis - self.num_partitioned_dimensions
    partitioned_sizes = (
        self._partitioned_dim_sizes + tuple([
            self._inner_dim_sizes[i] for i in range(axis_in_inner_dims)
        ]) + (lengths,))
    inner_sizes = self._inner_dim_sizes[axis_in_inner_dims + 1:]
    return RaggedTensorDynamicShape(partitioned_sizes, inner_sizes)


def broadcast_dynamic_shape(shape_x, shape_y):
  """Returns the shape formed by broadcasting two shapes to be compatible.

  Args:
    shape_x: A `RaggedTensorDynamicShape`
    shape_y: A `RaggedTensorDynamicShape`

  Returns:
    A `RaggedTensorDynamicShape`.
  Raises:
    ValueError: If `shape_x` and `shape_y` are not broadcast-compatible.
  """
  if not isinstance(shape_x, RaggedTensorDynamicShape):
    raise TypeError('shape_x must be a RaggedTensorDynamicShape')
  if not isinstance(shape_y, RaggedTensorDynamicShape):
    raise TypeError('shape_y must be a RaggedTensorDynamicShape')

  # Broadcast both shapes to have the same rank.
  if shape_x.rank is None or shape_y.rank is None:
    raise ValueError('Unable to broadcast: unknown rank')
  broadcast_rank = max(shape_x.rank, shape_y.rank)
  shape_x = shape_x.broadcast_to_rank(broadcast_rank)
  shape_y = shape_y.broadcast_to_rank(broadcast_rank)

  # Broadcast dimensions one at a time, starting from the outermost dimension.
  for axis in range(broadcast_rank):
    shape_x = shape_x.broadcast_dimension(axis, shape_y.dimension_size(axis))
    shape_y = shape_y.broadcast_dimension(axis, shape_x.dimension_size(axis))

  return shape_x


def broadcast_to(rt_input, shape, broadcast_inner_dimensions=True):
  """Broadcasts a potentially ragged tensor to a ragged shape.

  Tiles `rt_input` as necessary to match the given shape.

  Behavior is undefined if `rt_input` is not broadcast-compatible with `shape`.

  Args:
    rt_input: The potentially ragged tensor to broadcast.
    shape: A `RaggedTensorDynamicShape`
    broadcast_inner_dimensions: If false, then inner dimensions will not be
      tiled.

  Returns:
    A potentially ragged tensor whose values are taken from
    `rt_input`, and whose shape matches `shape`.
  """
  if not isinstance(shape, RaggedTensorDynamicShape):
    raise TypeError('shape must be a RaggedTensorDynamicShape')
  rt_input = ragged_factory_ops.convert_to_tensor_or_ragged_tensor(rt_input)

  # Broadcasting to a uniform shape.
  if shape.num_partitioned_dimensions == 0:
    return _broadcast_to_uniform_shape(rt_input, shape,
                                       broadcast_inner_dimensions)
  else:
    return _broadcast_to_ragged_shape(rt_input, shape,
                                      broadcast_inner_dimensions)


def _broadcast_to_uniform_shape(rt_input, shape, broadcast_inner_dimensions):
  """Broadcasts rt_input to the uniform shape `shape`."""
  if isinstance(rt_input, ragged_tensor.RaggedTensor):
    raise ValueError('Incompatible with shape: ragged rank mismatch')
  if broadcast_inner_dimensions:
    return array_ops.broadcast_to(rt_input, shape.inner_dim_sizes)
  else:
    return rt_input


def _broadcast_to_ragged_shape(rt_input, dst_shape, broadcast_inner_dimensions):
  """Broadcasts rt_input to the ragged shape `dst_shape`."""
  # dst_shape's rank and ragged_rank must be greater than or equal to rt_input's
  if rt_input.shape.ndims is None or dst_shape.rank is None:
    raise ValueError('Unable to broadcast: unknown rank')
  if rt_input.shape.ndims > dst_shape.rank:
    raise ValueError('Incompatible with shape: rank mismatch')
  if (isinstance(rt_input, ragged_tensor.RaggedTensor) and
      rt_input.ragged_rank >= dst_shape.num_partitioned_dimensions):
    raise ValueError('Incompatible with shape: ragged rank mismatch')

  src_shape = RaggedTensorDynamicShape.from_tensor(rt_input)
  src_shape = src_shape.broadcast_to_rank(dst_shape.rank)

  # Add dimensions to rt_input so its rank and ragged_rank matches dst_shape.
  if dst_shape.rank > rt_input.shape.ndims:
    if rt_input.shape.ndims < dst_shape.num_inner_dimensions + 1:
      rt_input = array_ops.reshape(
          rt_input, array_ops.concat([[-1], dst_shape.inner_dim_sizes], axis=0))
    for _ in range(dst_shape.rank - rt_input.shape.ndims):
      rt_input = ragged_factory_ops.from_row_lengths(
          rt_input, [ragged_array_ops.nrows(rt_input)])

  # Add ragged dimensions to match dst_shape.
  if ragged_tensor.is_ragged(rt_input):
    inner_rank_diff = (
        rt_input.inner_values.shape.ndims - 1 - dst_shape.num_inner_dimensions)
    if inner_rank_diff > 0:
      rt_input = rt_input.with_inner_values(
          ragged_conversion_ops.from_tensor(
              rt_input.inner_values, ragged_rank=inner_rank_diff))
  else:
    rt_input = ragged_conversion_ops.from_tensor(
        rt_input, ragged_rank=dst_shape.num_partitioned_dimensions - 1)

  # Do broadcasting for any dimensions that will remain uniform.  We can do
  # these all at once, since they're independent of one another.
  multiples = [1] * dst_shape.rank
  for axis in range(dst_shape.num_partitioned_dimensions):
    if not src_shape.is_ragged(axis) and not dst_shape.is_ragged(axis):
      src_size = src_shape.dimension_size(axis)
      dst_size = dst_shape.dimension_size(axis)
      if ((tensor_util.constant_value(src_size) in (1, None)) and
          (tensor_util.constant_value(dst_size) != 1)):
        multiples[axis] = array_ops.where(
            math_ops.equal(src_size, 1), dst_size, 1)
  if not all(isinstance(v, int) and v == 1 for v in multiples):
    multiples = array_ops.stack(multiples, axis=0)
    rt_input = ragged_array_ops.tile(rt_input, multiples)

  if broadcast_inner_dimensions:
    rt_input = rt_input.with_inner_values(
        array_ops.reshape(
            rt_input.inner_values,
            array_ops.concat([[-1], dst_shape.inner_dim_sizes], axis=0)))

  # Do broadcasting for dimensions that become ragged.  We must do these from
  # outermost to innermost.
  for axis in range(dst_shape.num_partitioned_dimensions):
    if not src_shape.is_ragged(axis) and dst_shape.is_ragged(axis):
      dst_size = dst_shape.dimension_size(axis)
      rt_input = _ragged_tile_axis(rt_input, axis, dst_size)

  return rt_input


def _ragged_tile_axis(rt_input, axis, repeats):
  """Tile a dimension of a RaggedTensor to match a ragged shape."""
  assert axis > 0  # Outermost dimension may not be ragged.

  if not ragged_tensor.is_ragged(rt_input):
    rt_input = ragged_conversion_ops.from_tensor(rt_input, ragged_rank=1)

  if axis > 1:
    return rt_input.with_values(
        _ragged_tile_axis(rt_input.values, axis - 1, repeats))
  else:
    src_row_splits = rt_input.nested_row_splits
    src_row_lengths = ragged_array_ops.nested_row_lengths(rt_input)
    splits = src_row_splits[0]

    dst_row_lengths = [repeats]
    for i in range(1, len(src_row_lengths)):
      dst_row_lengths.append(
          ragged_util.repeat_ranges(src_row_lengths[i], splits, repeats))
      splits = array_ops.gather(src_row_splits[i], splits)
    dst_values = ragged_util.repeat_ranges(rt_input.inner_values, splits,
                                           repeats)
    return ragged_factory_ops.from_nested_row_lengths(dst_values,
                                                      dst_row_lengths)

