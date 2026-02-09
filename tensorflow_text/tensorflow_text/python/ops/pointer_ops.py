# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Ops that consume or generate index-based pointers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_where_op
from tensorflow.python.ops.ragged import segment_id_ops


def gather_with_default(params, indices, default, name=None, axis=0):
  """Gather slices with `indices=-1` mapped to `default`.

  This operation is similar to `tf.gather()`, except that any value of `-1`
  in `indices` will be mapped to `default`.  Example:

  >>> gather_with_default(['a', 'b', 'c', 'd'], [2, 0, -1, 2, -1], '_')
  <tf.Tensor: shape=(5,), dtype=string,
      numpy=array([b'c', b'a', b'_', b'c', b'_'], dtype=object)>

  Args:
    params: The `Tensor` from which to gather values.  Must be at least rank
      `axis + 1`.
    indices: The index `Tensor`.  Must have dtype `int32` or `int64`, and values
      must be in the range `[-1, params.shape[axis])`.
    default: The value to use when `indices` is `-1`.  `default.shape` must
      be equal to `params.shape[axis + 1:]`.
    name: A name for the operation (optional).
    axis: The axis in `params` to gather `indices` from.  Must be a scalar
      `int32` or `int64`.  Supports negative indices.

  Returns:
    A `Tensor` with the same type as `param`, and with shape
    `params.shape[:axis] + indices.shape + params.shape[axis + 1:]`.
  """
  # This implementation basically just concatenates the default value and
  # the params together, and then uses gather(default_plus_params, indices + 1)
  # to get the appropriate values.  Most of the complexity below has to do
  # with properly handling cases where axis != 0, in which case we need to tile
  # the default before concatenating it.
  with ops.name_scope(name, 'GatherWithDefault',
                      [params, indices, default, axis]):
    # Convert inputs to tensors.
    indices = ops.convert_to_tensor(
        indices, name='indices', preferred_dtype=dtypes.int32)
    params = ops.convert_to_tensor(params, name='params')
    default = ops.convert_to_tensor(default, name='default', dtype=params.dtype)

    if axis == 0:
      tiled_default = array_ops_stack.stack([default])

    else:
      # Get ranks & shapes of inputs.
      params_rank = array_ops.rank(params)
      params_shape = array_ops.shape(params)
      default_shape = array_ops.shape(default)
      outer_params_shape = params_shape[:axis]

      # This will equal `axis` if axis>=0.
      outer_params_rank = array_ops.shape(outer_params_shape)[0]

      # Add dimensions (with size=1) to default, so its rank matches params.
      new_shape = array_ops.concat([
          array_ops.ones([outer_params_rank + 1], dtypes.int32), default_shape
      ],
                                   axis=0)
      reshaped_default = array_ops.reshape(default, new_shape)

      # Tile the default for any dimension dim<axis, so its size matches params.
      multiples = array_ops.concat([
          outer_params_shape,
          array_ops.ones(params_rank - outer_params_rank, dtypes.int32)
      ],
                                   axis=0)
      tiled_default = array_ops.tile(reshaped_default, multiples)

    # Prepend the default value to params (on the chosen axis).  Thus, the
    # default value is at index 0, and all other values have their index
    # incremented by one.
    default_plus_params = array_ops.concat([tiled_default, params], axis=axis)
    return array_ops.gather(default_plus_params, indices + 1, axis=axis)


def span_overlaps(source_start,
                  source_limit,
                  target_start,
                  target_limit,
                  contains=False,
                  contained_by=False,
                  partial_overlap=False,
                  name=None):
  """Returns a boolean tensor indicating which source and target spans overlap.

  The source and target spans are specified using B+1 dimensional tensors,
  with `B>=0` batch dimensions followed by a final dimension that lists the
  span offsets for each span in the batch:

  * The `i`th source span in batch `b1...bB` starts at
    `source_start[b1...bB, i]` (inclusive), and extends to just before
    `source_limit[b1...bB, i]` (exclusive).
  * The `j`th target span in batch `b1...bB` starts at
    `target_start[b1...bB, j]` (inclusive), and extends to just before
    `target_limit[b1...bB, j]` (exclusive).

  `result[b1...bB, i, j]` is true if the `i`th source span overlaps with the
  `j`th target span in batch `b1...bB`, where a source span overlaps a target
  span if any of the following are true:

    * The spans are identical.
    * `contains` is true, and the source span contains the target span.
    * `contained_by` is true, and the source span is contained by the target
      span.
    * `partial_overlap` is true, and there is a non-zero overlap between the
      source span and the target span.

  #### Example:
    Given the following source and target spans (with no batch dimensions):

    >>>  #         0    5    10   15   20   25   30   35   40
    >>>  #         |====|====|====|====|====|====|====|====|
    >>>  # Source: [-0-]     [-1-] [2] [-3-][-4-][-5-]
    >>>  # Target: [-0-][-1-]     [-2-] [3]   [-4-][-5-]
    >>>  #         |====|====|====|====|====|====|====|====|
    >>> source_start = [0, 10, 16, 20, 25, 30]
    >>> source_limit = [5, 15, 19, 25, 30, 35]
    >>> target_start = [0,  5, 15, 21, 27, 31]
    >>> target_limit = [5, 10, 20, 24, 32, 37]

    `result[i, j]` will be true at the following locations:

      * `[0, 0]` (always)
      * `[2, 2]` (if contained_by=True or partial_overlaps=True)
      * `[3, 3]` (if contains=True or partial_overlaps=True)
      * `[4, 4]` (if partial_overlaps=True)
      * `[5, 4]` (if partial_overlaps=True)
      * `[5, 5]` (if partial_overlaps=True)

  Args:
    source_start: A B+1 dimensional potentially ragged tensor with shape
      `[D1...DB, source_size]`: the start offset of each source span.
    source_limit: A B+1 dimensional potentially ragged tensor with shape
      `[D1...DB, source_size]`: the limit offset of each source span.
    target_start: A B+1 dimensional potentially ragged tensor with shape
      `[D1...DB, target_size]`: the start offset of each target span.
    target_limit: A B+1 dimensional potentially ragged tensor with shape
      `[D1...DB, target_size]`: the limit offset of each target span.
    contains: If true, then a source span is considered to overlap a target span
      when the source span contains the target span.
    contained_by: If true, then a source span is considered to overlap a target
      span when the source span is contained by the target span.
    partial_overlap: If true, then a source span is considered to overlap a
      target span when the source span partially overlaps the target span.
    name: A name for the operation (optional).

  Returns:
    A B+2 dimensional potentially ragged boolean tensor with shape
    `[D1...DB, source_size, target_size]`.

  Raises:
    ValueError: If the span tensors are incompatible.
  """
  _check_type(contains, 'contains', bool)
  _check_type(contained_by, 'contained_by', bool)
  _check_type(partial_overlap, 'partial_overlap', bool)

  scope_tensors = [source_start, source_limit, target_start, target_limit]
  with ops.name_scope(name, 'SpanOverlaps', scope_tensors):
    # Convert input tensors.
    source_start = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        source_start, name='source_start')
    source_limit = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        source_limit, name='source_limit')
    target_start = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        target_start, name='target_start')
    target_limit = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        target_limit, name='target_limit')
    span_tensors = [source_start, source_limit, target_start, target_limit]

    # Verify input tensor shapes and types.
    source_start.shape.assert_is_compatible_with(source_limit.shape)
    target_start.shape.assert_is_compatible_with(target_limit.shape)
    source_start.shape.assert_same_rank(target_start.shape)
    source_start.shape.assert_same_rank(target_limit.shape)
    source_limit.shape.assert_same_rank(target_start.shape)
    source_limit.shape.assert_same_rank(target_limit.shape)
    if not (source_start.dtype == target_start.dtype == source_limit.dtype ==
            target_limit.dtype):
      raise TypeError('source_start, source_limit, target_start, and '
                      'target_limit must all have the same dtype')
    ndims = set(
        [t.shape.ndims for t in span_tensors if t.shape.ndims is not None])
    assert len(ndims) <= 1  # because of assert_same_rank statements above.

    if all(not isinstance(t, ragged_tensor.RaggedTensor) for t in span_tensors):
      return _span_overlaps(source_start, source_limit, target_start,
                            target_limit, contains, contained_by,
                            partial_overlap)

    elif all(isinstance(t, ragged_tensor.RaggedTensor) for t in span_tensors):
      if not ndims:
        raise ValueError('For ragged inputs, the shape.ndims of at least one '
                         'span tensor must be statically known.')
      if list(ndims)[0] == 2:
        return _span_overlaps(source_start, source_limit, target_start,
                              target_limit, contains, contained_by,
                              partial_overlap)
      else:
        # Handle ragged batch dimension by recursion on values.
        row_splits = span_tensors[0].row_splits
        shape_checks = [
            check_ops.assert_equal(
                t.row_splits,
                row_splits,
                message='Mismatched ragged shapes for batch dimensions')
            for t in span_tensors[1:]
        ]
        with ops.control_dependencies(shape_checks):
          return ragged_tensor.RaggedTensor.from_row_splits(
              span_overlaps(source_start.values, source_limit.values,
                            target_start.values, target_limit.values, contains,
                            contained_by, partial_overlap), row_splits)

    else:
      # Mix of dense and ragged tensors.
      raise ValueError('Span tensors must all have the same ragged_rank')


def _span_overlaps(source_start, source_limit, target_start, target_limit,
                   contains, contained_by, partial_overlap):
  """Implementation of span_overlaps().

  If the inputs are ragged, then the source tensors must have exactly one
  batch dimension.  (I.e., `B=1` in the param descriptions below.)

  Args:
    source_start: `<int>[D1...DB, source_size]`
    source_limit: `<int>[D1...DB, source_size]`
    target_start: `<int>[D1...DB, target_size]`
    target_limit: `<int>[D1...DB, target_size]`
    contains: `bool`
    contained_by: `bool`
    partial_overlap: `bool`

  Returns:
    `<bool>[D1...DB, source_size, target_size]`
  """
  if isinstance(source_start, tensor.Tensor):
    # Reshape the source tensors to [D1...DB, source_size, 1] and the
    # target tensors to [D1...DB, 1, target_size], so we can use broadcasting.
    # In particular, elementwise_op(source_x, target_x) will have shape
    # [D1...DB, source_size, target_size].
    source_start = array_ops.expand_dims(source_start, -1)
    source_limit = array_ops.expand_dims(source_limit, -1)
    target_start = array_ops.expand_dims(target_start, -2)
    target_limit = array_ops.expand_dims(target_limit, -2)

    equal = math_ops.equal
    less_equal = math_ops.less_equal
    less = math_ops.less
    logical_and = math_ops.logical_and
    logical_or = math_ops.logical_or

  else:
    # Broadcast the source span indices to all have shape
    # [batch_size, (source_size), (target_size)].
    (source_start, source_limit) = _broadcast_ragged_sources_for_overlap(
        source_start, source_limit, target_start.row_splits)
    (target_start, target_limit) = _broadcast_ragged_targets_for_overlap(
        target_start, target_limit, source_start.row_splits)

    # Use map_flat_values to perform elementwise operations.
    equal = functools.partial(ragged_functional_ops.map_flat_values,
                              math_ops.equal)
    less_equal = functools.partial(ragged_functional_ops.map_flat_values,
                                   math_ops.less_equal)
    less = functools.partial(ragged_functional_ops.map_flat_values,
                             math_ops.less)
    logical_and = functools.partial(ragged_functional_ops.map_flat_values,
                                    math_ops.logical_and)
    logical_or = functools.partial(ragged_functional_ops.map_flat_values,
                                   math_ops.logical_or)

  if partial_overlap:
    return logical_or(
        logical_and(
            less_equal(source_start, target_start),
            less(target_start, source_limit)),
        logical_and(
            less_equal(target_start, source_start),
            less(source_start, target_limit)))
  elif contains and contained_by:
    return logical_or(
        logical_and(
            less_equal(source_start, target_start),
            less_equal(target_limit, source_limit)),
        logical_and(
            less_equal(target_start, source_start),
            less_equal(source_limit, target_limit)))
  elif contains:
    return logical_and(
        less_equal(source_start, target_start),
        less_equal(target_limit, source_limit))
  elif contained_by:
    return logical_and(
        less_equal(target_start, source_start),
        less_equal(source_limit, target_limit))
  else:
    return logical_and(
        equal(target_start, source_start), equal(source_limit, target_limit))


def _broadcast_ragged_targets_for_overlap(target_start, target_limit,
                                          source_splits):
  """Repeats target indices for each source item in the same batch.

  Args:
    target_start: `<int>[batch_size, (target_size)]`
    target_limit: `<int>[batch_size, (target_size)]`
    source_splits: `<int64>[batch_size, (source_size+1)]`

  Returns:
    `<int>[batch_size, (source_size), (target_size)]`.
    A tuple of ragged tensors `(tiled_target_start, tiled_target_limit)` where:

    * `tiled_target_start[b, s, t] = target_start[b, t]`
    * `tiled_target_limit[b, s, t] = target_limit[b, t]`
  """
  source_batch_ids = segment_id_ops.row_splits_to_segment_ids(source_splits)

  target_start = ragged_tensor.RaggedTensor.from_value_rowids(
      ragged_gather_ops.gather(target_start, source_batch_ids),
      source_batch_ids)
  target_limit = ragged_tensor.RaggedTensor.from_value_rowids(
      ragged_gather_ops.gather(target_limit, source_batch_ids),
      source_batch_ids)
  return (target_start, target_limit)


def _broadcast_ragged_sources_for_overlap(source_start, source_limit,
                                          target_splits):
  """Repeats source indices for each target item in the same batch.

  Args:
    source_start: `<int>[batch_size, (source_size)]`
    source_limit: `<int>[batch_size, (source_size)]`
    target_splits: `<int64>[batch_size, (target_size+1)]`

  Returns:
    `<int>[batch_size, (source_size), (target_size)]`.
    A tuple of tensors `(tiled_source_start, tiled_source_limit)` where:

    * `tiled_target_start[b, s, t] = source_start[b, s]`
    * `tiled_target_limit[b, s, t] = source_limit[b, s]`
  """
  source_splits = source_start.row_splits
  target_rowlens = target_splits[1:] - target_splits[:-1]
  source_batch_ids = segment_id_ops.row_splits_to_segment_ids(source_splits)

  # <int64>[sum(source_size[b] for b in range(batch_size))]
  # source_repeats[i] is the number of target spans in the batch that contains
  # source span i.  We need to add a new ragged dimension that repeats each
  # source span this number of times.
  source_repeats = ragged_gather_ops.gather(target_rowlens, source_batch_ids)

  # <int64>[sum(source_size[b] for b in range(batch_size)) + 1]
  # The row_splits tensor for the inner ragged dimension of the result tensors.
  inner_splits = array_ops.concat([[0], math_ops.cumsum(source_repeats)],
                                  axis=0)

  # <int64>[sum(source_size[b] * target_size[b] for b in range(batch_size))]
  # Indices for gathering source indices.
  source_indices = segment_id_ops.row_splits_to_segment_ids(inner_splits)

  source_start = ragged_tensor.RaggedTensor.from_nested_row_splits(
      array_ops.gather(source_start.values, source_indices),
      [source_splits, inner_splits])
  source_limit = ragged_tensor.RaggedTensor.from_nested_row_splits(
      array_ops.gather(source_limit.values, source_indices),
      [source_splits, inner_splits])

  return source_start, source_limit


def span_alignment(source_start,
                   source_limit,
                   target_start,
                   target_limit,
                   contains=False,
                   contained_by=False,
                   partial_overlap=False,
                   multivalent_result=False,
                   name=None):
  """Return an alignment from a set of source spans to a set of target spans.

  The source and target spans are specified using B+1 dimensional tensors,
  with `B>=0` batch dimensions followed by a final dimension that lists the
  span offsets for each span in the batch:

  * The `i`th source span in batch `b1...bB` starts at
    `source_start[b1...bB, i]` (inclusive), and extends to just before
    `source_limit[b1...bB, i]` (exclusive).
  * The `j`th target span in batch `b1...bB` starts at
    `target_start[b1...bB, j]` (inclusive), and extends to just before
    `target_limit[b1...bB, j]` (exclusive).

  `result[b1...bB, i]` contains the index (or indices) of the target span that
  overlaps with the `i`th source span in batch `b1...bB`.  The
  `multivalent_result` parameter indicates whether the result should contain
  a single span that aligns with the source span, or all spans that align with
  the source span.

  * If `multivalent_result` is false (the default), then `result[b1...bB, i]=j`
    indicates that the `j`th target span overlaps with the `i`th source span
    in batch `b1...bB`.  If no target spans overlap with the `i`th target span,
    then `result[b1...bB, i]=-1`.

  * If `multivalent_result` is true, then `result[b1...bB, i, n]=j` indicates
    that the `j`th target span is the `n`th span that overlaps with the `i`th
    source span in in batch `b1...bB`.

  For a definition of span overlap, see the docstring for `span_overlaps()`.

  #### Examples:

  Given the following source and target spans (with no batch dimensions):

  >>> #         0    5    10   15   20   25   30   35   40   45   50   55   60
  >>> #         |====|====|====|====|====|====|====|====|====|====|====|====|
  >>> # Source: [-0-]     [-1-] [2] [3]    [4][-5-][-6-][-7-][-8-][-9-]
  >>> # Target: [-0-][-1-]     [-2-][-3-][-4-] [5] [6]    [7]  [-8-][-9-][10]
  >>> #         |====|====|====|====|====|====|====|====|====|====|====|====|
  >>> source_starts = [0, 10, 16, 20, 27, 30, 35, 40, 45, 50]
  >>> source_limits = [5, 15, 19, 23, 30, 35, 40, 45, 50, 55]
  >>> target_starts = [0,  5, 15, 20, 25, 31, 35, 42, 47, 52, 57]
  >>> target_limits = [5, 10, 20, 25, 30, 34, 38, 45, 52, 57, 61]
  >>> span_alignment(source_starts, source_limits, target_starts, target_limits)
  <tf.Tensor: shape=(10,), dtype=int64,
      numpy=array([ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1])>
  >>> span_alignment(source_starts, source_limits, target_starts, target_limits,
  ...                multivalent_result=True)
  <tf.RaggedTensor [[0], [], [], [], [], [], [], [], [], []]>
  >>> span_alignment(source_starts, source_limits, target_starts, target_limits,
  ...                contains=True)
  <tf.Tensor: shape=(10,), dtype=int64,
      numpy=array([ 0, -1, -1, -1, -1,  5,  6,  7, -1, -1])>
  >>> span_alignment(source_starts, source_limits, target_starts, target_limits,
  ...                 partial_overlap=True, multivalent_result=True)
  <tf.RaggedTensor [[0], [], [2], [3], [4], [5], [6], [7], [8], [8, 9]]>

  Args:
    source_start: A B+1 dimensional potentially ragged tensor with shape
      `[D1...DB, source_size]`: the start offset of each source span.
    source_limit: A B+1 dimensional potentially ragged tensor with shape
      `[D1...DB, source_size]`: the limit offset of each source span.
    target_start: A B+1 dimensional potentially ragged tensor with shape
      `[D1...DB, target_size]`: the start offset of each target span.
    target_limit: A B+1 dimensional potentially ragged tensor with shape
      `[D1...DB, target_size]`: the limit offset of each target span.
    contains: If true, then a source span is considered to overlap a target span
      when the source span contains the target span.
    contained_by: If true, then a source span is considered to overlap a target
      span when the source span is contained by the target span.
    partial_overlap: If true, then a source span is considered to overlap a
      target span when the source span partially overlaps the target span.
    multivalent_result: Whether the result should contain a single target span
      index (if `multivalent_result=False`) or a list of target span indices (if
      `multivalent_result=True`) for each source span.
    name: A name for the operation (optional).

  Returns:
    An int64 tensor with values in the range: `-1 <= result < target_size`.
    If `multivalent_result=False`, then the returned tensor has shape
      `[source_size]`, where `source_size` is the length of the `source_start`
      and `source_limit` input tensors.  If `multivalent_result=True`, then the
      returned tensor has shape `[source_size, (num_aligned_target_spans)].
  """
  scope_tensors = [source_start, source_limit, target_start, target_limit]
  with ops.name_scope(name, 'SpanAlignment', scope_tensors):
    source_start = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        source_start, name='source_start')
    source_limit = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        source_limit, name='source_limit')
    target_start = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        target_start, name='target_start')
    target_limit = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        target_limit, name='target_limit')

    # <bool>[D1...DB, source_size, target_size]
    # overlaps[b1...bB, i, j] is true if source span i overlaps target span j
    # (in batch b1...bB).
    overlaps = span_overlaps(source_start, source_limit, target_start,
                             target_limit, contains, contained_by,
                             partial_overlap)

    # <int64>[D1...DB, source_size, (num_aligned_spans)]
    # alignment[b1...bB, i, n]=j if target span j is the n'th target span
    # that aligns with source span i (in batch b1...bB).
    alignment = _multivalent_span_alignment(overlaps)

    if not multivalent_result:
      # <int64>[D1...DB, source_size]
      # alignment[b1...bB, i]=j if target span j is the last target span
      # that aligns with source span i, or -1 if no target spans align.
      alignment = ragged_functional_ops.map_flat_values(
          math_ops.maximum, ragged_math_ops.reduce_max(alignment, axis=-1), -1)
    return alignment


def _multivalent_span_alignment(overlaps):
  """Returns the multivalent span alignment for a given overlaps tensor.

  Args:
    overlaps: `<int64>[D1...DB, source_size, target_size]`: `overlaps[b1...bB,
      i, j]` is true if source span `i` overlaps target span `j` (in batch
      `b1...bB`).

  Returns:
    `<int64>[D1...DB, source_size, (num_aligned_spans)]`:
    `result[b1...bB, i, n]=j` if target span `j` is the `n`'th target span
    that aligns with source span `i` (in batch `b1...bB`).
  """
  overlaps_ndims = overlaps.shape.ndims
  assert overlaps_ndims is not None  # guaranteed/checked by span_overlaps()
  assert overlaps_ndims >= 2

  # If there are multiple batch dimensions, then flatten them and recurse.
  if overlaps_ndims > 3:
    if not isinstance(overlaps, ragged_tensor.RaggedTensor):
      overlaps = ragged_tensor.RaggedTensor.from_tensor(
          overlaps, ragged_rank=overlaps.shape.ndims - 3)
    return overlaps.with_values(_multivalent_span_alignment(overlaps.values))

  elif overlaps_ndims == 2:  # no batch dimension
    assert not isinstance(overlaps, ragged_tensor.RaggedTensor)
    overlap_positions = array_ops.where(overlaps)
    return ragged_tensor.RaggedTensor.from_value_rowids(
        values=overlap_positions[:, 1],
        value_rowids=overlap_positions[:, 0],
        nrows=array_ops.shape(overlaps, out_type=dtypes.int64)[0])

  else:  # batch dimension
    if not isinstance(overlaps, ragged_tensor.RaggedTensor):
      overlaps = ragged_tensor.RaggedTensor.from_tensor(overlaps, ragged_rank=1)
    overlap_positions = ragged_where_op.where(overlaps.values)
    if isinstance(overlaps.values, ragged_tensor.RaggedTensor):
      overlaps_values_nrows = overlaps.values.nrows()
    else:
      overlaps_values_nrows = array_ops.shape(overlaps.values,
                                              out_type=dtypes.int64)[0]
    return overlaps.with_values(
        ragged_tensor.RaggedTensor.from_value_rowids(
            values=overlap_positions[:, 1],
            value_rowids=overlap_positions[:, 0],
            nrows=overlaps_values_nrows))


def _check_type(value, name, expected_type):
  """Raises TypeError if not isinstance(value, expected_type)."""
  if not isinstance(value, expected_type):
    raise TypeError('%s must be %s, not %s' % (name, expected_type.__name__,
                                               type(value).__name__))
