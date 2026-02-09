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

"""Library of ops to truncate segments."""
import abc

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow_text.python.ops import item_selector_ops

# pylint: disable=g-bad-import-order
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_trimmer_ops = load_library.load_op_library(resource_loader.get_path_to_datafile('_trimmer_ops.so'))
from tensorflow.python.ops import while_loop


class Trimmer(metaclass=abc.ABCMeta):
  """Truncates a list of segments using a pre-determined truncation strategy."""

  def trim(self, segments):
    """Truncate the list of `segments`.

    Truncate the list of `segments` using the truncation strategy defined by
    `generate_mask`.

    Args:
      segments: A list of `RaggedTensor`s w/ shape [num_batch, (num_items)].

    Returns:
      a list of `RaggedTensor`s with len(segments) number of items and where
      each item has the same shape as its counterpart in `segments` and
      with unwanted values dropped. The values are dropped according to the
      `TruncationStrategy` defined.
    """
    with ops.name_scope("Trimmer/Trim"):
      segments = [
          ragged_tensor.convert_to_tensor_or_ragged_tensor(s) for s in segments
      ]
      truncate_masks = self.generate_mask(segments)
      truncated_segments = [
          ragged_array_ops.boolean_mask(
              seg, mask.with_row_splits_dtype(seg.row_splits.dtype))
          for seg, mask in zip(segments, truncate_masks)
      ]
      return truncated_segments

  @abc.abstractmethod
  def generate_mask(self, segments):
    """Generates a boolean mask specifying which portions of `segments` to drop.

    Users should be able to use the results of generate_mask() to drop items
    in segments using `tf.ragged.boolean_mask(seg, mask)`.

    Args:
      segments: A list of `RaggedTensor` each w/ a shape of [num_batch,
        (num_items)].

    Returns:
      a list with len(segments) number of items and where each item is a
      `RaggedTensor` with the same shape as its counterpart in `segments` and
      with a boolean dtype where each value is True if the corresponding
      value in `segments` should be kept and False if it should be dropped
      instead.
    """
    raise NotImplementedError()


def _get_row_lengths(segments, axis=-1):
  axis = array_ops.get_positive_axis(axis, segments.shape.ndims) - 1
  foo = ragged_tensor.RaggedTensor.from_nested_row_lengths(
      segments.nested_row_lengths()[axis],
      segments.nested_row_lengths()[:axis])
  for _ in range(axis):
    foo = math_ops.reduce_sum(foo, -1)
  return foo


class WaterfallTrimmer(Trimmer):
  """A `Trimmer` that allocates a length budget to segments in order.

  A `Trimmer` that allocates a length budget to segments in order. It selects
  elements to drop, according to a max sequence length budget, and then applies
  this mask to actually drop the elements. See `generate_mask()` for more
  details.

  Example:

  >>> a = tf.ragged.constant([['a', 'b', 'c'], [], ['d']])
  >>> b = tf.ragged.constant([['1', '2', '3'], [], ['4', '5', '6', '7']])
  >>> trimmer = tf_text.WaterfallTrimmer(4)
  >>> trimmer.trim([a, b])
  [<tf.RaggedTensor [[b'a', b'b', b'c'], [], [b'd']]>,
   <tf.RaggedTensor [[b'1'], [], [b'4', b'5', b'6']]>]

  Here, for the first pair of elements, `['a', 'b', 'c']` and `['1', '2', '3']`,
  the `'2'` and `'3'` are dropped to fit the sequence within the max sequence
  length budget.
  """

  def __init__(self, max_seq_length, axis=-1):
    """Creates an instance of `WaterfallTruncator`.

    Args:
      max_seq_length: a scalar `Tensor` or a 1D `Tensor` of type int32 that
        describes the number max number of elements allowed in a batch. If a
        scalar is provided, the value is broadcasted and applied to all values
        across the batch.
      axis: Axis to apply trimming on.
    """
    self._max_seq_length = max_seq_length
    self._axis = axis

  def generate_mask(self, segments):
    """Calculates a truncation mask given a per-batch budget.

    Calculate a truncation mask given a budget of the max number of items for
    each or all batch row. The allocation of the budget is done using a
    'waterfall' algorithm. This algorithm allocates quota in a left-to-right
    manner and fill up the buckets until we run out of budget.

    For example if the budget of [5] and we have segments of size
    [3, 4, 2], the truncate budget will be allocated as [3, 2, 0].

    The budget can be a scalar, in which case the same budget is broadcasted
    and applied to all batch rows. It can also be a 1D `Tensor` of size
    `batch_size`, in which each batch row i will have a budget corresponding to
    `per_batch_quota[i]`.

    Example:

    >>> a = tf.ragged.constant([['a', 'b', 'c'], [], ['d']])
    >>> b = tf.ragged.constant([['1', '2', '3'], [], ['4', '5', '6', '7']])
    >>> trimmer = tf_text.WaterfallTrimmer(4)
    >>> trimmer.generate_mask([a, b])
    [<tf.RaggedTensor [[True, True, True], [], [True]]>,
     <tf.RaggedTensor [[True, False, False], [], [True, True, True, False]]>]

    Args:
      segments: A list of `RaggedTensor` each w/ a shape of [num_batch,
        (num_items)].

    Returns:
      a list with len(segments) of `RaggedTensor`s, see superclass for details.
    """
    with ops.name_scope("WaterfallTrimmer/generate_mask"):
      segment_row_lengths = [_get_row_lengths(s, self._axis) for s in segments]
      segment_row_lengths = array_ops_stack.stack(segment_row_lengths, axis=-1)

      # Broadcast budget to match the rank of segments[0]
      budget = ops.convert_to_tensor(self._max_seq_length)
      for _ in range(segments[0].shape.ndims - budget.shape.ndims):
        budget = array_ops.expand_dims(budget, -1)

      # Compute the allocation for each segment using a `waterfall` algorithm
      segment_lengths = math_ops.cast(segment_row_lengths, dtypes.int32)
      budget = math_ops.cast(budget, dtypes.int32)
      leftover_budget = math_ops.cumsum(
          -1 * segment_lengths, exclusive=False, axis=-1) + budget
      leftover_budget = segment_lengths + math_ops.minimum(leftover_budget, 0)
      results = math_ops.maximum(leftover_budget, 0)

      # Translate the results into boolean masks that match the shape of each
      # segment
      results = array_ops_stack.unstack(results, axis=-1)
      item_selectors = [
          item_selector_ops.FirstNItemSelector(i) for i in results
      ]
      return [
          i.get_selectable(s, self._axis)
          for s, i in zip(segments, item_selectors)
      ]


def _round_robin_allocation(row_lengths, max_seq_length):
  """Allocating quota via round robin algorithm."""
  distribution = array_ops.zeros_like(row_lengths)
  i = constant_op.constant(0)
  batch_size = array_ops.shape(row_lengths)[0]
  num_segments = array_ops.shape(row_lengths)[1]
  quota_used = array_ops.zeros([batch_size], dtypes.int32)
  max_seq_length_bc = max_seq_length + 0 * quota_used

  def _cond(i, dist, quota_used):
    del i
    have_quota = quota_used < max_seq_length_bc
    have_space = math_ops.reduce_any(dist < row_lengths, 1)
    return math_ops.reduce_any(math_ops.logical_and(have_quota, have_space))

  def _body(i, dist, quota_used):
    index = math_ops.mod(i, num_segments)
    updates = array_ops.where(dist[..., index] < row_lengths[..., index],
                              array_ops.ones_like(dist[..., index]),
                              array_ops.zeros_like(dist[..., index]))
    scatter_index = array_ops.tile([index], [batch_size])
    scatter_index = array_ops.expand_dims(scatter_index, -1)
    batch_dim = array_ops.reshape(math_ops.range(batch_size), [batch_size, 1])
    scatter_index_2d = array_ops.concat([batch_dim, scatter_index], -1)
    new_dist = array_ops.tensor_scatter_add(dist, scatter_index_2d, updates)
    return i + 1, new_dist, quota_used + updates

  _, results, _ = while_loop.while_loop(_cond, _body,
                                        (i, distribution, quota_used))
  return results


class RoundRobinTrimmer(Trimmer):
  """A `Trimmer` that allocates a length budget to segments via round robin.

  A `Trimmer` that allocates a length budget to segments using a round robin
  strategy, then drops elements outside of the segment's allocated budget.
  See `generate_mask()` for more details.
  """

  def __init__(self, max_seq_length, axis=-1):
    """Creates an instance of `RoundRobinTrimmer`.

    Args:
      max_seq_length: a scalar `Tensor` int32 that describes the number max
        number of elements allowed in a batch.
      axis: Axis to apply trimming on.
    """
    if (isinstance(max_seq_length, tensor.Tensor) and
        max_seq_length.shape.ndims > 0):
      self._max_seq_length = array_ops.reshape(max_seq_length, ())
    else:
      self._max_seq_length = max_seq_length
    self._axis = axis

  def generate_mask(self, segments):
    """Calculates a truncation mask given a per-batch budget.

    Calculate a truncation mask given a budget of the max number of items for
    each or all batch row. The allocation of the budget is done using a
    'round robin' algorithm. This algorithm allocates quota in each bucket,
    left-to-right repeatedly until all the buckets are filled.

    For example if the budget of [5] and we have segments of size
    [3, 4, 2], the truncate budget will be allocated as [2, 2, 1].

    Args:
      segments: A list of `RaggedTensor`s each with a shape of [num_batch,
        (num_items)].

    Returns:
      A list with len(segments) of `RaggedTensor`s, see superclass for details.
    """
    with ops.name_scope("RoundRobinTrimmer/generate_mask"):
      segments = list(
          map(ragged_tensor.convert_to_tensor_or_ragged_tensor, segments)
      )
      # The docs state that the segments argument is required to be a list of
      # RaggedTensors of rank 2. However, the python-only op worked with other
      # ranks, so we continue to execute that code for other ranks so as to not
      # break current models.
      rank_2 = segments[0].shape.ndims == 2
      last_axis = self._axis == -1 or self._axis == 1
      if not last_axis or not rank_2:
        segment_row_lengths = [
            _get_row_lengths(s, self._axis) for s in segments
        ]
        segment_row_lengths = array_ops_stack.stack(
            segment_row_lengths, axis=-1)
        segment_row_lengths = math_ops.cast(segment_row_lengths, dtypes.int32)
        budget = ops.convert_to_tensor(self._max_seq_length)
        results = _round_robin_allocation(segment_row_lengths, budget)

        results = array_ops_stack.unstack(results, axis=-1)
        item_selectors = [
            item_selector_ops.FirstNItemSelector(i) for i in results
        ]
        return [
            i.get_selectable(s, self._axis)
            for s, i in zip(segments, item_selectors)
        ]
      else:
        values = list(map(lambda x: x.values, segments))
        row_splits = list(map(lambda x: x.row_splits, segments))
        o_masks = gen_trimmer_ops.tf_text_round_robin_generate_masks(
            self._max_seq_length, values, row_splits
        )
        return [
            ragged_tensor.RaggedTensor.from_row_splits(m, s)
            for m, s in zip(o_masks, row_splits)
        ]

  def trim(self, segments):
    """Truncate the list of `segments`.

    Truncate the list of `segments` using the 'round-robin' strategy which
    allocates quota in each bucket, left-to-right repeatedly until all buckets
    are filled.

    For example if the budget of [5] and we have segments of size
    [3, 4, 2], the truncate budget will be allocated as [2, 2, 1].

    Args:
      segments: A list of `RaggedTensor`s w/ shape [num_batch, (num_items)].

    Returns:
      A list with len(segments) of `RaggedTensor`s, see superclass for details.
    """
    with ops.name_scope("RoundRobinTrimmer/trim"):
      segments = [
          ragged_tensor.convert_to_tensor_or_ragged_tensor(s) for s in segments
      ]
      # The docs state that the segments argument is required to be a list of
      # RaggedTensors of rank 2. However, the python-only op worked with other
      # ranks, so we continue to execute that code for other ranks so as to not
      # break current models.
      rank_2 = segments[0].shape.ndims == 2
      last_axis = self._axis == -1 or self._axis == 1
      if not last_axis or not rank_2:
        truncate_masks = self.generate_mask(segments)
        truncated_segments = [
            ragged_array_ops.boolean_mask(
                seg, mask.with_row_splits_dtype(seg.row_splits.dtype)
            )
            for seg, mask in zip(segments, truncate_masks)
        ]
        return truncated_segments
      else:
        values = list(map(lambda x: x.values, segments))
        row_splits = list(map(lambda x: x.row_splits, segments))
        (o_values, o_splits) = gen_trimmer_ops.tf_text_round_robin_trim(
            self._max_seq_length, values, row_splits
        )
        return [
            ragged_tensor.RaggedTensor.from_row_splits(m, s)
            for m, s in zip(o_values, o_splits)
        ]


def _shrink_longest_allocation(segment_lengths, max_row_length):
  """Allocating quota via a shrink-longest strategy."""
  depth = array_ops.shape(segment_lengths)[-1]

  def _condition(l):
    return math_ops.reduce_any(
        math_ops.greater(math_ops.reduce_sum(l, axis=-1), max_row_length))

  def _body(l):
    needs_truncation = math_ops.cast(
        math_ops.greater(math_ops.reduce_sum(l, axis=-1), max_row_length),
        dtypes.int32)
    minus_ones_or_zeros = array_ops.expand_dims(-1 * needs_truncation, axis=1)
    one_hot_to_max_position = array_ops.one_hot(
        math_ops.argmax(l, axis=-1), depth=depth, dtype=dtypes.int32)
    return (l + minus_ones_or_zeros * one_hot_to_max_position,)

  return while_loop.while_loop_v2(
      cond=_condition, body=_body, loop_vars=[segment_lengths])


class ShrinkLongestTrimmer(Trimmer):
  """A `Trimmer` that truncates the longest segment.

  A `Trimmer` that allocates a length budget to segments by shrinking whatever
  is the longest segment at each round at the end, until the total length of
  segments is no larger than the allocated budget.
  See `generate_mask()` for more details.
  """

  def __init__(self, max_seq_length, axis=-1):
    self._max_seq_length = max_seq_length
    self._axis = axis

  def generate_mask(self, segments):
    """Calculates a truncation mask given a per-batch budget.

    Calculate a truncation mask given a budget of the max number of items for
    each batch row. The allocation of the budget is done using a
    'shrink the largest segment' algorithm. This algorithm identifies the
    currently longest segment (in cases of tie, picking whichever segment occurs
    first) and reduces its length by 1 by dropping its last element, repeating
    until the total length of segments is no larger than `_max_seq_length`.

    For example if the budget is [7] and we have segments of size
    [3, 4, 4], the truncate budget will be allocated as [2, 2, 3], going through
    truncation steps
      # Truncate the second segment.
      [3, 3, 4]
      # Truncate the last segment.
      [3, 3, 3]
      # Truncate the first segment.
      [2, 3, 3]
      # Truncate the second segment.
      [2, 2, 3]

    Args:
      segments: A list of `RaggedTensor` each w/ a shape of [num_batch,
        (num_items)].

    Returns:
      a list with len(segments) of `RaggedTensor`s, see superclass for details.
    """
    with ops.name_scope("ShrinkLongestTrimmer/generate_mask"):
      segment_row_lengths = [_get_row_lengths(s, self._axis) for s in segments]
      segment_row_lengths = array_ops_stack.stack(segment_row_lengths, axis=-1)
      segment_row_lengths = math_ops.cast(segment_row_lengths, dtypes.int32)
      budget = ops.convert_to_tensor(self._max_seq_length)
      results = array_ops_stack.unstack(
          _shrink_longest_allocation(segment_row_lengths, budget), axis=-1)

      item_selectors = [
          item_selector_ops.FirstNItemSelector(i) for i in results
      ]
      return [
          i.get_selectable(s, self._axis)
          for s, i in zip(segments, item_selectors)
      ]
