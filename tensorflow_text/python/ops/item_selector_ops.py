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

"""Ops for selecting items in RaggedTensors."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor


class ItemSelector(object):
  """A class encapsulating the logic for selecting items.

  `ItemSelector` implementations contain algorithms for selecting items in a
  `RaggedTensor`. Users of `ItemSelector` implementations can call
  `get_selection_mask()` to retrieve a bool `RaggedTensor` mask indicating the
  items that have been selected. For example:

  >>> inputs = tf.ragged.constant([
  ...     [1, 2, 3, 4],
  ...     [100, 200]
  ...   ])
  >>> tf.random.set_seed(1234)
  >>> selector = RandomItemSelector(3, .5)
  >>> selected = selector.get_selection_mask(inputs, axis=1)
  >>> print(selected)
  <tf.RaggedTensor [[False, True, True, False],
                    [False, True]]>


  An important use case for these classes is in creating inputs for masked
  language model training dataset preparation. See `masked_language_model` for
  an example of using the selectors in such a context.

  Subclass writers will typically implement a selection algorithm by overriding
  `get_selection_mask()`.

  A helper function `get_selectable()` is provided to help subclass writers
  filter out excluded items from selection (e.g. CLS and SEP for bert style
  models). This will frequently serve as a prefilter for subclass item
  selection (see e.g. the implementation of `RandomItemSelector`. The base class
  behavior is to simply return the mask obtained by filtering out items listed
  in `unselectable_ids`.
  """

  def __init__(self, unselectable_ids=None):
    """Creates an instance of a `ItemSelector`.

    Args:
      unselectable_ids: a list, or `Tensor` of ids that are not selectable.
    """
    if unselectable_ids is None:
      unselectable_ids = []
    if isinstance(unselectable_ids, list):
      self._unselectable_ids = unselectable_ids
    elif isinstance(unselectable_ids, tensor.Tensor):
      if unselectable_ids.shape.rank not in (1, None):
        raise ValueError(f"`unselectable_ids` must have a rank of 1 or None, "
                         f"but was: {unselectable_ids.shape.rank}")
      self._unselectable_ids = array_ops_stack.unstack(unselectable_ids)
    else:
      raise ValueError("`unselectable_ids` must be either a list or " +
                       "`1 dimensional Tensor`, instead it is a " +
                       str(unselectable_ids))

  @property
  def unselectable_ids(self):
    return self._unselectable_ids

  def get_selectable(self, input_ids, axis):
    """Return a boolean mask of items that can be chosen for selection.

    The default implementation marks all items whose IDs are not in the
    `unselectable_ids` list. This can be overridden if there is a need for
    a more complex or algorithmic approach for selectability.

    Args:
      input_ids: a `RaggedTensor`.
      axis: axis to apply selection on.

    Returns:
      a `RaggedTensor` with dtype of bool and with shape
      `input_ids.shape[:axis]`. Its values are True if the
      corresponding item (or broadcasted subitems) should be considered for
      masking. In the default implementation, all `input_ids` items that are not
      listed in `unselectable_ids` (from the class arg) are considered
      selectable.
    """
    # merge to the desired axis
    input_ids = input_ids.merge_dims(1, axis) if axis > 1 else input_ids

    all_selectable_flats = [
        ragged_functional_ops.map_flat_values(math_ops.not_equal, input_ids,
                                              i).flat_values
        for i in self._unselectable_ids
    ]

    # if there are no blocked ids, mark everything as selectable
    if all_selectable_flats:
      reduce_flat = math_ops.reduce_all(all_selectable_flats, axis=0)
    else:
      reduce_flat = array_ops.ones_like(
          input_ids.flat_values, dtype=dtypes.bool)

    # reduce to the requested axis and broadcast to match original shape
    axis = array_ops.get_positive_axis(
        axis, input_ids.ragged_rank + input_ids.flat_values.shape.rank)
    results = input_ids.with_flat_values(reduce_flat)
    if axis < input_ids.ragged_rank:
      reduce_axis = list(range(input_ids.ragged_rank, axis, -1))
      results = math_ops.reduce_all(results, reduce_axis)

    return results

  def get_selection_mask(self, input_ids, axis=1):
    """Returns a mask of items that have been selected.

    The default implementation simply returns all items not excluded by
    `get_selectable`.

    Args:
      input_ids: A `RaggedTensor`.
      axis: (optional) An int detailing the dimension to apply selection on.
        Default is the 1st dimension.

    Returns:
      a `RaggedTensor` with shape `input_ids.shape[:axis]`. Its values are True
      if the corresponding item (or broadcasted subitems) should be selected.
    """
    return self.get_selectable(input_ids, axis)


class RandomItemSelector(ItemSelector):
  """An `ItemSelector` implementation that randomly selects items in a batch.

    `RandomItemSelector` randomly selects items in a batch subject to
    restrictions given (max_selections_per_batch, selection_rate and
    unselectable_ids).

    Example:
    >>> vocab = ["[UNK]", "[MASK]", "[RANDOM]", "[CLS]", "[SEP]",
    ...          "abc", "def", "ghi"]
    >>> # Note that commonly in masked language model work, there are
    >>> # special tokens we don't want to mask, like CLS, SEP, and probably
    >>> # any OOV (out-of-vocab) tokens here called UNK.
    >>> # Note that if e.g. there are bucketed OOV tokens in the code,
    >>> # that might be a use case for overriding `get_selectable()` to
    >>> # exclude a range of IDs rather than enumerating them.
    >>> tf.random.set_seed(1234)
    >>> selector = tf_text.RandomItemSelector(
    ...     max_selections_per_batch=2,
    ...     selection_rate=0.2,
    ...     unselectable_ids=[0, 3, 4])  # indices of UNK, CLS, SEP
    >>> selection = selector.get_selection_mask(
    ...     tf.ragged.constant([[3, 5, 7, 7], [4, 6, 7, 5]]), axis=1)
    >>> print(selection)
    <tf.RaggedTensor [[False, False, False, True], [False, False, True, False]]>

    The selection has skipped the first elements (the CLS and SEP token codings)
    and picked random elements from the other elements of the segments -- if
    run with a different random seed the selections might be different.
  """

  def __init__(self,
               max_selections_per_batch,
               selection_rate,
               unselectable_ids=None,
               shuffle_fn=None):
    """Creates instance of `RandomItemSelector`.

    By default the source of randomness will be the one set by
    tf.random.set_seed. Users can adjust this independently by providing
    a separate `shuffle_fn` to the selector.

    Args:
      max_selections_per_batch: An int of the max number of items to mask out.
      selection_rate: The rate at which items are randomly selected.
      unselectable_ids: (optional) A list of python ints or 1D `Tensor` of ints
        which are ids that will be not be masked.
      shuffle_fn: (optional) A function that shuffles a 1D `Tensor`. Default
        uses `tf.random.shuffle`.
    """
    if selection_rate is None:
      raise ValueError("`selection_rate` cannot be None")
    if shuffle_fn is None:
      self._shuffle_fn = random_ops.random_shuffle
    else:
      self._shuffle_fn = shuffle_fn

    self._max_selections_per_batch = max_selections_per_batch
    self._selection_rate = selection_rate
    super().__init__(unselectable_ids)

  @property
  def shuffle_fn(self):
    return self._shuffle_fn

  @property
  def max_selections_per_batch(self):
    return self._max_selections_per_batch

  @property
  def selection_rate(self):
    return self._selection_rate

  def get_selection_mask(self, input_ids, axis):
    selectable = self.get_selectable(input_ids, axis)

    # Run the selection algorithm on positions RT
    positions_flat = math_ops.range(array_ops.size(input_ids.flat_values))
    positions = input_ids.with_flat_values(positions_flat)
    # Mask out positions that are not selectable
    positions = ragged_array_ops.boolean_mask(positions, selectable)

    # merge to the desired axis
    positions = positions.merge_dims(1, axis) if axis > 1 else positions

    # Figure out how many we are going to select
    num_to_select = math_ops.ceil(
        math_ops.cast(positions.row_lengths(), dtypes.float32) *
        self.selection_rate)
    num_to_select = math_ops.minimum(num_to_select,
                                     self.max_selections_per_batch)
    num_to_select = math_ops.cast(num_to_select, dtypes.int64)

    # Shuffle and trim to items that are going to be selected
    def _shuffle_and_trim(x):
      positions, top_n = x
      if isinstance(positions, ragged_tensor.RaggedTensor):
        positions_at_axis = math_ops.range(positions.nrows())
        chosen_positions_at_axis = self._shuffle_fn(positions_at_axis)[:top_n]
        return array_ops.gather(positions, chosen_positions_at_axis)
      else:
        shuffled = self._shuffle_fn(positions)
        return shuffled[:top_n]

    selected_for_mask = map_fn.map_fn(
        _shuffle_and_trim, (positions, num_to_select),
        fn_output_signature=ragged_tensor.RaggedTensorSpec(
            ragged_rank=positions.ragged_rank - 1, dtype=positions.dtype))
    selected_for_mask.flat_values.set_shape([None])

    # Construct the result which is a boolean RT
    # Scatter 1's to positions that have been selected_for_mask
    update_values = array_ops.ones_like(selected_for_mask.flat_values)
    update_values = math_ops.cast(update_values, input_ids.dtype)
    update_indices = selected_for_mask.flat_values
    update_indices = array_ops.expand_dims(update_indices, -1)
    update_indices = math_ops.cast(update_indices, input_ids.dtype)

    results_flat = array_ops.zeros_like(input_ids.flat_values)
    results_flat = gen_array_ops.tensor_scatter_update(
        results_flat, update_indices, update_values)
    results = math_ops.cast(
        input_ids.with_flat_values(results_flat), dtypes.bool)

    if axis < results.ragged_rank:
      reduce_axis = list(range(results.ragged_rank, axis, -1))
      results = math_ops.reduce_all(results, reduce_axis)
    return results


def _get_row_lengths_merged_to_axis(segments, axis=-1):
  """Get the row lengths relative to a desired axis."""
  axis = array_ops.get_positive_axis(axis, segments.shape.ndims) - 1
  row_lengths = ragged_tensor.RaggedTensor.from_nested_row_lengths(
      segments.nested_row_lengths()[axis],
      segments.nested_row_lengths()[:axis])
  for _ in range(axis):
    row_lengths = math_ops.reduce_sum(row_lengths, -1)
  return row_lengths


def _get_selection_mask(original, num_to_select, axis=-1, reverse=False):
  """Computes a selection mask given how many items to select."""
  num_to_select = ops.convert_to_tensor(num_to_select)
  num_to_select = array_ops.reshape(num_to_select, [-1])
  row_lengths = _get_row_lengths_merged_to_axis(original, axis)
  num_to_select = array_ops.broadcast_to(num_to_select,
                                         array_ops.shape(row_lengths))
  num_to_select = math_ops.cast(num_to_select, row_lengths.dtype)
  num_to_select = math_ops.minimum(num_to_select, row_lengths)
  ones = array_ops.ones_like(ragged_math_ops.range(num_to_select))
  ones = math_ops.cast(ones, dtypes.int32)
  zeros_row_length = row_lengths - num_to_select
  zeros = math_ops.cast(
      array_ops.zeros_like(ragged_math_ops.range(zeros_row_length)),
      dtypes.int32)
  if reverse:
    results = array_ops.concat([zeros, ones], 1)
  else:
    results = array_ops.concat([ones, zeros], 1)
  results = math_ops.cast(results, dtypes.bool)
  return results


def _get_first_or_last_n_item_selectable(input_ids,
                                         axis,
                                         all_selectable,
                                         num_to_select,
                                         reverse=False):
  """Selects the first or last N selectable items.

  reverse=True means select last. Otherwise select the first N items.

  Args:
    input_ids: a `RaggedTensor`.
    axis: axis to apply selection on.
    all_selectable: a `RaggedTensor` with dtype of bool and with shape
      `input_ids.shape[:axis]`. Used to identify all selectable ids that could
      be selectable.
    num_to_select: An int which is the number of ids to select.
    reverse: boolean when False means select the first N selectable ids.
      Otherwise `reverse=True` means select the last N selectable ids.

  Returns:
    a `RaggedTensor` with dtype of bool and with shape `input_ids.shape[:axis]`.
    Its values are True for the first or last N selectable items.
  """
  axis = array_ops.get_positive_axis(
      axis, input_ids.ragged_rank + input_ids.flat_values.shape.rank)
  # Create a positions RT and mask out positions that are not selectable
  positions_flat = math_ops.range(array_ops.size(input_ids.flat_values))
  positions = input_ids.with_flat_values(positions_flat)
  selectable_positions = ragged_array_ops.boolean_mask(positions,
                                                       all_selectable)

  # merge to the desired axis
  selectable_positions = selectable_positions.merge_dims(
      1, axis) if axis > 1 else selectable_positions

  # Get a selection mask based off of how many items are desired for selection
  merged_axis = axis - (axis - 1)
  selection_mask = _get_selection_mask(selectable_positions, num_to_select,
                                       merged_axis, reverse)
  # Mask out positions that were not selected.
  selected_positions = ragged_array_ops.boolean_mask(selectable_positions,
                                                     selection_mask)

  # Now that we have all the positions which were chosen, we recreate a mask
  # (matching the original input's shape) where the value is True if it was
  # selected. We do this by creating a "all false" RT and scattering true
  # values to the positions chosen for selection.
  all_true = selected_positions.with_flat_values(
      array_ops.ones_like(selected_positions.flat_values))
  all_false = array_ops.zeros(
      array_ops.shape(input_ids.flat_values), dtypes.int32)
  results_flat = array_ops.tensor_scatter_update(
      all_false, array_ops.expand_dims(selected_positions.flat_values, -1),
      all_true.flat_values)
  results = input_ids.with_flat_values(results_flat)
  results = math_ops.cast(results, dtypes.bool)

  # Reduce until input.shape[:axis]
  for _ in range(input_ids.shape.ndims - axis - 1):
    results = math_ops.reduce_all(results, -1)
  return results


class FirstNItemSelector(ItemSelector):
  """An `ItemSelector` that selects the first `n` items in the batch."""

  def __init__(self, num_to_select, unselectable_ids=None):
    """Creates an instance of `FirstNItemSelector`.

    Example:
    >>> selector = FirstNItemSelector(2)
    >>> selection = selector.get_selection_mask(
    ...     tf.ragged.constant([[1, 2, 3], [5, 6, 7, 8]]), axis=1)
    >>> print(selection)
    <tf.RaggedTensor [[True, True, False], [True, True, False, False]]>

    This kind of selection mechanism is useful for batch trimming operations,
    e.g. for `RoundRobinTrimmer`.

    Args:
      num_to_select: An int which is the leading number of items to select.
      unselectable_ids: (optional) A list of int ids that cannot be selected.
        Default is empty list.
    """
    super().__init__(unselectable_ids)
    self._num_to_select = num_to_select

  def get_selectable(self, input_ids, axis):
    """See `get_selectable()` in superclass."""
    all_selectable = super().get_selectable(input_ids, axis)
    return _get_first_or_last_n_item_selectable(
        input_ids=input_ids, axis=axis, all_selectable=all_selectable,
        num_to_select=self._num_to_select, reverse=False)


class LastNItemSelector(ItemSelector):
  """An `ItemSelector` that selects the last `n` items in the batch."""

  def __init__(self, num_to_select, unselectable_ids=None):
    """Creates an instance of `LastNItemSelector`.

    Example:
    >>> selector = LastNItemSelector(2)
    >>> selection = selector.get_selection_mask(
    ...     tf.ragged.constant([[1, 2, 3, 4], [5, 6, 7]]), axis=1)
    >>> print(selection)
    <tf.RaggedTensor [[False, False, True, True], [False, True, True]]>

    Args:
      num_to_select: An int which is the leading number of items to select.
      unselectable_ids: (optional) A list of int ids that cannot be selected.
        Default is empty list.
    """
    super().__init__(unselectable_ids)
    self._num_to_select = num_to_select

  def get_selectable(self, input_ids, axis):
    """See `get_selectable()` in superclass."""
    all_selectable = super().get_selectable(input_ids, axis)
    return _get_first_or_last_n_item_selectable(
        input_ids=input_ids, axis=axis, all_selectable=all_selectable,
        num_to_select=self._num_to_select, reverse=True)


class NothingSelector(ItemSelector):
  """An `ItemSelector` that selects nothing."""

  def __init__(self):
    super().__init__([])

  def get_selectable(self, input_ids, axis):
    """Returns a prefilter mask which excludes all items."""
    flat_false_values = math_ops.cast(
        array_ops.zeros_like(input_ids.flat_values), dtypes.bool)
    results = input_ids.with_flat_values(flat_false_values)
    for _ in range(input_ids.ragged_rank - axis):
      results = math_ops.reduce_all(results, -1)
    return results
