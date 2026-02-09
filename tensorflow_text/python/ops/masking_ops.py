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

"""Ops for applying language model masking dynamically to inputs."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_where_op
from tensorflow_text.python.ops import item_selector_ops


# TODO(b/166323018): Replace once tensor_scatter_nd_update for RaggedTensor is
#                    available.
def _ragged_tensor_scatter_nd_update(params, indices, updates):
  """Version of tensor_scatter_nd_update() where the values are ragged."""
  # Create a RT in the shape of `params` and containing the "global" positions.
  # Here "global" means the element position in the flat values Tensor.
  global_positions_flat = math_ops.range(array_ops.size(params.flat_values))
  global_positions = params.with_flat_values(global_positions_flat)

  global_indices = array_ops.batch_gather(global_positions, indices)
  update_indices = global_indices.flat_values
  update_indices = array_ops.expand_dims(update_indices, -1)
  update_indices = math_ops.cast(update_indices, params.dtype)
  params_flat = params.flat_values
  update_values = math_ops.cast(updates.flat_values, params_flat.dtype)
  results_flat = array_ops.tensor_scatter_update(
      params_flat, update_indices, update_values)
  return params.with_flat_values(results_flat)


def _get_random(positions):
  """Get a random tensor like `positions`."""
  flat_random = random_ops.random_uniform(
      array_ops.shape(positions.flat_values), 0, 1, dtype=dtypes.float32)
  return positions.with_flat_values(flat_random)


def _get_selected_item_positions(item_selector, input_ids, axis=1):
  """Get the positions of the items that have been selected.

  Args:
    item_selector: an instance of `ItemSelector`.
    input_ids: a `RaggedTensor` with n dimensions, whose items will be
      selected on.
    axis: (optional) An int detailing the dimension to apply selection on.
      Default is the 1st dimension.

  Returns:
    A `RaggedTensor` of int64s, with rank 2, shape
   [batch, (num_selections)] and whose values are the positions of items
   that have been selected.
  """
  original_input_ids = input_ids

  # select items for masking
  selected_for_mask = item_selector.get_selection_mask(input_ids, axis)

  # create a positions RT
  original_input_ids = (
      original_input_ids.merge_dims(1, -1)
      if original_input_ids.ragged_rank > 1 else original_input_ids)
  positions = ragged_math_ops.range(original_input_ids.row_lengths())
  positions = input_ids.with_flat_values(positions.flat_values)

  # drop out not-masked positions
  results = ragged_array_ops.boolean_mask(positions, selected_for_mask)
  results = results.merge_dims(1, -1) if results.ragged_rank > 1 else results
  return results


def mask_language_model(
    input_ids,
    item_selector,
    mask_values_chooser,
    axis=1):
  """Applies dynamic language model masking.

  `mask_language_model` implements the `Masked LM and Masking Procedure`
  described in `BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding`  (https://arxiv.org/pdf/1810.04805.pdf).
  `mask_language_model` uses an `ItemSelector` to select the items for masking,
  and a `MaskValuesChooser` to assign the values to the selected items.
  The purpose of this is to bias the representation towards the actual
  observed item.

  Masking is performed on items in an axis. A decision is taken independently at
  random to mask with [MASK], mask with random tokens from the full vocab, or
  not mask at all. Note that the masking decision is broadcasted to the
  sub-dimensions.

  For example, in a RaggedTensor of shape `[batch, (wordpieces)]` and if axis=1,
  each wordpiece independently gets masked (or not).

  With the following input:

  ```
  [[b"Sp", b"##onge", b"bob", b"Sq", b"##uare", b"##pants" ],
  [b"Bar", b"##ack", b"Ob", b"##ama"],
  [b"Mar", b"##vel", b"A", b"##ven", b"##gers"]],
  ```

  `mask_language_model` could end up masking individual wordpieces:

  ```
  [[b"[MASK]", b"##onge", b"bob", b"Sq", b"[MASK]", b"##pants" ],
  [b"Bar", b"##ack", b"[MASK]", b"##ama"],
  [b"[MASK]", b"##vel", b"A", b"##ven", b"##gers"]]
  ```

  Or with random token inserted:

  ```
  [[b"[MASK]", b"##onge", b"bob", b"Sq", b"[MASK]", b"##pants" ],
  [b"Bar", b"##ack", b"Sq", b"##ama"],   # random token inserted for 'Ob'
  [b"Bar", b"##vel", b"A", b"##ven", b"##gers"]]  # random token inserted for
                                                  # 'Mar'
  ```

  In a RaggedTensor of shape `[batch, (words), (wordpieces)]`, whole words get
  masked (or not). If a word gets masked, all its tokens are independently
  either replaced by `[MASK]`, by random tokens, or no substitution occurs.
  Note that any arbitrary spans that can be constructed by a `RaggedTensor` can
  be masked in the same way.

  For example, if we have an `RaggedTensor` with shape
  `[batch, (token), (wordpieces)]`:

  ```
  [[[b"Sp", "##onge"], [b"bob"], [b"Sq", b"##uare", b"##pants"]],
   [[b"Bar", "##ack"], [b"Ob", b"##ama"]],
   [[b"Mar", "##vel"], [b"A", b"##ven", b"##gers"]]]
  ```

  `mask_language_model` could mask whole spans (items grouped together
  by the same 1st dimension):

  ```
  [[[b"[MASK]", "[MASK]"], [b"bob"], [b"Sq", b"##uare", b"##pants"]],
   [[b"Bar", "##ack"], [b"[MASK]", b"[MASK]"]],
   [[b"[MASK]", "[MASK]"], [b"A", b"##ven", b"##gers"]]]
  ```

   or insert random items in spans:

  ```
   [[[b"Mar", "##ama"], [b"bob"], [b"Sq", b"##uare", b"##pants"]],
    [[b"Bar", "##ack"], [b"##onge", b"##gers"]],
    [[b"Ob", "Sp"], [b"A", b"##ven", b"##gers"]]]
  ```

  Args:
    input_ids: A `RaggedTensor` of n dimensions (where n >= 2) on which
      masking will be applied to items up to dimension 1.
    item_selector: An instance of `ItemSelector` that is used for selecting
      items to be masked.
    mask_values_chooser: An instance of `MaskValuesChooser` which determines the
      values assigned to the ids chosen for masking.
    axis: the axis where items will be treated atomically for masking.
  Returns:
    A tuple of (masked_input_ids, masked_positions, masked_ids) where:

    masked_input_ids: A `RaggedTensor` in the same shape and dtype as
      `input_ids`, but with items in `masked_positions` possibly replaced
      with `mask_token`, random id, or no change.
    masked_positions: A `RaggedTensor` of ints with shape
      [batch, (num_masked)] containing the positions of items selected for
      masking.
    masked_ids: A `RaggedTensor` with shape [batch, (num_masked)] and same
      type as `input_ids` containing the original values before masking
      and thus used as labels for the task.
  """
  if not isinstance(item_selector, item_selector_ops.ItemSelector):
    raise ValueError("`item_selector` must be an instance of `ItemSelector`")

  if not isinstance(mask_values_chooser, MaskValuesChooser):
    raise ValueError("`mask_values_chooser` must be an instance of " +
                     "`MaskValuesChooser`")

  input_ids = ragged_tensor.convert_to_tensor_or_ragged_tensor(input_ids)

  # Identify the items that are maskable and obtain their positions in the
  # rank 2 space.
  masked_token_positions = _get_selected_item_positions(
      item_selector, input_ids, axis)

  # Flatten everything down to a 2D RaggedTensor
  masked_token_positions = (
      masked_token_positions if masked_token_positions.ragged_rank == 1 else
      masked_token_positions.merge_dims(1, -1))
  input_ids = (
      input_ids if input_ids.ragged_rank == 1 else input_ids.merge_dims(1, -1))

  # Gather all the current ids in the places selected for masking.
  masked_lm_ids = array_ops.batch_gather(input_ids, masked_token_positions)

  # Figure out what we are going to replace these values with -- either masked
  # token, random int id, or do nothing.
  mask_values = mask_values_chooser.get_mask_values(masked_lm_ids)

  # scatter the new mask values back to their respective positions
  new_input_ids = _ragged_tensor_scatter_nd_update(input_ids,
                                                   masked_token_positions,
                                                   mask_values)
  return new_input_ids, masked_token_positions, masked_lm_ids


class MaskValuesChooser(object):
  """Assigns values to the items chosen for masking.

  `MaskValuesChooser` encapsulates the logic for deciding the value to assign
  items that where chosen for masking. The following are the behavior in the
  default implementation:

  For `mask_token_rate` of the time, replace the item with the `[MASK]` token:

  ```
  my dog is hairy -> my dog is [MASK]
  ```

  For `random_token_rate` of the time, replace the item with a random word:

  ```
  my dog is hairy -> my dog is apple
  ```

  For `1 - mask_token_rate - random_token_rate` of the time, keep the item
  unchanged:

  ```
  my dog is hairy -> my dog is hairy.
  ```

  The default behavior is consistent with the methodology specified in
  `Masked LM and Masking Procedure` described in `BERT: Pre-training of Deep
  Bidirectional Transformers for Language Understanding`
  (https://arxiv.org/pdf/1810.04805.pdf).

  Users may further customize this with behavior through subclassing and
  overriding `get_mask_values()`.
  """

  def __init__(self,
               vocab_size,
               mask_token,
               mask_token_rate=0.8,
               random_token_rate=0.1):
    """Creates an instance of `MaskValueChooser`.

    Args:
      vocab_size: size of vocabulary.
      mask_token: The id of the mask token.
      mask_token_rate: (optional) A float between 0 and 1 which indicates how
        often the `mask_token` is substituted for tokens selected for masking.
        Default is 0.8, NOTE: `mask_token_rate` + `random_token_rate` <= 1.
      random_token_rate: A float between 0 and 1 which indicates how often a
        random token is substituted for tokens selected for masking. Default is
        0.1. NOTE: `mask_token_rate` + `random_token_rate` <= 1.
    """
    if mask_token_rate is None:
      raise ValueError("`mask_token_rate` cannot be None")
    if random_token_rate is None:
      raise ValueError("`random_token_rate` cannot be None")
    self._mask_token_rate = mask_token_rate
    self._random_token_rate = random_token_rate
    self._mask_token = mask_token
    self._vocab_size = vocab_size

  @property
  def mask_token(self):
    return self._mask_token

  @property
  def random_token_rate(self):
    return self._random_token_rate

  @property
  def vocab_size(self):
    return self._vocab_size

  def get_mask_values(self, masked_lm_ids):
    """Get the values used for masking, random injection or no-op.

    Args:
      masked_lm_ids: a `RaggedTensor` of n dimensions and dtype int32 or int64
        whose values are the ids of items that have been selected for masking.
    Returns:
      a `RaggedTensor` of the same dtype and shape with `masked_lm_ids` whose
      values contain either the mask token, randomly injected token or original
      value.
    """
    validate_rates = control_flow_assert.Assert(
        self._mask_token_rate + self._random_token_rate <= 1,
        ["mask_token_rate + random_token_rate must be <= 1"])
    with ops.control_dependencies([validate_rates]):

      # Generate a random number for all mask-able items. Items that should be
      # treated atomically (e.g. all wordpieces in a token, span, etc) will have
      # the same random number.
      random_uniform = _get_random(masked_lm_ids)

      # Merge down to rank 2.
      random_uniform = (
          random_uniform if random_uniform.ragged_rank == 1 else
          random_uniform.merge_dims(1, -1))
      mask_values = masked_lm_ids

      all_mask_flat = array_ops.tile([self._mask_token],
                                     array_ops.shape(mask_values.flat_values))

      # Maybe add mask token `mask_token_rate`% of the time
      should_mask_flat = random_uniform.flat_values < math_ops.cast(
          self._mask_token_rate, dtypes.float32)
      mask_values = mask_values.with_flat_values(
          ragged_where_op.where(
              should_mask_flat,
              x=math_ops.cast(all_mask_flat, mask_values.flat_values.dtype),
              y=mask_values.flat_values))

      # Maybe inject random token `random_token_rate`% of the time.
      all_random_flat = random_ops.random_uniform(
          array_ops.shape(mask_values.flat_values), maxval=math_ops.cast(
              self._vocab_size, dtypes.float32))
      should_inject_random_flat = math_ops.logical_and(
          random_uniform.flat_values > self._mask_token_rate,
          random_uniform.flat_values <
          self._mask_token_rate + self._random_token_rate)
      mask_values = mask_values.with_flat_values(
          ragged_where_op.where(
              should_inject_random_flat,
              x=math_ops.cast(all_random_flat, mask_values.flat_values.dtype),
              y=mask_values.flat_values))
      return mask_values
