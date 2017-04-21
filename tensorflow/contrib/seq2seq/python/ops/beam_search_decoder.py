# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""A decoder that performs beam search."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from tensorflow.contrib.rnn import core_rnn_cell
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.util import nest


__all__ = [
    "BeamSearchDecoderOutput",
    "BeamSearchDecoderState",
    "BeamSearchDecoder",
    "FinalBeamSearchDecoderOutput",
]


class BeamSearchDecoderState(
    collections.namedtuple("BeamSearchDecoderState", ("cell_state", "log_probs",
                                                      "finished", "lengths"))):
  pass


class BeamSearchDecoderOutput(
    collections.namedtuple("BeamSearchDecoderOutput",
                           ("scores", "predicted_ids", "parent_ids"))):
  pass


class FinalBeamSearchDecoderOutput(
    collections.namedtuple("FinalBeamDecoderOutput",
                           ["predicted_ids", "beam_search_decoder_output"])):
  """Final outputs returned by the beam search after all decoding is finished.

  Args:
    predicted_ids: The final prediction. A tensor of shape
      `[T, batch_size, beam_width]`.
    beam_search_output: An instance of `BeamSearchDecoderOutput` that describes
      the state of the beam search.
  """
  pass


class BeamSearchDecoder(decoder.Decoder):
  """BeamSearch sampling decoder."""

  def __init__(self,
               cell,
               embedding,
               start_tokens,
               end_token,
               initial_state,
               beam_width,
               output_layer=None,
               length_penalty_weight=0.0):
    """Initialize BeamSearchDecoder.

    Args:
      cell: An `RNNCell` instance.
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
      beam_width:  Python integer, the number of beams.
      output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`.  Optional layer to apply to the RNN output prior
        to storing the result or sampling.
      length_penalty_weight: Float weight to penalize length. Disabled with 0.0.

    Raises:
      TypeError: if `cell` is not an instance of `RNNCell`,
        or `output_layer` is not an instance of `tf.layers.Layer`.
      ValueError: If `start_tokens` is not a vector or
        `end_token` is not a scalar.
    """
    if not isinstance(cell, core_rnn_cell.RNNCell):
      raise TypeError("cell must be an RNNCell, received: %s" % type(cell))
    if (output_layer is not None
        and not isinstance(output_layer, layers_base._Layer)):  # pylint: disable=protected-access
      raise TypeError(
          "output_layer must be a Layer, received: %s" % type(output_layer))
    self._cell = cell
    self._output_layer = output_layer

    if callable(embedding):
      self._embedding_fn = embedding
    else:
      self._embedding_fn = (
          lambda ids: embedding_ops.embedding_lookup(embedding, ids))

    self._start_tokens = ops.convert_to_tensor(
        start_tokens, dtype=dtypes.int32, name="start_tokens")
    if self._start_tokens.get_shape().ndims != 1:
      raise ValueError("start_tokens must be a vector")
    self._end_token = ops.convert_to_tensor(
        end_token, dtype=dtypes.int32, name="end_token")
    if self._end_token.get_shape().ndims != 0:
      raise ValueError("end_token must be a scalar")

    self._batch_size = array_ops.size(start_tokens)
    self._beam_width = beam_width
    self._length_penalty_weight = length_penalty_weight
    self._initial_cell_state = nest.map_structure(self._maybe_split_batch_beams,
                                                  initial_state)
    self._start_tokens = array_ops.tile(
        array_ops.expand_dims(self._start_tokens, 1), [1, self._beam_width])
    self._start_inputs = self._embedding_fn(self._start_tokens)
    self._finished = array_ops.zeros(
        [self._batch_size, self._beam_width], dtype=dtypes.bool)

  @property
  def batch_size(self):
    return self._batch_size

  def _rnn_output_size(self):
    size = self._cell.output_size
    if self._output_layer is None:
      return size
    else:
      # To use layer's compute_output_shape, we need to convert the
      # RNNCell's output_size entries into shapes with an unknown
      # batch size.  We then pass this through the layer's
      # compute_output_shape and read off all but the first (batch)
      # dimensions to get the output size of the rnn with the layer
      # applied to the top.
      output_shape_with_unknown_batch = nest.map_structure(
          lambda s: tensor_shape.TensorShape([None]).concatenate(s),
          size)
      layer_output_shape = self._output_layer._compute_output_shape(  # pylint: disable=protected-access
          output_shape_with_unknown_batch)
      return nest.map_structure(lambda s: s[1:], layer_output_shape)

  @property
  def output_size(self):
    # Return the cell output and the id
    return BeamSearchDecoderOutput(
        scores=tensor_shape.TensorShape([self._beam_width]),
        predicted_ids=tensor_shape.TensorShape([self._beam_width]),
        parent_ids=tensor_shape.TensorShape([self._beam_width]))

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and int32 (the id)
    dtype = nest.flatten(self._initial_cell_state)[0].dtype
    return BeamSearchDecoderOutput(
        scores=nest.map_structure(lambda _: dtype, self._rnn_output_size()),
        predicted_ids=dtypes.int32,
        parent_ids=dtypes.int32)

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, start_inputs, initial_state)`.
    """
    finished, start_inputs = self._finished, self._start_inputs

    initial_state = BeamSearchDecoderState(
        cell_state=self._initial_cell_state,
        log_probs=array_ops.zeros(
            [self._batch_size, self._beam_width],
            dtype=nest.flatten(self._initial_cell_state)[0].dtype),
        finished=finished,
        lengths=array_ops.zeros(
            [self._batch_size, self._beam_width], dtype=dtypes.int32))

    return (finished, start_inputs, initial_state)

  def finalize(self, outputs, final_state):
    """Finalize and return the predicted_ids.

    Args:
      outputs: An instance of BeamSearchDecoderOutput.
      final_state: An instance of BeamSearchDecoderState. Passed through to the
        output.

    Returns:
      outputs: An instance of FinalBeamSearchDecoderOutput where the
        predicted_ids are the result of calling _gather_tree.
      final_state: The same input instance of BeamSearchDecoderState.
    """
    predicted_ids = _gather_tree(outputs.predicted_ids, outputs.parent_ids)
    outputs = FinalBeamSearchDecoderOutput(
        beam_search_decoder_output=outputs, predicted_ids=predicted_ids)
    return outputs, final_state

  def _merge_batch_beams(self, t):
    """Merges the tensor from a batch of beams into a batch by beams.

    More exactly, t is a tensor of dimension [batch_size, beam_width, ...]. We
    reshape this into [batch_size*beam_width, ...]

    Args:
      t: Tensor of dimension [batch_size, beam_width, ...]

    Returns:
      A reshaped version of t with dimension [batch_size * beam_width, ...].
    """
    t_static_shape = t.shape
    t_shape = array_ops.shape(t)
    static_batch_size = tensor_util.constant_value(self._batch_size)
    batch_size_beam_width = (
        None if static_batch_size is None
        else static_batch_size * self._beam_width)
    reshaped_t = array_ops.reshape(
        t, array_ops.concat(
            ([self._batch_size * self._beam_width], t_shape[2:]), 0))
    reshaped_t.set_shape(
        (tensor_shape.TensorShape([batch_size_beam_width])
         .concatenate(t_static_shape[2:])))
    return reshaped_t

  def _split_batch_beams(self, t):
    """Splits the tensor from a batch by beams into a batch of beams.

    More exactly, t is a tensor of dimension [batch_size*beam_width, ...]. We
    reshape this into [batch_size, beam_width, ...]

    Args:
      t: Tensor of dimension [batch_size*beam_width, ...]

    Returns:
      A reshaped version of t with dimension [batch_size, beam_width, ...].
    """
    t_static_shape = t.shape
    t_shape = array_ops.shape(t)
    reshaped_t = array_ops.reshape(
        t, array_ops.concat(
            ([self._batch_size, self._beam_width], t_shape[1:]), 0))
    static_batch_size = tensor_util.constant_value(self._batch_size)
    reshaped_t.set_shape(
        (tensor_shape.TensorShape([static_batch_size, self._beam_width])
         .concatenate(t_static_shape[1:])))
    return reshaped_t

  def _maybe_split_batch_beams(self, t):
    """Maybe splits the tensor from a batch by beams into a batch of beams.

    We do this so that we can use nest and not run into problems with shapes.

    Args:
      t: Tensor of dimension [batch_size*beam_width, ...]

    Returns:
      Either a reshaped version of t with dimension
      [batch_size, beam_width, ...] if t's first dimension is of size
      batch_size*beam_width or t if not.
    """
    t_shape = t.get_shape().as_list()
    if len(t_shape) >= 1:
      return self._split_batch_beams(t)
    else:
      return t

  def _maybe_merge_batch_beams(self, t):
    """Splits the tensor from a batch by beams into a batch of beams.

    More exactly, t is a tensor of dimension [batch_size*beam_width, ...]. We
    reshape this into [batch_size, beam_width, ...]

    Args:
      t: Tensor of dimension [batch_size*beam_width, ...]

    Returns:
      A reshaped version of t with dimension [batch_size, beam_width, ...].
    """
    t_shape = t.get_shape().as_list()
    if len(t_shape) >= 2:
      return self._merge_batch_beams(t)
    else:
      return t

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    batch_size = self._batch_size
    beam_width = self._beam_width
    end_token = self._end_token
    length_penalty_weight = self._length_penalty_weight

    with ops.name_scope(name, "BeamSearchDecoderStep", (time, inputs, state)):
      cell_state = state.cell_state
      inputs = nest.map_structure(self._merge_batch_beams, inputs)
      cell_state = nest.map_structure(self._maybe_merge_batch_beams, cell_state)
      try:
        cell_outputs, next_cell_state = self._cell(
            inputs, cell_state, tiling_factor=beam_width)
      except TypeError as e:
        if "unexpected keyword argument 'tiling_factor'" in str(e):
          cell_outputs, next_cell_state = self._cell(inputs, cell_state)
        else:
          raise

      cell_outputs = nest.map_structure(self._split_batch_beams, cell_outputs)
      next_cell_state = nest.map_structure(self._maybe_split_batch_beams,
                                           next_cell_state)

      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)

      beam_search_output, beam_search_state = _beam_search_step(
          time=time,
          logits=cell_outputs,
          beam_state=state,
          batch_size=batch_size,
          beam_width=beam_width,
          end_token=end_token,
          length_penalty_weight=length_penalty_weight)
      finished = beam_search_state.finished
      sample_ids = beam_search_output.predicted_ids
      next_inputs = control_flow_ops.cond(
          math_ops.reduce_all(finished), lambda: self._start_inputs,
          lambda: self._embedding_fn(sample_ids))

    return (beam_search_output, beam_search_state, next_inputs, finished)


def _beam_search_step(time, logits, beam_state, batch_size, beam_width,
                      end_token, length_penalty_weight):
  """Performs a single step of Beam Search Decoding.

  Args:
    time: Beam search time step, should start at 0. At time 0 we assume
      that all beams are equal and consider only the first beam for
      continuations.
    logits: Logits at the current time step. A tensor of shape `[B, vocab_size]`
    beam_state: Current state of the beam search. An instance of `BeamState`
    batch_size: The batch size for this input.
    beam_width: The size of the beams.
    end_token: The int32 end token.
    length_penalty_weight: Float weight to penalize length. Disabled with 0.0.

  Returns:
    A new beam state.
  """
  static_batch_size = tensor_util.constant_value(batch_size)

  # Calculate the current lengths of the predictions
  prediction_lengths = beam_state.lengths
  previously_finished = beam_state.finished

  # Calculate the total log probs for the new hypotheses
  # Final Shape: [batch_size, beam_width, vocab_size]
  probs = nn_ops.log_softmax(logits)
  probs = _mask_probs(probs, end_token, previously_finished)
  total_probs = array_ops.expand_dims(beam_state.log_probs, 2) + probs

  # Calculate the continuation lengths by adding to all continuing beams.
  vocab_size = logits.get_shape().as_list()[-1]
  lengths_to_add = array_ops.one_hot(
      array_ops.tile(
          array_ops.reshape(end_token, [1, 1]), [batch_size, beam_width]),
      vocab_size, 0, 1)
  add_mask = (1 - math_ops.to_int32(previously_finished))
  lengths_to_add = array_ops.expand_dims(add_mask, 2) * lengths_to_add
  new_prediction_lengths = array_ops.expand_dims(prediction_lengths,
                                                 2) + lengths_to_add

  # Calculate the scores for each beam
  scores = _get_scores(
      log_probs=total_probs,
      sequence_lengths=new_prediction_lengths,
      length_penalty_weight=length_penalty_weight)

  scores_flat = array_ops.reshape(scores, [batch_size, -1])
  # During the first time step we only consider the initial beam
  scores_flat = control_flow_ops.cond(
      ops.convert_to_tensor(time) > 0, lambda: scores_flat,
      lambda: scores[:, 0])

  # Pick the next beams according to the specified successors function
  next_beam_scores, word_indices = nn_ops.top_k(scores_flat, k=beam_width)
  next_beam_scores.set_shape([static_batch_size, beam_width])
  word_indices.set_shape([static_batch_size, beam_width])

  # Pick out the probs, beam_ids, and states according to the chosen predictions
  next_beam_probs = _tensor_gather_helper(
      gather_indices=word_indices,
      gather_from=total_probs,
      range_input=batch_size,
      range_size=beam_width * vocab_size,
      final_shape=[static_batch_size, beam_width])

  next_word_ids = math_ops.to_int32(word_indices % vocab_size)
  next_beam_ids = math_ops.to_int32(word_indices / vocab_size)

  # Append new ids to current predictions
  previously_finished = _tensor_gather_helper(
      gather_indices=next_beam_ids,
      gather_from=previously_finished,
      range_input=batch_size,
      range_size=beam_width,
      final_shape=[static_batch_size, beam_width])
  next_finished = math_ops.logical_or(previously_finished,
                                      math_ops.equal(next_word_ids, end_token))

  # Calculate the length of the next predictions.
  # 1. Finished beams remain unchanged
  # 2. Beams that are now finished (EOS predicted) remain unchanged
  # 3. Beams that are not yet finished have their length increased by 1
  lengths_to_add = math_ops.to_int32(
      math_ops.not_equal(next_word_ids, end_token))
  lengths_to_add = (1 - math_ops.to_int32(next_finished)) * lengths_to_add
  next_prediction_len = _tensor_gather_helper(
      gather_indices=next_beam_ids,
      gather_from=beam_state.lengths,
      range_input=batch_size,
      range_size=beam_width,
      final_shape=[static_batch_size, beam_width])
  next_prediction_len += lengths_to_add

  next_state = BeamSearchDecoderState(
      cell_state=beam_state.cell_state,
      log_probs=next_beam_probs,
      lengths=next_prediction_len,
      finished=next_finished)

  output = BeamSearchDecoderOutput(
      scores=next_beam_scores,
      predicted_ids=next_word_ids,
      parent_ids=next_beam_ids)

  return output, next_state


def _get_scores(log_probs, sequence_lengths, length_penalty_weight):
  """Calculates scores for beam search hypotheses.

  Args:
    log_probs: The log probabilities with shape [batch_size, beam_width].
    sequence_lengths: The array of sequence lengths.
    length_penalty_weight: Float weight to penalize length. Disabled with 0.0.

  Returns:
    The scores normalized by the length_penalty.
  """
  length_penality_ = _length_penalty(
      sequence_lengths=sequence_lengths, penalty_factor=length_penalty_weight)
  return log_probs / length_penality_


def _length_penalty(sequence_lengths, penalty_factor):
  """Calculates the length penalty. See https://arxiv.org/abs/1609.08144.

  Args:
    sequence_lengths: The sequence length of all hypotheses, a tensor
      of shape [beam_size, vocab_size].
    penalty_factor: A scalar that weights the length penalty.

  Returns:
    The length penalty factor, a tensor fo shape [beam_size].
  """
  # TODO(ebrevdo): cleanup based on constant-value of penalty_factor.
  return math_ops.div((5. + math_ops.to_float(sequence_lengths))
                      **penalty_factor, (5. + 1.)**penalty_factor)


def _mask_probs(probs, eos_token, finished):
  """Masks log probabilities.

  The result is that finished beams allocate all probability mass to eos and
  unfinished beams remain unchanged.

  Args:
    probs: Log probabiltiies of shape `[batch_size, beam_width, vocab_size]`
    eos_token: An int32 id corresponding to the EOS token to allocate
      probability to.
    finished: A boolean tensor of shape `[batch_size, beam_width]` that
      specifies which
      elements in the beam are finished already.

  Returns:
    A tensor of shape `[batch_size, beam_width, vocab_size]`, where unfinished
    beams stay unchanged and finished beams are replaced with a tensor with all
    probability on the EOS token.
  """
  vocab_size = array_ops.shape(probs)[2]
  finished_mask = array_ops.expand_dims(
      math_ops.to_float(1. - math_ops.to_float(finished)), 2)
  # These examples are not finished and we leave them
  non_finished_examples = finished_mask * probs
  # All finished examples are replaced with a vector that has all
  # probability on EOS
  finished_row = array_ops.one_hot(
      eos_token,
      vocab_size,
      dtype=dtypes.float32,
      on_value=0.,
      off_value=dtypes.float32.min)
  finished_examples = (1. - finished_mask) * finished_row
  return finished_examples + non_finished_examples


def _gather_tree_py(values, parents):
  """Gathers path through a tree backwards from the leave nodes.

  Used to reconstruct beams given their parents.

  Args:
    values: A [T, batch_size, beam_width] tensor of indices.
    parents: A [T, batch_size, beam_width] tensor of parent beam ids.

  Returns:
    The [T, batch_size, beam_width] numpy array of paths. For a given batch
      entry b, the best path is given by ret[:, b, 0].
  """
  num_timesteps = values.shape[0]
  num_beams = values.shape[2]
  batch_size = values.shape[1]
  ret = np.zeros_like(values)  # [T, MB, BW]
  ret[-1, :, :] = values[-1, :, :]
  for beam_id in range(num_beams):
    for batch in range(batch_size):
      parent = parents[-1][batch][beam_id]
      for timestep in reversed(range(num_timesteps - 1)):
        ret[timestep, batch, beam_id] = values[timestep][batch][parent]
        parent = parents[timestep][batch][parent]
  # now we are going to return ret as a [ts, mb, bw] tensor
  return np.array(ret).astype(values.dtype)


def _gather_tree(values, parents):
  """Tensor version of _gather_tree_py."""
  ret = script_ops.py_func(
      func=_gather_tree_py, inp=[values, parents], Tout=values.dtype)
  ret.set_shape(values.get_shape().as_list())
  return ret


def _tensor_gather_helper(gather_indices, gather_from, range_input, range_size,
                          final_shape):
  range_ = array_ops.expand_dims(math_ops.range(range_input) * range_size, 1)
  gather_indices = array_ops.reshape(gather_indices + range_, [-1])
  output = array_ops.gather(
      array_ops.reshape(gather_from, [-1]), gather_indices)
  output = array_ops.reshape(output, final_shape)
  output.set_shape(final_shape)
  return output
