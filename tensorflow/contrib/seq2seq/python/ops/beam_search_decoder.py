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
"""A decoder that performs beam search.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.contrib.rnn import core_rnn_cell
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest


__all__ = [
    "BeamSearchDecoderOutput",
    "BeamSearchDecoderState",
    "BeamSearchDecoder",
]


class BeamSearchDecoderOutput(
    collections.namedtuple("BeamSearchDecoderOutput", ("rnn_output",))):
  pass


class BeamSearchDecoderState(
    collections.namedtuple("BeamSearchDecoderState",
                           ("cell_state", "log_prob", "beam_ids"))):
  pass


class BeamSearchDecoder(decoder.Decoder):
  """BeamSearch sampling decoder."""

  def __init__(self, cell, embedding, start_tokens, end_token,
               initial_state, beam_width, output_layer=None):
    """Initialize BeamSearchDecoder.

    Args:
      cell: An `RNNCell` instance.
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      end_token: `int32` scalar, the token that marks end of decoding.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
      beam_width:  Python integer, the number of beams
      output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`.  Optional layer to apply to the RNN output prior
        to storing the result or sampling.

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
    self._initial_cell_state = initial_state
    self._output_layer = output_layer

    if callable(embedding):
      self._embedding_fn = embedding
    else:
      self._embedding_fn = (
          lambda ids: embedding_ops.embedding_lookup(embedding, ids))

    self._start_tokens = ops.convert_to_tensor(
        start_tokens, dtype=dtypes.int32, name="start_tokens")
    self._end_token = ops.convert_to_tensor(
        end_token, dtype=dtypes.int32, name="end_token")
    if self._start_tokens.get_shape().ndims != 1:
      raise ValueError("start_tokens must be a vector")
    self._batch_size = array_ops.size(start_tokens)
    self._beam_width = beam_width
    if self._end_token.get_shape().ndims != 0:
      raise ValueError("end_token must be a scalar")
    self._start_inputs = self._embedding_fn(self._start_tokens)

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
    prepend_beam_width = (
        lambda s: tensor_shape.TensorShape([self._beam_width]).concatenate(s))
    return BeamSearchDecoderOutput(
        rnn_output=nest.map_structure(
            prepend_beam_width, self._rnn_output_size()))

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and int32 (the id)
    dtype = nest.flatten(self._initial_cell_state)[0].dtype
    return BeamSearchDecoderOutput(
        rnn_output=nest.map_structure(lambda _: dtype, self._rnn_output_size()))

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    finished, first_inputs = self._finished, self._first_inputs

    initial_state = BeamSearchDecoderState(
        cell_state=self._initial_cell_state,
        log_probs=array_ops.zeros(
            [self.batch_size, self.beam_width],
            dtype=nest.flatten(self._initial_cell_state)[0].dtype),
        beam_ids=tensor_array_ops.TensorArray(
            size=0, dynamic_size=True, dtype=dtypes.int32,
            clear_after_read=False))

    return (finished, first_inputs, initial_state)

  def _merge_batch_beams(self, t):
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
    with ops.name_scope(name, "BeamSearchDecoderStep", (time, inputs, state)):
      cell_state = state.cell_state
      inputs = nest.map_structure(self._merge_batch_beams, inputs)
      cell_state = nest.map_structure(self._merge_batch_beams, cell_state)
      cell_outputs, next_cell_state = self._cell(inputs, cell_state)
      cell_outputs = nest.map_structure(self._split_batch_beams, cell_outputs)
      next_cell_state = nest.map_structure(self._split_batch_beams,
                                           next_cell_state)

      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)

      # TODO(cinjon): Calculate next_log_probs, next_beam_ids,
      #   finished, next_inputs, final_cell_state via beam search
      #   via self._embedding
      # ....
      next_beam_ids, next_log_probs, final_cell_state, next_inputs, finished = (
          None, None, None, None, None)

      beam_ids = state.beam_ids.write(time, next_beam_ids)

      outputs = BeamSearchDecoderOutput(cell_outputs)
      next_state = BeamSearchDecoderState(
          log_probs=next_log_probs,
          beam_ids=beam_ids,
          cell_state=final_cell_state)

    return (outputs, next_state, next_inputs, finished)
