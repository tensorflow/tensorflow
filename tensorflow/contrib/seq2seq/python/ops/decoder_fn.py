# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Seq2seq loss operations for use in neural networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.util import nest

__all__ = ["simple_decoder_fn_train",
           "simple_decoder_fn_inference"]

def simple_decoder_fn_train(encoder_state, name=None):
  with ops.name_scope(name, "simple_decoder_fn_train", [encoder_state]):
    if type(encoder_state) is not rnn_cell.LSTMStateTuple:
      encoder_state = ops.convert_to_tensor(encoder_state)

  def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
    with ops.name_scope(name, "simple_decoder_fn_train",
                        [time, cell_state, cell_input, cell_output,
                         context_state]):
      if cell_state is None:  # first call, return encoder_state
        return (None, encoder_state, cell_input, cell_output, context_state)
      else:
        return (None, cell_state, cell_input, cell_output, context_state)
  return decoder_fn


def simple_decoder_fn_inference(output_fn, encoder_state, embeddings,
                                start_of_sequence_id, end_of_sequence_id,
                                maximum_length, num_decoder_symbols,
                                dtype=dtypes.int32, name=None):
  with ops.name_scope(name, "simple_decoder_fn_inference",
                      [output_fn, encoder_state, embeddings,
                       start_of_sequence_id, end_of_sequence_id,
                       maximum_length, num_decoder_symbols, dtype]):
    if type(encoder_state) is not rnn_cell.LSTMStateTuple:
      encoder_state = ops.convert_to_tensor(encoder_state)
    start_of_sequence_id = ops.convert_to_tensor(start_of_sequence_id, dtype)
    end_of_sequence_id = ops.convert_to_tensor(end_of_sequence_id, dtype)
    maximum_length = ops.convert_to_tensor(maximum_length, dtype)
    num_decoder_symbols = ops.convert_to_tensor(num_decoder_symbols, dtype)
    encoder_info = nest.flatten(encoder_state)[0]
    batch_size = encoder_info.get_shape()[0].value
    if output_fn is None:
      output_fn = lambda x: x
    if batch_size is None:
      batch_size = array_ops.shape(encoder_info)[0]

  def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
    with ops.name_scope(name, "simple_decoder_fn_inference",
                        [time, cell_state, cell_input, cell_output,
                         context_state]):
      if cell_input is not None:
        raise ValueError("Expected cell_input to be None, but saw: %s" %
                         cell_input)
      if cell_output is None:
        # invariant that this is time == 0
        next_input_id = array_ops.ones([batch_size,], dtype=dtype) * (
            start_of_sequence_id)
        done = array_ops.zeros([batch_size,], dtype=dtypes.bool)
        cell_state = encoder_state
        cell_output = array_ops.zeros([num_decoder_symbols],
                                      dtype=dtypes.float32)
      else:
        cell_output = output_fn(cell_output)
        next_input_id = math_ops.cast(
            math_ops.argmax(cell_output, 1), dtype=dtype)
        done = math_ops.equal(next_input_id, end_of_sequence_id)
      next_input = array_ops.gather(embeddings, next_input_id)
      # if time > maxlen, return all true vector
      done = control_flow_ops.cond(math_ops.greater(time, maximum_length),
          lambda: array_ops.ones([batch_size,], dtype=dtypes.bool),
          lambda: done)
      return (done, cell_state, next_input, cell_output, context_state)
  return decoder_fn
