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

"""TensorFlow Ops for Sequence to Sequence models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import rnn
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope as vs


def sequence_classifier(decoding, labels, sampling_decoding=None, name=None):
  """Returns predictions and loss for sequence of predictions.

  Args:
    decoding: List of Tensors with predictions.
    labels: List of Tensors with labels.
    sampling_decoding: Optional, List of Tensor with predictions to be used
      in sampling. E.g. they shouldn't have dependncy on outputs.
      If not provided, decoding is used.
    name: Operation name.

  Returns:
    Predictions and losses tensors.
  """
  with ops.name_scope(name, "sequence_classifier", [decoding, labels]):
    predictions, xent_list = [], []
    for i, pred in enumerate(decoding):
      xent_list.append(nn.softmax_cross_entropy_with_logits(
          labels=labels[i], logits=pred,
          name="sequence_loss/xent_raw{0}".format(i)))
      if sampling_decoding:
        predictions.append(nn.softmax(sampling_decoding[i]))
      else:
        predictions.append(nn.softmax(pred))
    xent = math_ops.add_n(xent_list, name="sequence_loss/xent")
    loss = math_ops.reduce_sum(xent, name="sequence_loss")
    return array_ops.stack(predictions, axis=1), loss


def seq2seq_inputs(x, y, input_length, output_length, sentinel=None, name=None):
  """Processes inputs for Sequence to Sequence models.

  Args:
    x: Input Tensor [batch_size, input_length, embed_dim].
    y: Output Tensor [batch_size, output_length, embed_dim].
    input_length: length of input x.
    output_length: length of output y.
    sentinel: optional first input to decoder and final output expected.
      If sentinel is not provided, zeros are used. Due to fact that y is not
      available in sampling time, shape of sentinel will be inferred from x.
    name: Operation name.

  Returns:
    Encoder input from x, and decoder inputs and outputs from y.
  """
  with ops.name_scope(name, "seq2seq_inputs", [x, y]):
    in_x = array_ops.unstack(x, axis=1)
    y = array_ops.unstack(y, axis=1)
    if not sentinel:
      # Set to zeros of shape of y[0], using x for batch size.
      sentinel_shape = array_ops.stack(
          [array_ops.shape(x)[0], y[0].get_shape()[1]])
      sentinel = array_ops.zeros(sentinel_shape)
      sentinel.set_shape(y[0].get_shape())
    in_y = [sentinel] + y
    out_y = y + [sentinel]
    return in_x, in_y, out_y


def rnn_decoder(decoder_inputs, initial_state, cell, scope=None):
  """RNN Decoder that creates training and sampling sub-graphs.

  Args:
    decoder_inputs: Inputs for decoder, list of tensors.
      This is used only in training sub-graph.
    initial_state: Initial state for the decoder.
    cell: RNN cell to use for decoder.
    scope: Scope to use, if None new will be produced.

  Returns:
    List of tensors for outputs and states for training and sampling sub-graphs.
  """
  with vs.variable_scope(scope or "dnn_decoder"):
    states, sampling_states = [initial_state], [initial_state]
    outputs, sampling_outputs = [], []
    with ops.name_scope("training", values=[decoder_inputs, initial_state]):
      for i, inp in enumerate(decoder_inputs):
        if i > 0:
          vs.get_variable_scope().reuse_variables()
        output, new_state = cell(inp, states[-1])
        outputs.append(output)
        states.append(new_state)
    with ops.name_scope("sampling", values=[initial_state]):
      for i, _ in enumerate(decoder_inputs):
        if i == 0:
          sampling_outputs.append(outputs[i])
          sampling_states.append(states[i])
        else:
          sampling_output, sampling_state = cell(sampling_outputs[-1],
                                                 sampling_states[-1])
          sampling_outputs.append(sampling_output)
          sampling_states.append(sampling_state)
  return outputs, states, sampling_outputs, sampling_states


def rnn_seq2seq(encoder_inputs,
                decoder_inputs,
                encoder_cell,
                decoder_cell=None,
                dtype=dtypes.float32,
                scope=None):
  """RNN Sequence to Sequence model.

  Args:
    encoder_inputs: List of tensors, inputs for encoder.
    decoder_inputs: List of tensors, inputs for decoder.
    encoder_cell: RNN cell to use for encoder.
    decoder_cell: RNN cell to use for decoder, if None encoder_cell is used.
    dtype: Type to initialize encoder state with.
    scope: Scope to use, if None new will be produced.

  Returns:
    List of tensors for outputs and states for training and sampling sub-graphs.
  """
  with vs.variable_scope(scope or "rnn_seq2seq"):
    _, last_enc_state = rnn.static_rnn(
        encoder_cell, encoder_inputs, dtype=dtype)
    return rnn_decoder(decoder_inputs, last_enc_state, decoder_cell or
                       encoder_cell)
