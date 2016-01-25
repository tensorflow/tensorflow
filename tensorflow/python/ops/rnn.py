# Copyright 2015 Google Inc. All Rights Reserved.
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

"""RNN helpers for TensorFlow models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs


def rnn(cell, inputs, initial_state=None, dtype=None,
        sequence_length=None, scope=None):
  """Creates a recurrent neural network specified by RNNCell "cell".

  The simplest form of RNN network generated is:
    state = cell.zero_state(...)
    outputs = []
    states = []
    for input_ in inputs:
      output, state = cell(input_, state)
      outputs.append(output)
      states.append(state)
    return (outputs, states)

  However, a few other options are available:

  An initial state can be provided.
  If sequence_length is provided, dynamic calculation is performed.

  Dynamic calculation returns, at time t:
    (t >= max(sequence_length)
        ? (zeros(output_shape), zeros(state_shape))
        : cell(input, state)

  Thus saving computational time when unrolling past the max sequence length.

  Args:
    cell: An instance of RNNCell.
    inputs: A length T list of inputs, each a tensor of shape
      [batch_size, cell.input_size].
    initial_state: (optional) An initial state for the RNN.  This must be
      a tensor of appropriate type and shape [batch_size x cell.state_size].
    dtype: (optional) The data type for the initial state.  Required if
      initial_state is not provided.
    sequence_length: An int64 vector (tensor) size [batch_size].
    scope: VariableScope for the created subgraph; defaults to "RNN".

  Returns:
    A pair (outputs, states) where:
      outputs is a length T list of outputs (one for each input)
      states is a length T list of states (one state following each input)

  Raises:
    TypeError: If "cell" is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """

  if not isinstance(cell, rnn_cell.RNNCell):
    raise TypeError("cell must be an instance of RNNCell")
  if not isinstance(inputs, list):
    raise TypeError("inputs must be a list")
  if not inputs:
    raise ValueError("inputs must not be empty")

  outputs = []
  states = []
  with vs.variable_scope(scope or "RNN"):
    batch_size = array_ops.shape(inputs[0])[0]
    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If no initial_state is provided, dtype must be.")
      state = cell.zero_state(batch_size, dtype)

    if sequence_length:  # Prepare variables
      zero_output_state = (
          array_ops.zeros(array_ops.pack([batch_size, cell.output_size]),
                          inputs[0].dtype),
          array_ops.zeros(array_ops.pack([batch_size, cell.state_size]),
                          state.dtype))
      max_sequence_length = math_ops.reduce_max(sequence_length)

    for time, input_ in enumerate(inputs):
      if time > 0: vs.get_variable_scope().reuse_variables()
      # pylint: disable=cell-var-from-loop
      def output_state():
        return cell(input_, state)
      # pylint: enable=cell-var-from-loop
      if sequence_length:
        (output, state) = control_flow_ops.cond(
            time >= max_sequence_length,
            lambda: zero_output_state, output_state)
      else:
        (output, state) = output_state()

      outputs.append(output)
      states.append(state)

    return (outputs, states)


def state_saving_rnn(cell, inputs, state_saver, state_name,
                     sequence_length=None, scope=None):
  """RNN that accepts a state saver for time-truncated RNN calculation.

  Args:
    cell: An instance of RNNCell.
    inputs: A length T list of inputs, each a tensor of shape
      [batch_size, cell.input_size].
    state_saver: A state saver object with methods `state` and `save_state`.
    state_name: The name to use with the state_saver.
    sequence_length: (optional) An int64 vector (tensor) size [batch_size].
      See the documentation for rnn() for more details about sequence_length.
    scope: VariableScope for the created subgraph; defaults to "RNN".

  Returns:
    A pair (outputs, states) where:
      outputs is a length T list of outputs (one for each input)
      states is a length T list of states (one state following each input)

  Raises:
    TypeError: If "cell" is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """
  initial_state = state_saver.state(state_name)
  (outputs, states) = rnn(cell, inputs, initial_state=initial_state,
                          sequence_length=sequence_length, scope=scope)
  save_state = state_saver.save_state(state_name, states[-1])
  with ops.control_dependencies([save_state]):
    outputs[-1] = array_ops.identity(outputs[-1])

  return (outputs, states)


def _reverse_seq(input_seq, lengths):
  """Reverse a list of Tensors up to specified lengths.

  Args:
    input_seq: Sequence of seq_len tensors of dimension (batch_size, depth)
    lengths:   A tensor of dimension batch_size, containing lengths for each
               sequence in the batch. If "None" is specified, simply reverses
               the list.

  Returns:
    time-reversed sequence
  """
  if lengths is None:
    return list(reversed(input_seq))

  for input_ in input_seq:
    input_.set_shape(input_.get_shape().with_rank(2))

  # Join into (time, batch_size, depth)
  s_joined = array_ops.pack(input_seq)

  # Reverse along dimension 0
  s_reversed = array_ops.reverse_sequence(s_joined, lengths, 0, 1)
  # Split again into list
  result = array_ops.unpack(s_reversed)
  return result


def bidirectional_rnn(cell_fw, cell_bw, inputs,
                      initial_state_fw=None, initial_state_bw=None,
                      dtype=None, sequence_length=None, scope=None):
  """Creates a bidirectional recurrent neural network.

  Similar to the unidirectional case above (rnn) but takes input and builds
  independent forward and backward RNNs with the final forward and backward
  outputs depth-concatenated, such that the output will have the format
  [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
  forward and backward cell must match. The initial state for both directions
  is zero by default (but can be set optionally) and no intermediate states are
  ever returned -- the network is fully unrolled for the given (passed in)
  length(s) of the sequence(s) or completely unrolled if length(s) is not given.

  Args:
    cell_fw: An instance of RNNCell, to be used for forward direction.
    cell_bw: An instance of RNNCell, to be used for backward direction.
    inputs: A length T list of inputs, each a tensor of shape
      [batch_size, cell.input_size].
    initial_state_fw: (optional) An initial state for the forward RNN.
      This must be a tensor of appropriate type and shape
      [batch_size x cell.state_size].
    initial_state_bw: (optional) Same as for initial_state_fw.
    dtype: (optional) The data type for the initial state.  Required if either
      of the initial states are not provided.
    sequence_length: (optional) An int64 vector (tensor) of size [batch_size],
      containing the actual lengths for each of the sequences.
    scope: VariableScope for the created subgraph; defaults to "BiRNN"

  Returns:
    A set of output `Tensors` where:
      outputs is a length T list of outputs (one for each input), which
      are depth-concatenated forward and backward outputs

  Raises:
    TypeError: If "cell_fw" or "cell_bw" is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """

  if not isinstance(cell_fw, rnn_cell.RNNCell):
    raise TypeError("cell_fw must be an instance of RNNCell")
  if not isinstance(cell_bw, rnn_cell.RNNCell):
    raise TypeError("cell_bw must be an instance of RNNCell")
  if not isinstance(inputs, list):
    raise TypeError("inputs must be a list")
  if not inputs:
    raise ValueError("inputs must not be empty")

  name = scope or "BiRNN"
  # Forward direction
  with vs.variable_scope(name + "_FW"):
    output_fw, _ = rnn(cell_fw, inputs, initial_state_fw, dtype,
                       sequence_length)

  # Backward direction
  with vs.variable_scope(name + "_BW"):
    tmp, _ = rnn(cell_bw, _reverse_seq(inputs, sequence_length),
                 initial_state_bw, dtype, sequence_length)
  output_bw = _reverse_seq(tmp, sequence_length)
  # Concat each of the forward/backward outputs
  outputs = [array_ops.concat(1, [fw, bw])
             for fw, bw in zip(output_fw, output_bw)]

  return outputs
