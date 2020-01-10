"""RNN helpers for TensorFlow models."""

import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.python.ops import control_flow_ops


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
    inputs: A length T list of inputs, each a vector with shape [batch_size].
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
  with tf.variable_scope(scope or "RNN"):
    batch_size = tf.shape(inputs[0])[0]
    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If no initial_state is provided, dtype must be.")
      state = cell.zero_state(batch_size, dtype)

    if sequence_length:  # Prepare variables
      zero_output_state = (
          tf.zeros(tf.pack([batch_size, cell.output_size]),
                   inputs[0].dtype),
          tf.zeros(tf.pack([batch_size, cell.state_size]),
                   state.dtype))
      max_sequence_length = tf.reduce_max(sequence_length)

    output_state = (None, None)
    for time, input_ in enumerate(inputs):
      if time > 0:
        tf.get_variable_scope().reuse_variables()
      output_state = cell(input_, state)
      if sequence_length:
        (output, state) = control_flow_ops.cond(
            time >= max_sequence_length,
            lambda: zero_output_state, lambda: output_state)
      else:
        (output, state) = output_state

      outputs.append(output)
      states.append(state)

    return (outputs, states)


def state_saving_rnn(cell, inputs, state_saver, state_name,
                     sequence_length=None, scope=None):
  """RNN that accepts a state saver for time-truncated RNN calculation.

  Args:
    cell: An instance of RNNCell.
    inputs: A length T list of inputs, each a vector with shape [batch_size].
    state_saver: A StateSaver object.
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
  initial_state = state_saver.State(state_name)
  (outputs, states) = rnn(cell, inputs, initial_state=initial_state,
                          sequence_length=sequence_length, scope=scope)
  save_state = state_saver.SaveState(state_name, states[-1])
  with tf.control_dependencies([save_state]):
    outputs[-1] = tf.identity(outputs[-1])

  return (outputs, states)
