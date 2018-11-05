from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.poplar import popnn_ops_grad


def basic_lstm_layer(inputs, num_channels, weights, initial_state=None,
               is_training=False, partials_dtype=dtypes.float32, name=None):
  """Runs the forward step of a Basic LSTM layer.

  Args:
    :param inputs `3-D` tensor with shape `[seq_len, batch_size, input_size]`.
    :param num_channels The output size of the LSTM layer.
    :param weights A 3-tuple of tensor of shapes
      `([4, input_size, num_channels], [4, num_channels, num_channels], [4, num_channels])`.
      This represents the input weights, output weights and biases for a Basic
      LSTM respectively. The order of gates is Forget, Input, Candidate and
      Output gate.
    :param initial_state If provided, a 2-tuple of tensor of shapes
      `([batch_size, num_channels], [batch_size, num_channels])`. This represents
      the initial hidden and cell state respectively. If not provided the
      initial state is all initialised to 0.
    :param is_training whether this operation will be used in is_training or
      inference.
    :param partials_dtype The type which will be used for intermediate
                          calculations. Defaults to float32.
    :param name Optional name for this operation.
  Returns:
    :return: tuple (output, output_state)
      WHERE
      str output is a tensor of shape `[seq_len, batch_size, num_channels]`.
      str output_states is a tuple of tensor of the same shape and structure as
        `initial_state`.
  Raises:
    ValueError: weights or initial_state are not a correct tuple size.
  """

  # Check the weights are passed in correctly.
  if not isinstance(weights, tuple) or len(weights) != 3:
    raise ValueError(
        'Weights needs to be a 3-tuple of tensor (input_weights, output_weights, biases).')
  input_weights, output_weights, biases = weights

  if initial_state is not None:
    # Check the inital state is passed in correctly.
    if not isinstance(initial_state, tuple) or len(initial_state) != 2:
      raise ValueError(
          'Initial state needs to be a 2-tuple of tensor (initial_hidden_state, initial_cell_state).')
    init_h_state, init_c_state = initial_state
  else:
    # Create a new state with all 0's.
    # Each state shape is [batch_size, num_channels].
    batch_size = inputs.get_shape().dims[0]
    shape = [inputs.get_shape().dims[0], num_channels]
    init_h_state = array_ops.zeros(
        shape, dtype=inputs.dtype, name=str(name or '') + "init_h_state")
    init_c_state = array_ops.zeros(
        shape, dtype=inputs.dtype, name=str(name or '') + "init_c_state")

  if partials_dtype is None:
    partials_dtype = inputs.dtype

  outs, out_h_state, out_c_state, _ = gen_popnn_ops.popnn_lstm_layer(inputs,
                                                                     init_h_state,
                                                                     init_c_state,
                                                                     input_weights,
                                                                     output_weights,
                                                                     biases,
                                                                     num_channels=num_channels,
                                                                     is_training=is_training,
                                                                     partials_dtype=partials_dtype,
                                                                     name=name)

  return outs, (out_h_state, out_c_state)
