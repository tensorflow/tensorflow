Basic LSTM recurrent network cell.

The implementation is based on: http://arxiv.org/abs/1409.2329.

We add forget_bias (default: 1) to the biases of the forget gate in order to
reduce the scale of forgetting in the beginning of the training.

Unlike rnn_cell.LSTMCell, this is a monolithic op and should be much faster.
The weight and bias matrixes should be compatible as long as the variable
scope matches, and you use `use_compatible_names=True`.
- - -

#### `tf.contrib.rnn.LSTMBlockCell.__call__(x, states_prev, scope=None)` {#LSTMBlockCell.__call__}

Long short-term memory cell (LSTM).


- - -

#### `tf.contrib.rnn.LSTMBlockCell.__init__(num_units, forget_bias=1.0, use_peephole=False, use_compatible_names=False)` {#LSTMBlockCell.__init__}

Initialize the basic LSTM cell.

##### Args:


*  <b>`num_units`</b>: int, The number of units in the LSTM cell.
*  <b>`forget_bias`</b>: float, The bias added to forget gates (see above).
*  <b>`use_peephole`</b>: Whether to use peephole connections or not.
*  <b>`use_compatible_names`</b>: If True, use the same variable naming as
    rnn_cell.LSTMCell


- - -

#### `tf.contrib.rnn.LSTMBlockCell.output_size` {#LSTMBlockCell.output_size}




- - -

#### `tf.contrib.rnn.LSTMBlockCell.state_size` {#LSTMBlockCell.state_size}




- - -

#### `tf.contrib.rnn.LSTMBlockCell.zero_state(batch_size, dtype)` {#LSTMBlockCell.zero_state}

Return zero-filled state tensor(s).

##### Args:


*  <b>`batch_size`</b>: int, float, or unit Tensor representing the batch size.
*  <b>`dtype`</b>: the data type to use for the state.

##### Returns:

  If `state_size` is an int or TensorShape, then the return value is a
  `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.

  If `state_size` is a nested list or tuple, then the return value is
  a nested list or tuple (of the same structure) of `2-D` tensors with
the shapes `[batch_size x s]` for each s in `state_size`.


