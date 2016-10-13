Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
- - -

#### `tf.nn.rnn_cell.GRUCell.__call__(inputs, state, scope=None)` {#GRUCell.__call__}

Gated recurrent unit (GRU) with nunits cells.


- - -

#### `tf.nn.rnn_cell.GRUCell.__init__(num_units, input_size=None, activation=tanh)` {#GRUCell.__init__}




- - -

#### `tf.nn.rnn_cell.GRUCell.output_size` {#GRUCell.output_size}




- - -

#### `tf.nn.rnn_cell.GRUCell.state_size` {#GRUCell.state_size}




- - -

#### `tf.nn.rnn_cell.GRUCell.zero_state(batch_size, dtype)` {#GRUCell.zero_state}

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


