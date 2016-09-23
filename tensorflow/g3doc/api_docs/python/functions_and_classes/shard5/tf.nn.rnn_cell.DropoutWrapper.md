Operator adding dropout to inputs and outputs of the given cell.
- - -

#### `tf.nn.rnn_cell.DropoutWrapper.__call__(inputs, state, scope=None)` {#DropoutWrapper.__call__}

Run the cell with the declared dropouts.


- - -

#### `tf.nn.rnn_cell.DropoutWrapper.__init__(cell, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)` {#DropoutWrapper.__init__}

Create a cell with added input and/or output dropout.

Dropout is never used on the state.

##### Args:


*  <b>`cell`</b>: an RNNCell, a projection to output_size is added to it.
*  <b>`input_keep_prob`</b>: unit Tensor or float between 0 and 1, input keep
    probability; if it is float and 1, no input dropout will be added.
*  <b>`output_keep_prob`</b>: unit Tensor or float between 0 and 1, output keep
    probability; if it is float and 1, no output dropout will be added.
*  <b>`seed`</b>: (optional) integer, the randomness seed.

##### Raises:


*  <b>`TypeError`</b>: if cell is not an RNNCell.
*  <b>`ValueError`</b>: if keep_prob is not between 0 and 1.


- - -

#### `tf.nn.rnn_cell.DropoutWrapper.output_size` {#DropoutWrapper.output_size}




- - -

#### `tf.nn.rnn_cell.DropoutWrapper.state_size` {#DropoutWrapper.state_size}




- - -

#### `tf.nn.rnn_cell.DropoutWrapper.zero_state(batch_size, dtype)` {#DropoutWrapper.zero_state}

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


