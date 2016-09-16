Operator adding an output projection to the given cell.

Note: in many cases it may be more efficient to not use this wrapper,
but instead concatenate the whole sequence of your outputs in time,
do the projection on this batch-concatenated sequence, then split it
if needed or directly feed into a softmax.
- - -

#### `tf.nn.rnn_cell.OutputProjectionWrapper.__call__(inputs, state, scope=None)` {#OutputProjectionWrapper.__call__}

Run the cell and output projection on inputs, starting from state.


- - -

#### `tf.nn.rnn_cell.OutputProjectionWrapper.__init__(cell, output_size)` {#OutputProjectionWrapper.__init__}

Create a cell with output projection.

##### Args:


*  <b>`cell`</b>: an RNNCell, a projection to output_size is added to it.
*  <b>`output_size`</b>: integer, the size of the output after projection.

##### Raises:


*  <b>`TypeError`</b>: if cell is not an RNNCell.
*  <b>`ValueError`</b>: if output_size is not positive.


- - -

#### `tf.nn.rnn_cell.OutputProjectionWrapper.output_size` {#OutputProjectionWrapper.output_size}




- - -

#### `tf.nn.rnn_cell.OutputProjectionWrapper.state_size` {#OutputProjectionWrapper.state_size}




- - -

#### `tf.nn.rnn_cell.OutputProjectionWrapper.zero_state(batch_size, dtype)` {#OutputProjectionWrapper.zero_state}

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


