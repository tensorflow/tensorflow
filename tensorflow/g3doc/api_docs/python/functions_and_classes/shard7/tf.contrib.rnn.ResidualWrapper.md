RNNCell wrapper that ensures cell inputs are added to the outputs.
- - -

#### `tf.contrib.rnn.ResidualWrapper.__call__(inputs, state, scope=None)` {#ResidualWrapper.__call__}

Run the cell and add its inputs to its outputs.

##### Args:


*  <b>`inputs`</b>: cell inputs.
*  <b>`state`</b>: cell state.
*  <b>`scope`</b>: optional cell scope.

##### Returns:

  Tuple of cell outputs and new state.

##### Raises:


*  <b>`TypeError`</b>: If cell inputs and outputs have different structure (type).
*  <b>`ValueError`</b>: If cell inputs and outputs have different structure (value).


- - -

#### `tf.contrib.rnn.ResidualWrapper.__init__(cell)` {#ResidualWrapper.__init__}

Constructs a `ResidualWrapper` for `cell`.

##### Args:


*  <b>`cell`</b>: An instance of `RNNCell`.


- - -

#### `tf.contrib.rnn.ResidualWrapper.output_size` {#ResidualWrapper.output_size}




- - -

#### `tf.contrib.rnn.ResidualWrapper.state_size` {#ResidualWrapper.state_size}




- - -

#### `tf.contrib.rnn.ResidualWrapper.zero_state(batch_size, dtype)` {#ResidualWrapper.zero_state}

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


