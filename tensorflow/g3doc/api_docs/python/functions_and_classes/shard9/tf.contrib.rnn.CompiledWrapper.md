Wraps step execution in an XLA JIT scope.
- - -

#### `tf.contrib.rnn.CompiledWrapper.__call__(inputs, state, scope=None)` {#CompiledWrapper.__call__}




- - -

#### `tf.contrib.rnn.CompiledWrapper.__init__(cell, compile_stateful=False)` {#CompiledWrapper.__init__}

Create CompiledWrapper cell.

##### Args:


*  <b>`cell`</b>: Instance of `RNNCell`.
*  <b>`compile_stateful`</b>: Whether to compile stateful ops like initializers
    and random number generators (default: False).


- - -

#### `tf.contrib.rnn.CompiledWrapper.output_size` {#CompiledWrapper.output_size}




- - -

#### `tf.contrib.rnn.CompiledWrapper.state_size` {#CompiledWrapper.state_size}




- - -

#### `tf.contrib.rnn.CompiledWrapper.zero_state(batch_size, dtype)` {#CompiledWrapper.zero_state}

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


