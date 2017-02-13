Operator that ensures an RNNCell runs on a particular device.
- - -

#### `tf.contrib.rnn.DeviceWrapper.__call__(inputs, state, scope=None)` {#DeviceWrapper.__call__}

Run the cell on specified device.


- - -

#### `tf.contrib.rnn.DeviceWrapper.__init__(cell, device)` {#DeviceWrapper.__init__}

Construct a `DeviceWrapper` for `cell` with device `device`.

Ensures the wrapped `cell` is called with `tf.device(device)`.

##### Args:


*  <b>`cell`</b>: An instance of `RNNCell`.
*  <b>`device`</b>: A device string or function, for passing to `tf.device`.


- - -

#### `tf.contrib.rnn.DeviceWrapper.output_size` {#DeviceWrapper.output_size}

Integer or TensorShape: size of outputs produced by this cell.


- - -

#### `tf.contrib.rnn.DeviceWrapper.state_size` {#DeviceWrapper.state_size}

size(s) of state(s) used by this cell.

It can be represented by an Integer, a TensorShape or a tuple of Integers
or TensorShapes.


- - -

#### `tf.contrib.rnn.DeviceWrapper.zero_state(batch_size, dtype)` {#DeviceWrapper.zero_state}

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


