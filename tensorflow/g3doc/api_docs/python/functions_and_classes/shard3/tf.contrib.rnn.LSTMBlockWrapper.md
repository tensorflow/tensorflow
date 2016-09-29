This is a helper class that provides housekeeping for LSTM cells.

This may be useful for alternative LSTM and similar type of cells.
The subclasses must implement `_call_cell` method and `num_units` property.
- - -

#### `tf.contrib.rnn.LSTMBlockWrapper.__call__(inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)` {#LSTMBlockWrapper.__call__}

Run this LSTM on inputs, starting from the given state.

##### Args:


*  <b>`inputs`</b>: `3-D` tensor with shape `[time_len x batch_size x input_size]`
    or a list of `time_len` tensors of shape `[batch_size x input_size]`.
*  <b>`initial_state`</b>: a tuple `(initial_cell_state, initial_output)` with tensors
    of shape `[batch_size, self._num_units]`. If this is not provided, the
    cell is expected to create a zero initial state of type `dtype`.
*  <b>`dtype`</b>: The data type for the initial state and expected output. Required
    if `initial_state` is not provided or RNN state has a heterogeneous
    dtype.
*  <b>`sequence_length`</b>: Specifies the length of each sequence in inputs. An int32
    or int64 vector (tensor) size [batch_size], values in [0, time_len).
    Defaults to `time_len` for each element.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to class name.

##### Returns:

  A pair containing:
  - Output: A `3-D` tensor of shape `[time_len x batch_size x output_size]`
    or a list of time_len tensors of shape `[batch_size x output_size]`, to
    match the type of the `inputs`.
  - Final state: a tuple `(cell_state, output)` matching initial_state.

##### Raises:


*  <b>`ValueError`</b>: in case of shape mismatches


- - -

#### `tf.contrib.rnn.LSTMBlockWrapper.num_units` {#LSTMBlockWrapper.num_units}

Number of units in this cell (output dimension).


