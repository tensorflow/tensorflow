FusedRNNCell implementation of LSTM.

This is an extremely efficient LSTM implementation, that uses a single TF op
for the entire LSTM. It should be both faster and more memory-efficient than
LSTMBlockCell defined above.

The implementation is based on: http://arxiv.org/abs/1409.2329.

We add forget_bias (default: 1) to the biases of the forget gate in order to
reduce the scale of forgetting in the beginning of the training.

The variable naming is consistent with rnn_cell.LSTMCell.
- - -

#### `tf.contrib.rnn.LSTMBlockFusedCell.__call__(inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)` {#LSTMBlockFusedCell.__call__}

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

#### `tf.contrib.rnn.LSTMBlockFusedCell.__init__(num_units, forget_bias=1.0, cell_clip=None, use_peephole=False)` {#LSTMBlockFusedCell.__init__}

Initialize the LSTM cell.

##### Args:


*  <b>`num_units`</b>: int, The number of units in the LSTM cell.
*  <b>`forget_bias`</b>: float, The bias added to forget gates (see above).
*  <b>`cell_clip`</b>: clip the cell to this value. Defaults to `3`.
*  <b>`use_peephole`</b>: Whether to use peephole connections or not.


- - -

#### `tf.contrib.rnn.LSTMBlockFusedCell.num_units` {#LSTMBlockFusedCell.num_units}

Number of units in this cell (output dimension).


