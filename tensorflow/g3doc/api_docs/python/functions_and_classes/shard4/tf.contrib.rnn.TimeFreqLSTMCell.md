Time-Frequency Long short-term memory unit (LSTM) recurrent network cell.

This implementation is based on:

  Tara N. Sainath and Bo Li
  "Modeling Time-Frequency Patterns with LSTM vs. Convolutional Architectures
  for LVCSR Tasks." submitted to INTERSPEECH, 2016.

It uses peep-hole connections and optional cell clipping.
- - -

#### `tf.contrib.rnn.TimeFreqLSTMCell.__call__(inputs, state, scope=None)` {#TimeFreqLSTMCell.__call__}

Run one step of LSTM.

##### Args:


*  <b>`inputs`</b>: input Tensor, 2D, batch x num_units.
*  <b>`state`</b>: state Tensor, 2D, batch x state_size.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to
    "TimeFreqLSTMCell".

##### Returns:

  A tuple containing:
  - A 2D, batch x output_dim, Tensor representing the output of the LSTM
    after reading "inputs" when previous state was "state".
    Here output_dim is num_units.
  - A 2D, batch x state_size, Tensor representing the new state of LSTM
    after reading "inputs" when previous state was "state".

##### Raises:


*  <b>`ValueError`</b>: if an input_size was specified and the provided inputs have
    a different dimension.


- - -

#### `tf.contrib.rnn.TimeFreqLSTMCell.__init__(num_units, use_peepholes=False, cell_clip=None, initializer=None, num_unit_shards=1, forget_bias=1.0, feature_size=None, frequency_skip=None)` {#TimeFreqLSTMCell.__init__}

Initialize the parameters for an LSTM cell.

##### Args:


*  <b>`num_units`</b>: int, The number of units in the LSTM cell
*  <b>`use_peepholes`</b>: bool, set True to enable diagonal/peephole connections.
*  <b>`cell_clip`</b>: (optional) A float value, if provided the cell state is clipped
    by this value prior to the cell output activation.
*  <b>`initializer`</b>: (optional) The initializer to use for the weight and
    projection matrices.
*  <b>`num_unit_shards`</b>: int, How to split the weight matrix.  If >1, the weight
    matrix is stored across num_unit_shards.
*  <b>`forget_bias`</b>: float, Biases of the forget gate are initialized by default
    to 1 in order to reduce the scale of forgetting at the beginning
    of the training.
*  <b>`feature_size`</b>: int, The size of the input feature the LSTM spans over.
*  <b>`frequency_skip`</b>: int, The amount the LSTM filter is shifted by in
    frequency.


- - -

#### `tf.contrib.rnn.TimeFreqLSTMCell.output_size` {#TimeFreqLSTMCell.output_size}




- - -

#### `tf.contrib.rnn.TimeFreqLSTMCell.state_size` {#TimeFreqLSTMCell.state_size}




- - -

#### `tf.contrib.rnn.TimeFreqLSTMCell.zero_state(batch_size, dtype)` {#TimeFreqLSTMCell.zero_state}

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


