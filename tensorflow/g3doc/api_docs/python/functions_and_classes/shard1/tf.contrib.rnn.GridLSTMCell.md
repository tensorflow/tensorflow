Grid Long short-term memory unit (LSTM) recurrent network cell.

The default is based on:
  Nal Kalchbrenner, Ivo Danihelka and Alex Graves
  "Grid Long Short-Term Memory," Proc. ICLR 2016.
  http://arxiv.org/abs/1507.01526

When peephole connections are used, the implementation is based on:
  Tara N. Sainath and Bo Li
  "Modeling Time-Frequency Patterns with LSTM vs. Convolutional Architectures
  for LVCSR Tasks." submitted to INTERSPEECH, 2016.

The code uses optional peephole connections, shared_weights and cell clipping.
- - -

#### `tf.contrib.rnn.GridLSTMCell.__call__(inputs, state, scope=None)` {#GridLSTMCell.__call__}

Run one step of LSTM.

##### Args:


*  <b>`inputs`</b>: input Tensor, 2D, batch x num_units.
*  <b>`state`</b>: state Tensor, 2D, batch x state_size.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to "LSTMCell".

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

#### `tf.contrib.rnn.GridLSTMCell.__init__(num_units, use_peepholes=False, share_time_frequency_weights=False, cell_clip=None, initializer=None, num_unit_shards=1, forget_bias=1.0, feature_size=None, frequency_skip=None, num_frequency_blocks=1, couple_input_forget_gates=False, state_is_tuple=False)` {#GridLSTMCell.__init__}

Initialize the parameters for an LSTM cell.

##### Args:


*  <b>`num_units`</b>: int, The number of units in the LSTM cell
*  <b>`use_peepholes`</b>: bool, default False. Set True to enable diagonal/peephole
    connections.
*  <b>`share_time_frequency_weights`</b>: bool, default False. Set True to enable
    shared cell weights between time and frequency LSTMs.
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
*  <b>`num_frequency_blocks`</b>: int, The total number of frequency blocks needed to
    cover the whole input feature.
*  <b>`couple_input_forget_gates`</b>: bool, Whether to couple the input and forget
    gates, i.e. f_gate = 1.0 - i_gate, to reduce model parameters and
    computation cost.
*  <b>`state_is_tuple`</b>: If True, accepted and returned states are 2-tuples of
    the `c_state` and `m_state`.  By default (False), they are concatenated
    along the column axis.  This default behavior will soon be deprecated.


- - -

#### `tf.contrib.rnn.GridLSTMCell.output_size` {#GridLSTMCell.output_size}




- - -

#### `tf.contrib.rnn.GridLSTMCell.state_size` {#GridLSTMCell.state_size}




- - -

#### `tf.contrib.rnn.GridLSTMCell.state_tuple_type` {#GridLSTMCell.state_tuple_type}




- - -

#### `tf.contrib.rnn.GridLSTMCell.zero_state(batch_size, dtype)` {#GridLSTMCell.zero_state}

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


