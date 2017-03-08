Bidirectional GridLstm cell.

The bidirection connection is only used in the frequency direction, which
hence doesn't affect the time direction's real-time processing that is
required for online recognition systems.
The current implementation uses different weights for the two directions.
- - -

#### `tf.contrib.rnn.BidirectionalGridLSTMCell.__call__(inputs, state, scope=None)` {#BidirectionalGridLSTMCell.__call__}

Run one step of LSTM.

##### Args:


*  <b>`inputs`</b>: input Tensor, 2D, [batch, num_units].
*  <b>`state`</b>: tuple of Tensors, 2D, [batch, state_size].
*  <b>`scope`</b>: (optional) VariableScope for the created subgraph; if None, it
    defaults to "BidirectionalGridLSTMCell".

##### Returns:

  A tuple containing:
  - A 2D, [batch, output_dim], Tensor representing the output of the LSTM
    after reading "inputs" when previous state was "state".
    Here output_dim is num_units.
  - A 2D, [batch, state_size], Tensor representing the new state of LSTM
    after reading "inputs" when previous state was "state".

##### Raises:


*  <b>`ValueError`</b>: if an input_size was specified and the provided inputs have
    a different dimension.


- - -

#### `tf.contrib.rnn.BidirectionalGridLSTMCell.__init__(num_units, use_peepholes=False, share_time_frequency_weights=False, cell_clip=None, initializer=None, num_unit_shards=1, forget_bias=1.0, feature_size=None, frequency_skip=None, num_frequency_blocks=None, start_freqindex_list=None, end_freqindex_list=None, couple_input_forget_gates=False, backward_slice_offset=0)` {#BidirectionalGridLSTMCell.__init__}

Initialize the parameters for an LSTM cell.

##### Args:


*  <b>`num_units`</b>: int, The number of units in the LSTM cell
*  <b>`use_peepholes`</b>: (optional) bool, default False. Set True to enable
    diagonal/peephole connections.
*  <b>`share_time_frequency_weights`</b>: (optional) bool, default False. Set True to
    enable shared cell weights between time and frequency LSTMs.
*  <b>`cell_clip`</b>: (optional) A float value, default None, if provided the cell
    state is clipped by this value prior to the cell output activation.
*  <b>`initializer`</b>: (optional) The initializer to use for the weight and
    projection matrices, default None.
*  <b>`num_unit_shards`</b>: (optional) int, defualt 1, How to split the weight
    matrix. If > 1,the weight matrix is stored across num_unit_shards.
*  <b>`forget_bias`</b>: (optional) float, default 1.0, The initial bias of the
    forget gates, used to reduce the scale of forgetting at the beginning
    of the training.
*  <b>`feature_size`</b>: (optional) int, default None, The size of the input feature
    the LSTM spans over.
*  <b>`frequency_skip`</b>: (optional) int, default None, The amount the LSTM filter
    is shifted by in frequency.
*  <b>`num_frequency_blocks`</b>: [required] A list of frequency blocks needed to
    cover the whole input feature splitting defined by start_freqindex_list
    and end_freqindex_list.
*  <b>`start_freqindex_list`</b>: [optional], list of ints, default None,  The
    starting frequency index for each frequency block.
*  <b>`end_freqindex_list`</b>: [optional], list of ints, default None. The ending
    frequency index for each frequency block.
*  <b>`couple_input_forget_gates`</b>: (optional) bool, default False, Whether to
    couple the input and forget gates, i.e. f_gate = 1.0 - i_gate, to reduce
    model parameters and computation cost.
*  <b>`backward_slice_offset`</b>: (optional) int32, default 0, the starting offset to
    slice the feature for backward processing.


- - -

#### `tf.contrib.rnn.BidirectionalGridLSTMCell.output_size` {#BidirectionalGridLSTMCell.output_size}




- - -

#### `tf.contrib.rnn.BidirectionalGridLSTMCell.state_size` {#BidirectionalGridLSTMCell.state_size}




- - -

#### `tf.contrib.rnn.BidirectionalGridLSTMCell.state_tuple_type` {#BidirectionalGridLSTMCell.state_tuple_type}




- - -

#### `tf.contrib.rnn.BidirectionalGridLSTMCell.zero_state(batch_size, dtype)` {#BidirectionalGridLSTMCell.zero_state}

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


