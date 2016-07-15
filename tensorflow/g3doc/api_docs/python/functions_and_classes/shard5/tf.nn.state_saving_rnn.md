### `tf.nn.state_saving_rnn(cell, inputs, state_saver, state_name, sequence_length=None, scope=None)` {#state_saving_rnn}

RNN that accepts a state saver for time-truncated RNN calculation.

##### Args:


*  <b>`cell`</b>: An instance of `RNNCell`.
*  <b>`inputs`</b>: A length T list of inputs, each a tensor of shape
    `[batch_size, input_size]`.
*  <b>`state_saver`</b>: A state saver object with methods `state` and `save_state`.
*  <b>`state_name`</b>: Python string or tuple of strings.  The name to use with the
    state_saver. If the cell returns tuples of states (i.e.,
    `cell.state_size` is a tuple) then `state_name` should be a tuple of
    strings having the same length as `cell.state_size`.  Otherwise it should
    be a single string.
*  <b>`sequence_length`</b>: (optional) An int32/int64 vector size [batch_size].
    See the documentation for rnn() for more details about sequence_length.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to "RNN".

##### Returns:

  A pair (outputs, state) where:
    outputs is a length T list of outputs (one for each input)
    states is the final state

##### Raises:


*  <b>`TypeError`</b>: If `cell` is not an instance of RNNCell.
*  <b>`ValueError`</b>: If `inputs` is `None` or an empty list, or if the arity and
   type of `state_name` does not match that of `cell.state_size`.

