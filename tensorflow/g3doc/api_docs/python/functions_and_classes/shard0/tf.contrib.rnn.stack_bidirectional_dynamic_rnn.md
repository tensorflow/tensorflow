### `tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs, initial_states_fw=None, initial_states_bw=None, dtype=None, sequence_length=None, scope=None)` {#stack_bidirectional_dynamic_rnn}

Creates a dynamic bidirectional recurrent neural network.

Stacks several bidirectional rnn layers. The combined forward and backward
layer outputs are used as input of the next layer. tf.bidirectional_rnn
does not allow to share forward and backward information between layers.
The input_size of the first forward and backward cells must match.
The initial state for both directions is zero and no intermediate states
are returned.

##### Args:


*  <b>`cells_fw`</b>: List of instances of RNNCell, one per layer,
    to be used for forward direction.
*  <b>`cells_bw`</b>: List of instances of RNNCell, one per layer,
    to be used for backward direction.
*  <b>`inputs`</b>: A length T list of inputs, each a tensor of shape
    [batch_size, input_size], or a nested tuple of such elements.
*  <b>`initial_states_fw`</b>: (optional) A list of the initial states (one per layer)
    for the forward RNN.
    Each tensor must has an appropriate type and shape
    `[batch_size, cell_fw.state_size]`.
*  <b>`initial_states_bw`</b>: (optional) Same as for `initial_states_fw`, but using
    the corresponding properties of `cells_bw`.
*  <b>`dtype`</b>: (optional) The data type for the initial state.  Required if
    either of the initial states are not provided.
*  <b>`sequence_length`</b>: (optional) An int32/int64 vector, size `[batch_size]`,
    containing the actual lengths for each of the sequences.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to None.

##### Returns:

  A tuple (outputs, output_state_fw, output_state_bw) where:

*  <b>`outputs`</b>: Output `Tensor` shaped:
      `batch_size, max_time, layers_output]`. Where layers_output
      are depth-concatenated forward and backward outputs.
    output_states_fw is the final states, one tensor per layer,
      of the forward rnn.
    output_states_bw is the final states, one tensor per layer,
      of the backward rnn.

##### Raises:


*  <b>`TypeError`</b>: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
*  <b>`ValueError`</b>: If inputs is `None`, not a list or an empty list.

