### `tf.nn.rnn(cell, inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)` {#rnn}

Creates a recurrent neural network specified by RNNCell `cell`.

##### The simplest form of RNN network generated is:

  state = cell.zero_state(...)
  outputs = []
  for input_ in inputs:
    output, state = cell(input_, state)
    outputs.append(output)
  return (outputs, state)

However, a few other options are available:

An initial state can be provided.
If the sequence_length vector is provided, dynamic calculation is performed.
This method of calculation does not compute the RNN steps past the maximum
sequence length of the minibatch (thus saving computational time),
and properly propagates the state at an example's sequence length
to the final state output.

The dynamic calculation performed is, at time t for batch row b,
  (output, state)(b, t) =
    (t >= sequence_length(b))
      ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
      : cell(input(b, t), state(b, t - 1))

##### Args:


*  <b>`cell`</b>: An instance of RNNCell.
*  <b>`inputs`</b>: A length T list of inputs, each a tensor of shape
    [batch_size, input_size].
*  <b>`initial_state`</b>: (optional) An initial state for the RNN.
    If `cell.state_size` is an integer, this must be
    a tensor of appropriate type and shape `[batch_size x cell.state_size]`.
    If `cell.state_size` is a tuple, this should be a tuple of
    tensors having shapes `[batch_size, s] for s in cell.state_size`.
*  <b>`dtype`</b>: (optional) The data type for the initial state.  Required if
    initial_state is not provided.
*  <b>`sequence_length`</b>: Specifies the length of each sequence in inputs.
    An int32 or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to "RNN".

##### Returns:

  A pair (outputs, state) where:
    - outputs is a length T list of outputs (one for each input)
    - state is the final state

##### Raises:


*  <b>`TypeError`</b>: If `cell` is not an instance of RNNCell.
*  <b>`ValueError`</b>: If `inputs` is `None` or an empty list, or if the input depth
    (column size) cannot be inferred from inputs via shape inference.

