Abstract object representing a fused RNN cell.

A fused RNN cell represents the entire RNN expanded over the time
dimension. In effect, this represents an entire recurrent network.

Unlike RNN cells which are subclasses of rnn_cell.RNNCell , a `FusedRNNCell`
operates on the entire time sequence at once, by putting the loop over time
inside the cell. This usually leads to much more efficient, but more complex
and less flexible implementations.

Every `FusedRNNCell` must implement `__call__` with the following signature.
- - -

#### `tf.contrib.rnn.FusedRNNCell.__call__(inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)` {#FusedRNNCell.__call__}

Run this fused RNN on inputs, starting from the given state.

##### Args:


*  <b>`inputs`</b>: `3-D` tensor with shape `[time_len x batch_size x input_size]`
    or a list of `time_len` tensors of shape `[batch_size x input_size]`.
*  <b>`initial_state`</b>: either a tensor with shape `[batch_size x state_size]`
    or a tuple with shapes `[batch_size x s] for s in state_size`, if the
    cell takes tuples. If this is not provided, the cell is expected to
    create a zero initial state of type `dtype`.
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
  - Final state: Either a single `2-D` tensor, or a tuple of tensors
    matching the arity and shapes of `initial_state`.


