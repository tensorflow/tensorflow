Abstract object representing an RNN cell.

An RNN cell, in the most abstract setting, is anything that has
a state and performs some operation that takes a matrix of inputs.
This operation results in an output matrix with `self.output_size` columns.
If `self.state_size` is an integer, this operation also results in a new
state matrix with `self.state_size` columns.  If `self.state_size` is a
tuple of integers, then it results in a tuple of `len(state_size)` state
matrices, each with the a column size corresponding to values in `state_size`.

This module provides a number of basic commonly used RNN cells, such as
LSTM (Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number
of operators that allow add dropouts, projections, or embeddings for inputs.
Constructing multi-layer cells is supported by the class `MultiRNNCell`,
or by calling the `rnn` ops several times. Every `RNNCell` must have the
properties below and and implement `__call__` with the following signature.
- - -

#### `tf.nn.rnn_cell.RNNCell.output_size` {#RNNCell.output_size}

Integer: size of outputs produced by this cell.


- - -

#### `tf.nn.rnn_cell.RNNCell.state_size` {#RNNCell.state_size}

Integer or tuple of integers: size(s) of state(s) used by this cell.


- - -

#### `tf.nn.rnn_cell.RNNCell.zero_state(batch_size, dtype)` {#RNNCell.zero_state}

Return zero-filled state tensor(s).

##### Args:


*  <b>`batch_size`</b>: int, float, or unit Tensor representing the batch size.
*  <b>`dtype`</b>: the data type to use for the state.

##### Returns:

  If `state_size` is an int, then the return value is a `2-D` tensor of
  shape `[batch_size x state_size]` filled with zeros.

  If `state_size` is a nested list or tuple, then the return value is
  a nested list or tuple (of the same structure) of `2-D` tensors with
the shapes `[batch_size x s]` for each s in `state_size`.


