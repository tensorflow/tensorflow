Abstract object representing an RNN cell.

The definition of cell in this package differs from the definition used in the
literature. In the literature, cell refers to an object with a single scalar
output. The definition in this package refers to a horizontal array of such
units.

An RNN cell, in the most abstract setting, is anything that has
a state and performs some operation that takes a matrix of inputs.
This operation results in an output matrix with `self.output_size` columns.
If `self.state_size` is an integer, this operation also results in a new
state matrix with `self.state_size` columns.  If `self.state_size` is a
tuple of integers, then it results in a tuple of `len(state_size)` state
matrices, each with a column size corresponding to values in `state_size`.

This module provides a number of basic commonly used RNN cells, such as
LSTM (Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number
of operators that allow add dropouts, projections, or embeddings for inputs.
Constructing multi-layer cells is supported by the class `MultiRNNCell`,
or by calling the `rnn` ops several times. Every `RNNCell` must have the
properties below and implement `__call__` with the following signature.
- - -

#### `tf.contrib.rnn.RNNCell.__call__(inputs, state, scope=None)` {#RNNCell.__call__}

Run this RNN cell on inputs, starting from the given state.

##### Args:


*  <b>`inputs`</b>: `2-D` tensor with shape `[batch_size x input_size]`.
*  <b>`state`</b>: if `self.state_size` is an integer, this should be a `2-D Tensor`
    with shape `[batch_size x self.state_size]`.  Otherwise, if
    `self.state_size` is a tuple of integers, this should be a tuple
    with shapes `[batch_size x s] for s in self.state_size`.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to class name.

##### Returns:

  A pair containing:

  - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
  - New state: Either a single `2-D` tensor, or a tuple of tensors matching
    the arity and shapes of `state`.


- - -

#### `tf.contrib.rnn.RNNCell.output_size` {#RNNCell.output_size}

Integer or TensorShape: size of outputs produced by this cell.


- - -

#### `tf.contrib.rnn.RNNCell.state_size` {#RNNCell.state_size}

size(s) of state(s) used by this cell.

It can be represented by an Integer, a TensorShape or a tuple of Integers
or TensorShapes.


- - -

#### `tf.contrib.rnn.RNNCell.zero_state(batch_size, dtype)` {#RNNCell.zero_state}

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


