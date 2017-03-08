Basic attention cell wrapper.

Implementation based on https://arxiv.org/abs/1409.0473.
- - -

#### `tf.contrib.rnn.AttentionCellWrapper.__call__(inputs, state, scope=None)` {#AttentionCellWrapper.__call__}

Long short-term memory cell with attention (LSTMA).


- - -

#### `tf.contrib.rnn.AttentionCellWrapper.__init__(cell, attn_length, attn_size=None, attn_vec_size=None, input_size=None, state_is_tuple=False)` {#AttentionCellWrapper.__init__}

Create a cell with attention.

##### Args:


*  <b>`cell`</b>: an RNNCell, an attention is added to it.
*  <b>`attn_length`</b>: integer, the size of an attention window.
*  <b>`attn_size`</b>: integer, the size of an attention vector. Equal to
      cell.output_size by default.
*  <b>`attn_vec_size`</b>: integer, the number of convolutional features calculated
      on attention state and a size of the hidden layer built from
      base cell state. Equal attn_size to by default.
*  <b>`input_size`</b>: integer, the size of a hidden linear layer,
      built from inputs and attention. Derived from the input tensor
      by default.
*  <b>`state_is_tuple`</b>: If True, accepted and returned states are n-tuples, where
    `n = len(cells)`.  By default (False), the states are all
    concatenated along the column axis.

##### Raises:


*  <b>`TypeError`</b>: if cell is not an RNNCell.
*  <b>`ValueError`</b>: if cell returns a state tuple but the flag
      `state_is_tuple` is `False` or if attn_length is zero or less.


- - -

#### `tf.contrib.rnn.AttentionCellWrapper.output_size` {#AttentionCellWrapper.output_size}




- - -

#### `tf.contrib.rnn.AttentionCellWrapper.state_size` {#AttentionCellWrapper.state_size}




- - -

#### `tf.contrib.rnn.AttentionCellWrapper.zero_state(batch_size, dtype)` {#AttentionCellWrapper.zero_state}

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


