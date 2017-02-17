Operator adding input embedding to the given cell.

Note: in many cases it may be more efficient to not use this wrapper,
but instead concatenate the whole sequence of your inputs in time,
do the embedding on this batch-concatenated sequence, then split it and
feed into your RNN.
- - -

#### `tf.contrib.rnn.EmbeddingWrapper.__call__(inputs, state, scope=None)` {#EmbeddingWrapper.__call__}

Run the cell on embedded inputs.


- - -

#### `tf.contrib.rnn.EmbeddingWrapper.__init__(cell, embedding_classes, embedding_size, initializer=None)` {#EmbeddingWrapper.__init__}

Create a cell with an added input embedding.

##### Args:


*  <b>`cell`</b>: an RNNCell, an embedding will be put before its inputs.
*  <b>`embedding_classes`</b>: integer, how many symbols will be embedded.
*  <b>`embedding_size`</b>: integer, the size of the vectors we embed into.
*  <b>`initializer`</b>: an initializer to use when creating the embedding;
    if None, the initializer from variable scope or a default one is used.

##### Raises:


*  <b>`TypeError`</b>: if cell is not an RNNCell.
*  <b>`ValueError`</b>: if embedding_classes is not positive.


- - -

#### `tf.contrib.rnn.EmbeddingWrapper.output_size` {#EmbeddingWrapper.output_size}




- - -

#### `tf.contrib.rnn.EmbeddingWrapper.state_size` {#EmbeddingWrapper.state_size}




- - -

#### `tf.contrib.rnn.EmbeddingWrapper.zero_state(batch_size, dtype)` {#EmbeddingWrapper.zero_state}

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


