Computes the alpha values in a linear-chain CRF.

See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
- - -

#### `tf.contrib.crf.CrfForwardRnnCell.__call__(inputs, state, scope=None)` {#CrfForwardRnnCell.__call__}

Build the CrfForwardRnnCell.

##### Args:


*  <b>`inputs`</b>: A [batch_size, num_tags] matrix of unary potentials.
*  <b>`state`</b>: A [batch_size, num_tags] matrix containing the previous alpha
      values.
*  <b>`scope`</b>: Unused variable scope of this cell.

##### Returns:

  new_alphas, new_alphas: A pair of [batch_size, num_tags] matrices
      values containing the new alpha values.


- - -

#### `tf.contrib.crf.CrfForwardRnnCell.__init__(transition_params)` {#CrfForwardRnnCell.__init__}

Initialize the CrfForwardRnnCell.

##### Args:


*  <b>`transition_params`</b>: A [num_tags, num_tags] matrix of binary potentials.
      This matrix is expanded into a [1, num_tags, num_tags] in preparation
      for the broadcast summation occurring within the cell.


- - -

#### `tf.contrib.crf.CrfForwardRnnCell.output_size` {#CrfForwardRnnCell.output_size}




- - -

#### `tf.contrib.crf.CrfForwardRnnCell.state_size` {#CrfForwardRnnCell.state_size}




- - -

#### `tf.contrib.crf.CrfForwardRnnCell.zero_state(batch_size, dtype)` {#CrfForwardRnnCell.zero_state}

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


