Basic LSTM recurrent network cell.

The implementation is based on: http://arxiv.org/abs/1409.2329.

We add forget_bias (default: 1) to the biases of the forget gate in order to
reduce the scale of forgetting in the beginning of the training.

It does not allow cell clipping, a projection layer, and does not
use peep-hole connections: it is the basic baseline.

For advanced models, please use the full LSTMCell that follows.
- - -

#### `tf.contrib.rnn.BasicLSTMCell.__call__(inputs, state, scope=None)` {#BasicLSTMCell.__call__}

Long short-term memory cell (LSTM).


- - -

#### `tf.contrib.rnn.BasicLSTMCell.__init__(num_units, forget_bias=1.0, input_size=None, state_is_tuple=True, activation=tanh)` {#BasicLSTMCell.__init__}

Initialize the basic LSTM cell.

##### Args:


*  <b>`num_units`</b>: int, The number of units in the LSTM cell.
*  <b>`forget_bias`</b>: float, The bias added to forget gates (see above).
*  <b>`input_size`</b>: Deprecated and unused.
*  <b>`state_is_tuple`</b>: If True, accepted and returned states are 2-tuples of
    the `c_state` and `m_state`.  If False, they are concatenated
    along the column axis.  The latter behavior will soon be deprecated.
*  <b>`activation`</b>: Activation function of the inner states.


- - -

#### `tf.contrib.rnn.BasicLSTMCell.output_size` {#BasicLSTMCell.output_size}




- - -

#### `tf.contrib.rnn.BasicLSTMCell.state_size` {#BasicLSTMCell.state_size}




- - -

#### `tf.contrib.rnn.BasicLSTMCell.zero_state(batch_size, dtype)` {#BasicLSTMCell.zero_state}

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


