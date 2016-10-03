LSTM unit with layer normalization and recurrent dropout.

This class adds layer normalization and recurrent dropout to a
basic LSTM unit. Layer normalization implementation is based on:

  https://arxiv.org/abs/1607.06450.

"Layer Normalization"
Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

and is applied before the internal nonlinearities.
Recurrent dropout is base on:

  https://arxiv.org/abs/1603.05118

"Recurrent Dropout without Memory Loss"
Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.
- - -

#### `tf.contrib.rnn.LayerNormBasicLSTMCell.__call__(inputs, state, scope=None)` {#LayerNormBasicLSTMCell.__call__}

LSTM cell with layer normalization and recurrent dropout.


- - -

#### `tf.contrib.rnn.LayerNormBasicLSTMCell.__init__(num_units, forget_bias=1.0, input_size=None, activation=tanh, layer_norm=True, norm_gain=1.0, norm_shift=0.0, dropout_keep_prob=1.0, dropout_prob_seed=None)` {#LayerNormBasicLSTMCell.__init__}

Initializes the basic LSTM cell.

##### Args:


*  <b>`num_units`</b>: int, The number of units in the LSTM cell.
*  <b>`forget_bias`</b>: float, The bias added to forget gates (see above).
*  <b>`input_size`</b>: Deprecated and unused.
*  <b>`activation`</b>: Activation function of the inner states.
*  <b>`layer_norm`</b>: If `True`, layer normalization will be applied.
*  <b>`norm_gain`</b>: float, The layer normalization gain initial value. If
    `layer_norm` has been set to `False`, this argument will be ignored.
*  <b>`norm_shift`</b>: float, The layer normalization shift initial value. If
    `layer_norm` has been set to `False`, this argument will be ignored.
*  <b>`dropout_keep_prob`</b>: unit Tensor or float between 0 and 1 representing the
    recurrent dropout probability value. If float and 1.0, no dropout will
    be applied.
*  <b>`dropout_prob_seed`</b>: (optional) integer, the randomness seed.


- - -

#### `tf.contrib.rnn.LayerNormBasicLSTMCell.output_size` {#LayerNormBasicLSTMCell.output_size}




- - -

#### `tf.contrib.rnn.LayerNormBasicLSTMCell.state_size` {#LayerNormBasicLSTMCell.state_size}




- - -

#### `tf.contrib.rnn.LayerNormBasicLSTMCell.zero_state(batch_size, dtype)` {#LayerNormBasicLSTMCell.zero_state}

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


