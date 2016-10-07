This is an adaptor for RNNCell classes to be used with `FusedRNNCell`.
- - -

#### `tf.contrib.rnn.FusedRNNCellAdaptor.__call__(inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)` {#FusedRNNCellAdaptor.__call__}




- - -

#### `tf.contrib.rnn.FusedRNNCellAdaptor.__init__(cell, use_dynamic_rnn=False)` {#FusedRNNCellAdaptor.__init__}

Initialize the adaptor.

##### Args:


*  <b>`cell`</b>: an instance of a subclass of a `rnn_cell.RNNCell`.
*  <b>`use_dynamic_rnn`</b>: whether to use dynamic (or static) RNN.


