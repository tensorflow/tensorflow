Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

Stores two elements: `(c, h)`, in that order.

Only used when `state_is_tuple=True`.
- - -

#### `tf.nn.rnn_cell.LSTMStateTuple.__getnewargs__()` {#LSTMStateTuple.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.nn.rnn_cell.LSTMStateTuple.__getstate__()` {#LSTMStateTuple.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.nn.rnn_cell.LSTMStateTuple.__new__(_cls, c, h)` {#LSTMStateTuple.__new__}

Create new instance of LSTMStateTuple(c, h)


- - -

#### `tf.nn.rnn_cell.LSTMStateTuple.__repr__()` {#LSTMStateTuple.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.nn.rnn_cell.LSTMStateTuple.c` {#LSTMStateTuple.c}

Alias for field number 0


- - -

#### `tf.nn.rnn_cell.LSTMStateTuple.dtype` {#LSTMStateTuple.dtype}




- - -

#### `tf.nn.rnn_cell.LSTMStateTuple.h` {#LSTMStateTuple.h}

Alias for field number 1


