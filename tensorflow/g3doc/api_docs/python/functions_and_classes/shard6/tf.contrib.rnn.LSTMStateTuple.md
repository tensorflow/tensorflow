Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

Stores two elements: `(c, h)`, in that order.

Only used when `state_is_tuple=True`.
- - -

#### `tf.contrib.rnn.LSTMStateTuple.__getnewargs__()` {#LSTMStateTuple.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.contrib.rnn.LSTMStateTuple.__getstate__()` {#LSTMStateTuple.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.contrib.rnn.LSTMStateTuple.__new__(_cls, c, h)` {#LSTMStateTuple.__new__}

Create new instance of LSTMStateTuple(c, h)


- - -

#### `tf.contrib.rnn.LSTMStateTuple.__repr__()` {#LSTMStateTuple.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.contrib.rnn.LSTMStateTuple.c` {#LSTMStateTuple.c}

Alias for field number 0


- - -

#### `tf.contrib.rnn.LSTMStateTuple.dtype` {#LSTMStateTuple.dtype}




- - -

#### `tf.contrib.rnn.LSTMStateTuple.h` {#LSTMStateTuple.h}

Alias for field number 1


