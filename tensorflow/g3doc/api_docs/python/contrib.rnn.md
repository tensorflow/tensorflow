<!-- This file is machine generated: DO NOT EDIT! -->

# RNN and Cells (contrib)
[TOC]

RNN Cells and additional RNN operations. See @{$python/contrib.rnn} guide.

- - -

### `class tf.contrib.rnn.RNNCell` {#RNNCell}

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



- - -

### `class tf.contrib.rnn.BasicRNNCell` {#BasicRNNCell}

The most basic RNN cell.
- - -

#### `tf.contrib.rnn.BasicRNNCell.__call__(inputs, state, scope=None)` {#BasicRNNCell.__call__}

Most basic RNN: output = new_state = act(W * input + U * state + B).


- - -

#### `tf.contrib.rnn.BasicRNNCell.__init__(num_units, input_size=None, activation=tanh)` {#BasicRNNCell.__init__}




- - -

#### `tf.contrib.rnn.BasicRNNCell.output_size` {#BasicRNNCell.output_size}




- - -

#### `tf.contrib.rnn.BasicRNNCell.state_size` {#BasicRNNCell.state_size}




- - -

#### `tf.contrib.rnn.BasicRNNCell.zero_state(batch_size, dtype)` {#BasicRNNCell.zero_state}

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



- - -

### `class tf.contrib.rnn.BasicLSTMCell` {#BasicLSTMCell}

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



- - -

### `class tf.contrib.rnn.GRUCell` {#GRUCell}

Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
- - -

#### `tf.contrib.rnn.GRUCell.__call__(inputs, state, scope=None)` {#GRUCell.__call__}

Gated recurrent unit (GRU) with nunits cells.


- - -

#### `tf.contrib.rnn.GRUCell.__init__(num_units, input_size=None, activation=tanh)` {#GRUCell.__init__}




- - -

#### `tf.contrib.rnn.GRUCell.output_size` {#GRUCell.output_size}




- - -

#### `tf.contrib.rnn.GRUCell.state_size` {#GRUCell.state_size}




- - -

#### `tf.contrib.rnn.GRUCell.zero_state(batch_size, dtype)` {#GRUCell.zero_state}

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



- - -

### `class tf.contrib.rnn.LSTMCell` {#LSTMCell}

Long short-term memory unit (LSTM) recurrent network cell.

The default non-peephole implementation is based on:

  http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

S. Hochreiter and J. Schmidhuber.
"Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

The peephole implementation is based on:

  https://research.google.com/pubs/archive/43905.pdf

Hasim Sak, Andrew Senior, and Francoise Beaufays.
"Long short-term memory recurrent neural network architectures for
 large scale acoustic modeling." INTERSPEECH, 2014.

The class uses optional peep-hole connections, optional cell clipping, and
an optional projection layer.
- - -

#### `tf.contrib.rnn.LSTMCell.__call__(inputs, state, scope=None)` {#LSTMCell.__call__}

Run one step of LSTM.

##### Args:


*  <b>`inputs`</b>: input Tensor, 2D, batch x num_units.
*  <b>`state`</b>: if `state_is_tuple` is False, this must be a state Tensor,
    `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
    tuple of state Tensors, both `2-D`, with column sizes `c_state` and
    `m_state`.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to "lstm_cell".

##### Returns:

  A tuple containing:

  - A `2-D, [batch x output_dim]`, Tensor representing the output of the
    LSTM after reading `inputs` when previous state was `state`.
    Here output_dim is:
       num_proj if num_proj was set,
       num_units otherwise.
  - Tensor(s) representing the new state of LSTM after reading `inputs` when
    the previous state was `state`.  Same type and shape(s) as `state`.

##### Raises:


*  <b>`ValueError`</b>: If input size cannot be inferred from inputs via
    static shape inference.


- - -

#### `tf.contrib.rnn.LSTMCell.__init__(num_units, input_size=None, use_peepholes=False, cell_clip=None, initializer=None, num_proj=None, proj_clip=None, num_unit_shards=None, num_proj_shards=None, forget_bias=1.0, state_is_tuple=True, activation=tanh)` {#LSTMCell.__init__}

Initialize the parameters for an LSTM cell.

##### Args:


*  <b>`num_units`</b>: int, The number of units in the LSTM cell
*  <b>`input_size`</b>: Deprecated and unused.
*  <b>`use_peepholes`</b>: bool, set True to enable diagonal/peephole connections.
*  <b>`cell_clip`</b>: (optional) A float value, if provided the cell state is clipped
    by this value prior to the cell output activation.
*  <b>`initializer`</b>: (optional) The initializer to use for the weight and
    projection matrices.
*  <b>`num_proj`</b>: (optional) int, The output dimensionality for the projection
    matrices.  If None, no projection is performed.
*  <b>`proj_clip`</b>: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
    provided, then the projected values are clipped elementwise to within
    `[-proj_clip, proj_clip]`.
*  <b>`num_unit_shards`</b>: Deprecated, will be removed by Jan. 2017.
    Use a variable_scope partitioner instead.
*  <b>`num_proj_shards`</b>: Deprecated, will be removed by Jan. 2017.
    Use a variable_scope partitioner instead.
*  <b>`forget_bias`</b>: Biases of the forget gate are initialized by default to 1
    in order to reduce the scale of forgetting at the beginning of
    the training.
*  <b>`state_is_tuple`</b>: If True, accepted and returned states are 2-tuples of
    the `c_state` and `m_state`.  If False, they are concatenated
    along the column axis.  This latter behavior will soon be deprecated.
*  <b>`activation`</b>: Activation function of the inner states.


- - -

#### `tf.contrib.rnn.LSTMCell.output_size` {#LSTMCell.output_size}




- - -

#### `tf.contrib.rnn.LSTMCell.state_size` {#LSTMCell.state_size}




- - -

#### `tf.contrib.rnn.LSTMCell.zero_state(batch_size, dtype)` {#LSTMCell.zero_state}

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



- - -

### `class tf.contrib.rnn.LayerNormBasicLSTMCell` {#LayerNormBasicLSTMCell}

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



- - -

### `class tf.contrib.rnn.LSTMStateTuple` {#LSTMStateTuple}

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



- - -

### `class tf.contrib.rnn.MultiRNNCell` {#MultiRNNCell}

RNN cell composed sequentially of multiple simple cells.
- - -

#### `tf.contrib.rnn.MultiRNNCell.__call__(inputs, state, scope=None)` {#MultiRNNCell.__call__}

Run this multi-layer cell on inputs, starting from state.


- - -

#### `tf.contrib.rnn.MultiRNNCell.__init__(cells, state_is_tuple=True)` {#MultiRNNCell.__init__}

Create a RNN cell composed sequentially of a number of RNNCells.

##### Args:


*  <b>`cells`</b>: list of RNNCells that will be composed in this order.
*  <b>`state_is_tuple`</b>: If True, accepted and returned states are n-tuples, where
    `n = len(cells)`.  If False, the states are all
    concatenated along the column axis.  This latter behavior will soon be
    deprecated.

##### Raises:


*  <b>`ValueError`</b>: if cells is empty (not allowed), or at least one of the cells
    returns a state tuple but the flag `state_is_tuple` is `False`.


- - -

#### `tf.contrib.rnn.MultiRNNCell.output_size` {#MultiRNNCell.output_size}




- - -

#### `tf.contrib.rnn.MultiRNNCell.state_size` {#MultiRNNCell.state_size}




- - -

#### `tf.contrib.rnn.MultiRNNCell.zero_state(batch_size, dtype)` {#MultiRNNCell.zero_state}

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



- - -

### `class tf.contrib.rnn.LSTMBlockWrapper` {#LSTMBlockWrapper}

This is a helper class that provides housekeeping for LSTM cells.

This may be useful for alternative LSTM and similar type of cells.
The subclasses must implement `_call_cell` method and `num_units` property.
- - -

#### `tf.contrib.rnn.LSTMBlockWrapper.__call__(inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)` {#LSTMBlockWrapper.__call__}

Run this LSTM on inputs, starting from the given state.

##### Args:


*  <b>`inputs`</b>: `3-D` tensor with shape `[time_len, batch_size, input_size]`
    or a list of `time_len` tensors of shape `[batch_size, input_size]`.
*  <b>`initial_state`</b>: a tuple `(initial_cell_state, initial_output)` with tensors
    of shape `[batch_size, self._num_units]`. If this is not provided, the
    cell is expected to create a zero initial state of type `dtype`.
*  <b>`dtype`</b>: The data type for the initial state and expected output. Required
    if `initial_state` is not provided or RNN state has a heterogeneous
    dtype.
*  <b>`sequence_length`</b>: Specifies the length of each sequence in inputs. An
    `int32` or `int64` vector (tensor) size `[batch_size]`, values in `[0,
    time_len).`
    Defaults to `time_len` for each element.
*  <b>`scope`</b>: `VariableScope` for the created subgraph; defaults to class name.

##### Returns:

  A pair containing:

  - Output: A `3-D` tensor of shape `[time_len, batch_size, output_size]`
    or a list of time_len tensors of shape `[batch_size, output_size]`,
    to match the type of the `inputs`.
  - Final state: a tuple `(cell_state, output)` matching `initial_state`.

##### Raises:


*  <b>`ValueError`</b>: in case of shape mismatches


- - -

#### `tf.contrib.rnn.LSTMBlockWrapper.num_units` {#LSTMBlockWrapper.num_units}

Number of units in this cell (output dimension).



- - -

### `class tf.contrib.rnn.DropoutWrapper` {#DropoutWrapper}

Operator adding dropout to inputs and outputs of the given cell.
- - -

#### `tf.contrib.rnn.DropoutWrapper.__call__(inputs, state, scope=None)` {#DropoutWrapper.__call__}

Run the cell with the declared dropouts.


- - -

#### `tf.contrib.rnn.DropoutWrapper.__init__(cell, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)` {#DropoutWrapper.__init__}

Create a cell with added input and/or output dropout.

Dropout is never used on the state.

##### Args:


*  <b>`cell`</b>: an RNNCell, a projection to output_size is added to it.
*  <b>`input_keep_prob`</b>: unit Tensor or float between 0 and 1, input keep
    probability; if it is float and 1, no input dropout will be added.
*  <b>`output_keep_prob`</b>: unit Tensor or float between 0 and 1, output keep
    probability; if it is float and 1, no output dropout will be added.
*  <b>`seed`</b>: (optional) integer, the randomness seed.

##### Raises:


*  <b>`TypeError`</b>: if cell is not an RNNCell.
*  <b>`ValueError`</b>: if keep_prob is not between 0 and 1.


- - -

#### `tf.contrib.rnn.DropoutWrapper.output_size` {#DropoutWrapper.output_size}




- - -

#### `tf.contrib.rnn.DropoutWrapper.state_size` {#DropoutWrapper.state_size}




- - -

#### `tf.contrib.rnn.DropoutWrapper.zero_state(batch_size, dtype)` {#DropoutWrapper.zero_state}

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



- - -

### `class tf.contrib.rnn.EmbeddingWrapper` {#EmbeddingWrapper}

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



- - -

### `class tf.contrib.rnn.InputProjectionWrapper` {#InputProjectionWrapper}

Operator adding an input projection to the given cell.

Note: in many cases it may be more efficient to not use this wrapper,
but instead concatenate the whole sequence of your inputs in time,
do the projection on this batch-concatenated sequence, then split it.
- - -

#### `tf.contrib.rnn.InputProjectionWrapper.__call__(inputs, state, scope=None)` {#InputProjectionWrapper.__call__}

Run the input projection and then the cell.


- - -

#### `tf.contrib.rnn.InputProjectionWrapper.__init__(cell, num_proj, input_size=None)` {#InputProjectionWrapper.__init__}

Create a cell with input projection.

##### Args:


*  <b>`cell`</b>: an RNNCell, a projection of inputs is added before it.
*  <b>`num_proj`</b>: Python integer.  The dimension to project to.
*  <b>`input_size`</b>: Deprecated and unused.

##### Raises:


*  <b>`TypeError`</b>: if cell is not an RNNCell.


- - -

#### `tf.contrib.rnn.InputProjectionWrapper.output_size` {#InputProjectionWrapper.output_size}




- - -

#### `tf.contrib.rnn.InputProjectionWrapper.state_size` {#InputProjectionWrapper.state_size}




- - -

#### `tf.contrib.rnn.InputProjectionWrapper.zero_state(batch_size, dtype)` {#InputProjectionWrapper.zero_state}

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



- - -

### `class tf.contrib.rnn.OutputProjectionWrapper` {#OutputProjectionWrapper}

Operator adding an output projection to the given cell.

Note: in many cases it may be more efficient to not use this wrapper,
but instead concatenate the whole sequence of your outputs in time,
do the projection on this batch-concatenated sequence, then split it
if needed or directly feed into a softmax.
- - -

#### `tf.contrib.rnn.OutputProjectionWrapper.__call__(inputs, state, scope=None)` {#OutputProjectionWrapper.__call__}

Run the cell and output projection on inputs, starting from state.


- - -

#### `tf.contrib.rnn.OutputProjectionWrapper.__init__(cell, output_size)` {#OutputProjectionWrapper.__init__}

Create a cell with output projection.

##### Args:


*  <b>`cell`</b>: an RNNCell, a projection to output_size is added to it.
*  <b>`output_size`</b>: integer, the size of the output after projection.

##### Raises:


*  <b>`TypeError`</b>: if cell is not an RNNCell.
*  <b>`ValueError`</b>: if output_size is not positive.


- - -

#### `tf.contrib.rnn.OutputProjectionWrapper.output_size` {#OutputProjectionWrapper.output_size}




- - -

#### `tf.contrib.rnn.OutputProjectionWrapper.state_size` {#OutputProjectionWrapper.state_size}




- - -

#### `tf.contrib.rnn.OutputProjectionWrapper.zero_state(batch_size, dtype)` {#OutputProjectionWrapper.zero_state}

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



- - -

### `class tf.contrib.rnn.DeviceWrapper` {#DeviceWrapper}

Operator that ensures an RNNCell runs on a particular device.
- - -

#### `tf.contrib.rnn.DeviceWrapper.__call__(inputs, state, scope=None)` {#DeviceWrapper.__call__}

Run the cell on specified device.


- - -

#### `tf.contrib.rnn.DeviceWrapper.__init__(cell, device)` {#DeviceWrapper.__init__}

Construct a `DeviceWrapper` for `cell` with device `device`.

Ensures the wrapped `cell` is called with `tf.device(device)`.

##### Args:


*  <b>`cell`</b>: An instance of `RNNCell`.
*  <b>`device`</b>: A device string or function, for passing to `tf.device`.


- - -

#### `tf.contrib.rnn.DeviceWrapper.output_size` {#DeviceWrapper.output_size}

Integer or TensorShape: size of outputs produced by this cell.


- - -

#### `tf.contrib.rnn.DeviceWrapper.state_size` {#DeviceWrapper.state_size}

size(s) of state(s) used by this cell.

It can be represented by an Integer, a TensorShape or a tuple of Integers
or TensorShapes.


- - -

#### `tf.contrib.rnn.DeviceWrapper.zero_state(batch_size, dtype)` {#DeviceWrapper.zero_state}

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



- - -

### `class tf.contrib.rnn.ResidualWrapper` {#ResidualWrapper}

RNNCell wrapper that ensures cell inputs are added to the outputs.
- - -

#### `tf.contrib.rnn.ResidualWrapper.__call__(inputs, state, scope=None)` {#ResidualWrapper.__call__}

Run the cell and add its inputs to its outputs.

##### Args:


*  <b>`inputs`</b>: cell inputs.
*  <b>`state`</b>: cell state.
*  <b>`scope`</b>: optional cell scope.

##### Returns:

  Tuple of cell outputs and new state.

##### Raises:


*  <b>`TypeError`</b>: If cell inputs and outputs have different structure (type).
*  <b>`ValueError`</b>: If cell inputs and outputs have different structure (value).


- - -

#### `tf.contrib.rnn.ResidualWrapper.__init__(cell)` {#ResidualWrapper.__init__}

Constructs a `ResidualWrapper` for `cell`.

##### Args:


*  <b>`cell`</b>: An instance of `RNNCell`.


- - -

#### `tf.contrib.rnn.ResidualWrapper.output_size` {#ResidualWrapper.output_size}




- - -

#### `tf.contrib.rnn.ResidualWrapper.state_size` {#ResidualWrapper.state_size}




- - -

#### `tf.contrib.rnn.ResidualWrapper.zero_state(batch_size, dtype)` {#ResidualWrapper.zero_state}

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



- - -

### `class tf.contrib.rnn.LSTMBlockCell` {#LSTMBlockCell}

Basic LSTM recurrent network cell.

The implementation is based on: http://arxiv.org/abs/1409.2329.

We add `forget_bias` (default: 1) to the biases of the forget gate in order to
reduce the scale of forgetting in the beginning of the training.

Unlike `core_rnn_cell.LSTMCell`, this is a monolithic op and should be much
faster.  The weight and bias matrixes should be compatible as long as the
variable scope matches.
- - -

#### `tf.contrib.rnn.LSTMBlockCell.__call__(x, states_prev, scope=None)` {#LSTMBlockCell.__call__}

Long short-term memory cell (LSTM).


- - -

#### `tf.contrib.rnn.LSTMBlockCell.__init__(num_units, forget_bias=1.0, use_peephole=False)` {#LSTMBlockCell.__init__}

Initialize the basic LSTM cell.

##### Args:


*  <b>`num_units`</b>: int, The number of units in the LSTM cell.
*  <b>`forget_bias`</b>: float, The bias added to forget gates (see above).
*  <b>`use_peephole`</b>: Whether to use peephole connections or not.


- - -

#### `tf.contrib.rnn.LSTMBlockCell.output_size` {#LSTMBlockCell.output_size}




- - -

#### `tf.contrib.rnn.LSTMBlockCell.state_size` {#LSTMBlockCell.state_size}




- - -

#### `tf.contrib.rnn.LSTMBlockCell.zero_state(batch_size, dtype)` {#LSTMBlockCell.zero_state}

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



- - -

### `class tf.contrib.rnn.GRUBlockCell` {#GRUBlockCell}

Block GRU cell implementation.

The implementation is based on:  http://arxiv.org/abs/1406.1078
Computes the LSTM cell forward propagation for 1 time step.

This kernel op implements the following mathematical equations:

Biases are initialized with:

* `b_ru` - constant_initializer(1.0)
* `b_c` - constant_initializer(0.0)

```
x_h_prev = [x, h_prev]

[r_bar u_bar] = x_h_prev * w_ru + b_ru

r = sigmoid(r_bar)
u = sigmoid(u_bar)

h_prevr = h_prev \circ r

x_h_prevr = [x h_prevr]

c_bar = x_h_prevr * w_c + b_c
c = tanh(c_bar)

h = (1-u) \circ c + u \circ h_prev
```
- - -

#### `tf.contrib.rnn.GRUBlockCell.__call__(x, h_prev, scope=None)` {#GRUBlockCell.__call__}

GRU cell.


- - -

#### `tf.contrib.rnn.GRUBlockCell.__init__(cell_size)` {#GRUBlockCell.__init__}

Initialize the Block GRU cell.

##### Args:


*  <b>`cell_size`</b>: int, GRU cell size.


- - -

#### `tf.contrib.rnn.GRUBlockCell.output_size` {#GRUBlockCell.output_size}




- - -

#### `tf.contrib.rnn.GRUBlockCell.state_size` {#GRUBlockCell.state_size}




- - -

#### `tf.contrib.rnn.GRUBlockCell.zero_state(batch_size, dtype)` {#GRUBlockCell.zero_state}

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



- - -

### `class tf.contrib.rnn.FusedRNNCell` {#FusedRNNCell}

Abstract object representing a fused RNN cell.

A fused RNN cell represents the entire RNN expanded over the time
dimension. In effect, this represents an entire recurrent network.

Unlike RNN cells which are subclasses of `rnn_cell.RNNCell`, a `FusedRNNCell`
operates on the entire time sequence at once, by putting the loop over time
inside the cell. This usually leads to much more efficient, but more complex
and less flexible implementations.

Every `FusedRNNCell` must implement `__call__` with the following signature.
- - -

#### `tf.contrib.rnn.FusedRNNCell.__call__(inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)` {#FusedRNNCell.__call__}

Run this fused RNN on inputs, starting from the given state.

##### Args:


*  <b>`inputs`</b>: `3-D` tensor with shape `[time_len x batch_size x input_size]`
    or a list of `time_len` tensors of shape `[batch_size x input_size]`.
*  <b>`initial_state`</b>: either a tensor with shape `[batch_size x state_size]`
    or a tuple with shapes `[batch_size x s] for s in state_size`, if the
    cell takes tuples. If this is not provided, the cell is expected to
    create a zero initial state of type `dtype`.
*  <b>`dtype`</b>: The data type for the initial state and expected output. Required
    if `initial_state` is not provided or RNN state has a heterogeneous
      dtype.
*  <b>`sequence_length`</b>: Specifies the length of each sequence in inputs. An
    `int32` or `int64` vector (tensor) size `[batch_size]`, values in `[0,
    time_len)`.
    Defaults to `time_len` for each element.
*  <b>`scope`</b>: `VariableScope` or `string` for the created subgraph; defaults to
    class name.

##### Returns:

  A pair containing:

  - Output: A `3-D` tensor of shape `[time_len x batch_size x output_size]`
    or a list of `time_len` tensors of shape `[batch_size x output_size]`,
    to match the type of the `inputs`.
  - Final state: Either a single `2-D` tensor, or a tuple of tensors
    matching the arity and shapes of `initial_state`.



- - -

### `class tf.contrib.rnn.FusedRNNCellAdaptor` {#FusedRNNCellAdaptor}

This is an adaptor for RNNCell classes to be used with `FusedRNNCell`.
- - -

#### `tf.contrib.rnn.FusedRNNCellAdaptor.__call__(inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)` {#FusedRNNCellAdaptor.__call__}




- - -

#### `tf.contrib.rnn.FusedRNNCellAdaptor.__init__(cell, use_dynamic_rnn=False)` {#FusedRNNCellAdaptor.__init__}

Initialize the adaptor.

##### Args:


*  <b>`cell`</b>: an instance of a subclass of a `rnn_cell.RNNCell`.
*  <b>`use_dynamic_rnn`</b>: whether to use dynamic (or static) RNN.



- - -

### `class tf.contrib.rnn.TimeReversedFusedRNN` {#TimeReversedFusedRNN}

This is an adaptor to time-reverse a FusedRNNCell.

For example,

```python
cell = tf.contrib.rnn.BasicRNNCell(10)
fw_lstm = tf.contrib.rnn.FusedRNNCellAdaptor(cell, use_dynamic_rnn=True)
bw_lstm = tf.contrib.rnn.TimeReversedFusedRNN(fw_lstm)
fw_out, fw_state = fw_lstm(inputs)
bw_out, bw_state = bw_lstm(inputs)
```
- - -

#### `tf.contrib.rnn.TimeReversedFusedRNN.__call__(inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)` {#TimeReversedFusedRNN.__call__}




- - -

#### `tf.contrib.rnn.TimeReversedFusedRNN.__init__(cell)` {#TimeReversedFusedRNN.__init__}





- - -

### `class tf.contrib.rnn.LSTMBlockFusedCell` {#LSTMBlockFusedCell}

FusedRNNCell implementation of LSTM.

This is an extremely efficient LSTM implementation, that uses a single TF op
for the entire LSTM. It should be both faster and more memory-efficient than
LSTMBlockCell defined above.

The implementation is based on: http://arxiv.org/abs/1409.2329.

We add forget_bias (default: 1) to the biases of the forget gate in order to
reduce the scale of forgetting in the beginning of the training.

The variable naming is consistent with `core_rnn_cell.LSTMCell`.
- - -

#### `tf.contrib.rnn.LSTMBlockFusedCell.__call__(inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)` {#LSTMBlockFusedCell.__call__}

Run this LSTM on inputs, starting from the given state.

##### Args:


*  <b>`inputs`</b>: `3-D` tensor with shape `[time_len, batch_size, input_size]`
    or a list of `time_len` tensors of shape `[batch_size, input_size]`.
*  <b>`initial_state`</b>: a tuple `(initial_cell_state, initial_output)` with tensors
    of shape `[batch_size, self._num_units]`. If this is not provided, the
    cell is expected to create a zero initial state of type `dtype`.
*  <b>`dtype`</b>: The data type for the initial state and expected output. Required
    if `initial_state` is not provided or RNN state has a heterogeneous
    dtype.
*  <b>`sequence_length`</b>: Specifies the length of each sequence in inputs. An
    `int32` or `int64` vector (tensor) size `[batch_size]`, values in `[0,
    time_len).`
    Defaults to `time_len` for each element.
*  <b>`scope`</b>: `VariableScope` for the created subgraph; defaults to class name.

##### Returns:

  A pair containing:

  - Output: A `3-D` tensor of shape `[time_len, batch_size, output_size]`
    or a list of time_len tensors of shape `[batch_size, output_size]`,
    to match the type of the `inputs`.
  - Final state: a tuple `(cell_state, output)` matching `initial_state`.

##### Raises:


*  <b>`ValueError`</b>: in case of shape mismatches


- - -

#### `tf.contrib.rnn.LSTMBlockFusedCell.__init__(num_units, forget_bias=1.0, cell_clip=None, use_peephole=False)` {#LSTMBlockFusedCell.__init__}

Initialize the LSTM cell.

##### Args:


*  <b>`num_units`</b>: int, The number of units in the LSTM cell.
*  <b>`forget_bias`</b>: float, The bias added to forget gates (see above).
*  <b>`cell_clip`</b>: clip the cell to this value. Defaults to `3`.
*  <b>`use_peephole`</b>: Whether to use peephole connections or not.


- - -

#### `tf.contrib.rnn.LSTMBlockFusedCell.num_units` {#LSTMBlockFusedCell.num_units}

Number of units in this cell (output dimension).



- - -

### `class tf.contrib.rnn.CoupledInputForgetGateLSTMCell` {#CoupledInputForgetGateLSTMCell}

Long short-term memory unit (LSTM) recurrent network cell.

The default non-peephole implementation is based on:

  http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

S. Hochreiter and J. Schmidhuber.
"Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

The peephole implementation is based on:

  https://research.google.com/pubs/archive/43905.pdf

Hasim Sak, Andrew Senior, and Francoise Beaufays.
"Long short-term memory recurrent neural network architectures for
 large scale acoustic modeling." INTERSPEECH, 2014.

The coupling of input and forget gate is based on:

  http://arxiv.org/pdf/1503.04069.pdf

Greff et al. "LSTM: A Search Space Odyssey"

The class uses optional peep-hole connections, and an optional projection
layer.
- - -

#### `tf.contrib.rnn.CoupledInputForgetGateLSTMCell.__call__(inputs, state, scope=None)` {#CoupledInputForgetGateLSTMCell.__call__}

Run one step of LSTM.

##### Args:


*  <b>`inputs`</b>: input Tensor, 2D, batch x num_units.
*  <b>`state`</b>: if `state_is_tuple` is False, this must be a state Tensor,
    `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
    tuple of state Tensors, both `2-D`, with column sizes `c_state` and
    `m_state`.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to "LSTMCell".

##### Returns:

  A tuple containing:
  - A `2-D, [batch x output_dim]`, Tensor representing the output of the
    LSTM after reading `inputs` when previous state was `state`.
    Here output_dim is:
       num_proj if num_proj was set,
       num_units otherwise.
  - Tensor(s) representing the new state of LSTM after reading `inputs` when
    the previous state was `state`.  Same type and shape(s) as `state`.

##### Raises:


*  <b>`ValueError`</b>: If input size cannot be inferred from inputs via
    static shape inference.


- - -

#### `tf.contrib.rnn.CoupledInputForgetGateLSTMCell.__init__(num_units, use_peepholes=False, initializer=None, num_proj=None, proj_clip=None, num_unit_shards=1, num_proj_shards=1, forget_bias=1.0, state_is_tuple=False, activation=tanh)` {#CoupledInputForgetGateLSTMCell.__init__}

Initialize the parameters for an LSTM cell.

##### Args:


*  <b>`num_units`</b>: int, The number of units in the LSTM cell
*  <b>`use_peepholes`</b>: bool, set True to enable diagonal/peephole connections.
*  <b>`initializer`</b>: (optional) The initializer to use for the weight and
    projection matrices.
*  <b>`num_proj`</b>: (optional) int, The output dimensionality for the projection
    matrices.  If None, no projection is performed.
*  <b>`proj_clip`</b>: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
  provided, then the projected values are clipped elementwise to within
  `[-proj_clip, proj_clip]`.

*  <b>`num_unit_shards`</b>: How to split the weight matrix.  If >1, the weight
    matrix is stored across num_unit_shards.
*  <b>`num_proj_shards`</b>: How to split the projection matrix.  If >1, the
    projection matrix is stored across num_proj_shards.
*  <b>`forget_bias`</b>: Biases of the forget gate are initialized by default to 1
    in order to reduce the scale of forgetting at the beginning of
    the training.
*  <b>`state_is_tuple`</b>: If True, accepted and returned states are 2-tuples of
    the `c_state` and `m_state`.  By default (False), they are concatenated
    along the column axis.  This default behavior will soon be deprecated.
*  <b>`activation`</b>: Activation function of the inner states.


- - -

#### `tf.contrib.rnn.CoupledInputForgetGateLSTMCell.output_size` {#CoupledInputForgetGateLSTMCell.output_size}




- - -

#### `tf.contrib.rnn.CoupledInputForgetGateLSTMCell.state_size` {#CoupledInputForgetGateLSTMCell.state_size}




- - -

#### `tf.contrib.rnn.CoupledInputForgetGateLSTMCell.zero_state(batch_size, dtype)` {#CoupledInputForgetGateLSTMCell.zero_state}

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



- - -

### `class tf.contrib.rnn.TimeFreqLSTMCell` {#TimeFreqLSTMCell}

Time-Frequency Long short-term memory unit (LSTM) recurrent network cell.

This implementation is based on:

  Tara N. Sainath and Bo Li
  "Modeling Time-Frequency Patterns with LSTM vs. Convolutional Architectures
  for LVCSR Tasks." submitted to INTERSPEECH, 2016.

It uses peep-hole connections and optional cell clipping.
- - -

#### `tf.contrib.rnn.TimeFreqLSTMCell.__call__(inputs, state, scope=None)` {#TimeFreqLSTMCell.__call__}

Run one step of LSTM.

##### Args:


*  <b>`inputs`</b>: input Tensor, 2D, batch x num_units.
*  <b>`state`</b>: state Tensor, 2D, batch x state_size.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to
    "TimeFreqLSTMCell".

##### Returns:

  A tuple containing:
  - A 2D, batch x output_dim, Tensor representing the output of the LSTM
    after reading "inputs" when previous state was "state".
    Here output_dim is num_units.
  - A 2D, batch x state_size, Tensor representing the new state of LSTM
    after reading "inputs" when previous state was "state".

##### Raises:


*  <b>`ValueError`</b>: if an input_size was specified and the provided inputs have
    a different dimension.


- - -

#### `tf.contrib.rnn.TimeFreqLSTMCell.__init__(num_units, use_peepholes=False, cell_clip=None, initializer=None, num_unit_shards=1, forget_bias=1.0, feature_size=None, frequency_skip=None)` {#TimeFreqLSTMCell.__init__}

Initialize the parameters for an LSTM cell.

##### Args:


*  <b>`num_units`</b>: int, The number of units in the LSTM cell
*  <b>`use_peepholes`</b>: bool, set True to enable diagonal/peephole connections.
*  <b>`cell_clip`</b>: (optional) A float value, if provided the cell state is clipped
    by this value prior to the cell output activation.
*  <b>`initializer`</b>: (optional) The initializer to use for the weight and
    projection matrices.
*  <b>`num_unit_shards`</b>: int, How to split the weight matrix.  If >1, the weight
    matrix is stored across num_unit_shards.
*  <b>`forget_bias`</b>: float, Biases of the forget gate are initialized by default
    to 1 in order to reduce the scale of forgetting at the beginning
    of the training.
*  <b>`feature_size`</b>: int, The size of the input feature the LSTM spans over.
*  <b>`frequency_skip`</b>: int, The amount the LSTM filter is shifted by in
    frequency.


- - -

#### `tf.contrib.rnn.TimeFreqLSTMCell.output_size` {#TimeFreqLSTMCell.output_size}




- - -

#### `tf.contrib.rnn.TimeFreqLSTMCell.state_size` {#TimeFreqLSTMCell.state_size}




- - -

#### `tf.contrib.rnn.TimeFreqLSTMCell.zero_state(batch_size, dtype)` {#TimeFreqLSTMCell.zero_state}

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



- - -

### `class tf.contrib.rnn.GridLSTMCell` {#GridLSTMCell}

Grid Long short-term memory unit (LSTM) recurrent network cell.

The default is based on:
  Nal Kalchbrenner, Ivo Danihelka and Alex Graves
  "Grid Long Short-Term Memory," Proc. ICLR 2016.
  http://arxiv.org/abs/1507.01526

When peephole connections are used, the implementation is based on:
  Tara N. Sainath and Bo Li
  "Modeling Time-Frequency Patterns with LSTM vs. Convolutional Architectures
  for LVCSR Tasks." submitted to INTERSPEECH, 2016.

The code uses optional peephole connections, shared_weights and cell clipping.
- - -

#### `tf.contrib.rnn.GridLSTMCell.__call__(inputs, state, scope=None)` {#GridLSTMCell.__call__}

Run one step of LSTM.

##### Args:


*  <b>`inputs`</b>: input Tensor, 2D, [batch, feature_size].
*  <b>`state`</b>: Tensor or tuple of Tensors, 2D, [batch, state_size], depends on the
    flag self._state_is_tuple.
*  <b>`scope`</b>: (optional) VariableScope for the created subgraph; if None, it
    defaults to "GridLSTMCell".

##### Returns:

  A tuple containing:
  - A 2D, [batch, output_dim], Tensor representing the output of the LSTM
    after reading "inputs" when previous state was "state".
    Here output_dim is num_units.
  - A 2D, [batch, state_size], Tensor representing the new state of LSTM
    after reading "inputs" when previous state was "state".

##### Raises:


*  <b>`ValueError`</b>: if an input_size was specified and the provided inputs have
    a different dimension.


- - -

#### `tf.contrib.rnn.GridLSTMCell.__init__(num_units, use_peepholes=False, share_time_frequency_weights=False, cell_clip=None, initializer=None, num_unit_shards=1, forget_bias=1.0, feature_size=None, frequency_skip=None, num_frequency_blocks=None, start_freqindex_list=None, end_freqindex_list=None, couple_input_forget_gates=False, state_is_tuple=False)` {#GridLSTMCell.__init__}

Initialize the parameters for an LSTM cell.

##### Args:


*  <b>`num_units`</b>: int, The number of units in the LSTM cell
*  <b>`use_peepholes`</b>: (optional) bool, default False. Set True to enable
    diagonal/peephole connections.
*  <b>`share_time_frequency_weights`</b>: (optional) bool, default False. Set True to
    enable shared cell weights between time and frequency LSTMs.
*  <b>`cell_clip`</b>: (optional) A float value, default None, if provided the cell
    state is clipped by this value prior to the cell output activation.
*  <b>`initializer`</b>: (optional) The initializer to use for the weight and
    projection matrices, default None.
*  <b>`num_unit_shards`</b>: (optional) int, defualt 1, How to split the weight
    matrix. If > 1,the weight matrix is stored across num_unit_shards.
*  <b>`forget_bias`</b>: (optional) float, default 1.0, The initial bias of the
    forget gates, used to reduce the scale of forgetting at the beginning
    of the training.
*  <b>`feature_size`</b>: (optional) int, default None, The size of the input feature
    the LSTM spans over.
*  <b>`frequency_skip`</b>: (optional) int, default None, The amount the LSTM filter
    is shifted by in frequency.
*  <b>`num_frequency_blocks`</b>: [required] A list of frequency blocks needed to
    cover the whole input feature splitting defined by start_freqindex_list
    and end_freqindex_list.
*  <b>`start_freqindex_list`</b>: [optional], list of ints, default None,  The
    starting frequency index for each frequency block.
*  <b>`end_freqindex_list`</b>: [optional], list of ints, default None. The ending
    frequency index for each frequency block.
*  <b>`couple_input_forget_gates`</b>: (optional) bool, default False, Whether to
    couple the input and forget gates, i.e. f_gate = 1.0 - i_gate, to reduce
    model parameters and computation cost.
*  <b>`state_is_tuple`</b>: If True, accepted and returned states are 2-tuples of
    the `c_state` and `m_state`.  By default (False), they are concatenated
    along the column axis.  This default behavior will soon be deprecated.

##### Raises:


*  <b>`ValueError`</b>: if the num_frequency_blocks list is not specified


- - -

#### `tf.contrib.rnn.GridLSTMCell.output_size` {#GridLSTMCell.output_size}




- - -

#### `tf.contrib.rnn.GridLSTMCell.state_size` {#GridLSTMCell.state_size}




- - -

#### `tf.contrib.rnn.GridLSTMCell.state_tuple_type` {#GridLSTMCell.state_tuple_type}




- - -

#### `tf.contrib.rnn.GridLSTMCell.zero_state(batch_size, dtype)` {#GridLSTMCell.zero_state}

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



- - -

### `class tf.contrib.rnn.AttentionCellWrapper` {#AttentionCellWrapper}

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



- - -

### `class tf.contrib.rnn.CompiledWrapper` {#CompiledWrapper}

Wraps step execution in an XLA JIT scope.
- - -

#### `tf.contrib.rnn.CompiledWrapper.__call__(inputs, state, scope=None)` {#CompiledWrapper.__call__}




- - -

#### `tf.contrib.rnn.CompiledWrapper.__init__(cell, compile_stateful=False)` {#CompiledWrapper.__init__}

Create CompiledWrapper cell.

##### Args:


*  <b>`cell`</b>: Instance of `RNNCell`.
*  <b>`compile_stateful`</b>: Whether to compile stateful ops like initializers
    and random number generators (default: False).


- - -

#### `tf.contrib.rnn.CompiledWrapper.output_size` {#CompiledWrapper.output_size}




- - -

#### `tf.contrib.rnn.CompiledWrapper.state_size` {#CompiledWrapper.state_size}




- - -

#### `tf.contrib.rnn.CompiledWrapper.zero_state(batch_size, dtype)` {#CompiledWrapper.zero_state}

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



- - -

### `tf.contrib.rnn.static_rnn(cell, inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)` {#static_rnn}

Creates a recurrent neural network specified by RNNCell `cell`.

The simplest form of RNN network generated is:

```python
  state = cell.zero_state(...)
  outputs = []
  for input_ in inputs:
    output, state = cell(input_, state)
    outputs.append(output)
  return (outputs, state)
```
However, a few other options are available:

An initial state can be provided.
If the sequence_length vector is provided, dynamic calculation is performed.
This method of calculation does not compute the RNN steps past the maximum
sequence length of the minibatch (thus saving computational time),
and properly propagates the state at an example's sequence length
to the final state output.

The dynamic calculation performed is, at time `t` for batch row `b`,

```python
  (output, state)(b, t) =
    (t >= sequence_length(b))
      ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
      : cell(input(b, t), state(b, t - 1))
```

##### Args:


*  <b>`cell`</b>: An instance of RNNCell.
*  <b>`inputs`</b>: A length T list of inputs, each a `Tensor` of shape
    `[batch_size, input_size]`, or a nested tuple of such elements.
*  <b>`initial_state`</b>: (optional) An initial state for the RNN.
    If `cell.state_size` is an integer, this must be
    a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
    If `cell.state_size` is a tuple, this should be a tuple of
    tensors having shapes `[batch_size, s] for s in cell.state_size`.
*  <b>`dtype`</b>: (optional) The data type for the initial state and expected output.
    Required if initial_state is not provided or RNN state has a heterogeneous
    dtype.
*  <b>`sequence_length`</b>: Specifies the length of each sequence in inputs.
    An int32 or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to "rnn".

##### Returns:

  A pair (outputs, state) where:

  - outputs is a length T list of outputs (one for each input), or a nested
    tuple of such elements.
  - state is the final state

##### Raises:


*  <b>`TypeError`</b>: If `cell` is not an instance of RNNCell.
*  <b>`ValueError`</b>: If `inputs` is `None` or an empty list, or if the input depth
    (column size) cannot be inferred from inputs via shape inference.


- - -

### `tf.contrib.rnn.static_state_saving_rnn(cell, inputs, state_saver, state_name, sequence_length=None, scope=None)` {#static_state_saving_rnn}

RNN that accepts a state saver for time-truncated RNN calculation.

##### Args:


*  <b>`cell`</b>: An instance of `RNNCell`.
*  <b>`inputs`</b>: A length T list of inputs, each a `Tensor` of shape
    `[batch_size, input_size]`.
*  <b>`state_saver`</b>: A state saver object with methods `state` and `save_state`.
*  <b>`state_name`</b>: Python string or tuple of strings.  The name to use with the
    state_saver. If the cell returns tuples of states (i.e.,
    `cell.state_size` is a tuple) then `state_name` should be a tuple of
    strings having the same length as `cell.state_size`.  Otherwise it should
    be a single string.
*  <b>`sequence_length`</b>: (optional) An int32/int64 vector size [batch_size].
    See the documentation for rnn() for more details about sequence_length.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to "rnn".

##### Returns:

  A pair (outputs, state) where:
    outputs is a length T list of outputs (one for each input)
    states is the final state

##### Raises:


*  <b>`TypeError`</b>: If `cell` is not an instance of RNNCell.
*  <b>`ValueError`</b>: If `inputs` is `None` or an empty list, or if the arity and
   type of `state_name` does not match that of `cell.state_size`.


- - -

### `tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs, initial_state_fw=None, initial_state_bw=None, dtype=None, sequence_length=None, scope=None)` {#static_bidirectional_rnn}

Creates a bidirectional recurrent neural network.

Similar to the unidirectional case above (rnn) but takes input and builds
independent forward and backward RNNs with the final forward and backward
outputs depth-concatenated, such that the output will have the format
[time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
forward and backward cell must match. The initial state for both directions
is zero by default (but can be set optionally) and no intermediate states are
ever returned -- the network is fully unrolled for the given (passed in)
length(s) of the sequence(s) or completely unrolled if length(s) is not given.

##### Args:


*  <b>`cell_fw`</b>: An instance of RNNCell, to be used for forward direction.
*  <b>`cell_bw`</b>: An instance of RNNCell, to be used for backward direction.
*  <b>`inputs`</b>: A length T list of inputs, each a tensor of shape
    [batch_size, input_size], or a nested tuple of such elements.
*  <b>`initial_state_fw`</b>: (optional) An initial state for the forward RNN.
    This must be a tensor of appropriate type and shape
    `[batch_size, cell_fw.state_size]`.
    If `cell_fw.state_size` is a tuple, this should be a tuple of
    tensors having shapes `[batch_size, s] for s in cell_fw.state_size`.
*  <b>`initial_state_bw`</b>: (optional) Same as for `initial_state_fw`, but using
    the corresponding properties of `cell_bw`.
*  <b>`dtype`</b>: (optional) The data type for the initial state.  Required if
    either of the initial states are not provided.
*  <b>`sequence_length`</b>: (optional) An int32/int64 vector, size `[batch_size]`,
    containing the actual lengths for each of the sequences.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to
    "bidirectional_rnn"

##### Returns:

  A tuple (outputs, output_state_fw, output_state_bw) where:
    outputs is a length `T` list of outputs (one for each input), which
      are depth-concatenated forward and backward outputs.
    output_state_fw is the final state of the forward rnn.
    output_state_bw is the final state of the backward rnn.

##### Raises:


*  <b>`TypeError`</b>: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
*  <b>`ValueError`</b>: If inputs is None or an empty list.


- - -

### `tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs, initial_states_fw=None, initial_states_bw=None, dtype=None, sequence_length=None, scope=None)` {#stack_bidirectional_dynamic_rnn}

Creates a dynamic bidirectional recurrent neural network.

Stacks several bidirectional rnn layers. The combined forward and backward
layer outputs are used as input of the next layer. tf.bidirectional_rnn
does not allow to share forward and backward information between layers.
The input_size of the first forward and backward cells must match.
The initial state for both directions is zero and no intermediate states
are returned.

##### Args:


*  <b>`cells_fw`</b>: List of instances of RNNCell, one per layer,
    to be used for forward direction.
*  <b>`cells_bw`</b>: List of instances of RNNCell, one per layer,
    to be used for backward direction.
*  <b>`inputs`</b>: A length T list of inputs, each a tensor of shape
    [batch_size, input_size], or a nested tuple of such elements.
*  <b>`initial_states_fw`</b>: (optional) A list of the initial states (one per layer)
    for the forward RNN.
    Each tensor must has an appropriate type and shape
    `[batch_size, cell_fw.state_size]`.
*  <b>`initial_states_bw`</b>: (optional) Same as for `initial_states_fw`, but using
    the corresponding properties of `cells_bw`.
*  <b>`dtype`</b>: (optional) The data type for the initial state.  Required if
    either of the initial states are not provided.
*  <b>`sequence_length`</b>: (optional) An int32/int64 vector, size `[batch_size]`,
    containing the actual lengths for each of the sequences.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to None.

##### Returns:

  A tuple (outputs, output_state_fw, output_state_bw) where:

*  <b>`outputs`</b>: Output `Tensor` shaped:
      `batch_size, max_time, layers_output]`. Where layers_output
      are depth-concatenated forward and backward outputs.
    output_states_fw is the final states, one tensor per layer,
      of the forward rnn.
    output_states_bw is the final states, one tensor per layer,
      of the backward rnn.

##### Raises:


*  <b>`TypeError`</b>: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
*  <b>`ValueError`</b>: If inputs is `None`, not a list or an empty list.


