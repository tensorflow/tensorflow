<!-- This file is machine generated: DO NOT EDIT! -->

# RNN (contrib)
[TOC]

Additional RNN operations and cells.

## This package provides additional contributed RNNCells.

### Fused RNNCells
- - -

### `class tf.contrib.rnn.LSTMFusedCell` {#LSTMFusedCell}

Basic LSTM recurrent network cell.

The implementation is based on: http://arxiv.org/abs/1409.2329.

We add forget_bias (default: 1) to the biases of the forget gate in order to
reduce the scale of forgetting in the beginning of the training.

Unlike BasicLSTMCell, this is a monolithic op and should be much faster. The
weight and bias matrixes should be compatible as long as the variabel scope
matches.
- - -

#### `tf.contrib.rnn.LSTMFusedCell.__init__(num_units, forget_bias=1.0, use_peephole=False)` {#LSTMFusedCell.__init__}

Initialize the basic LSTM cell.

##### Args:


*  <b>`num_units`</b>: int, The number of units in the LSTM cell.
*  <b>`forget_bias`</b>: float, The bias added to forget gates (see above).
*  <b>`use_peephole`</b>: Whether to use peephole connections or not.


- - -

#### `tf.contrib.rnn.LSTMFusedCell.output_size` {#LSTMFusedCell.output_size}




- - -

#### `tf.contrib.rnn.LSTMFusedCell.state_size` {#LSTMFusedCell.state_size}




- - -

#### `tf.contrib.rnn.LSTMFusedCell.zero_state(batch_size, dtype)` {#LSTMFusedCell.zero_state}

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




### LSTM-like cells
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

#### `tf.contrib.rnn.GridLSTMCell.__init__(num_units, use_peepholes=False, share_time_frequency_weights=False, cell_clip=None, initializer=None, num_unit_shards=1, forget_bias=1.0, feature_size=None, frequency_skip=None)` {#GridLSTMCell.__init__}

Initialize the parameters for an LSTM cell.

##### Args:


*  <b>`num_units`</b>: int, The number of units in the LSTM cell
*  <b>`use_peepholes`</b>: bool, default False. Set True to enable diagonal/peephole
    connections.
*  <b>`share_time_frequency_weights`</b>: bool, default False. Set True to enable
    shared cell weights between time and frequency LSTMs.
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

#### `tf.contrib.rnn.GridLSTMCell.output_size` {#GridLSTMCell.output_size}




- - -

#### `tf.contrib.rnn.GridLSTMCell.state_size` {#GridLSTMCell.state_size}




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




### RNNCell wrappers
- - -

### `class tf.contrib.rnn.AttentionCellWrapper` {#AttentionCellWrapper}

Basic attention cell wrapper.

Implementation based on https://arxiv.org/pdf/1601.06733.pdf.
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



