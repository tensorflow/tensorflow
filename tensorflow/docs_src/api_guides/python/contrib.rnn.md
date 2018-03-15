# RNN and Cells (contrib)
[TOC]

Module for constructing RNN Cells and additional RNN operations.

## Base interface for all RNN Cells

*   @{tf.contrib.rnn.RNNCell}

## Core RNN Cells for use with TensorFlow's core RNN methods

*   @{tf.contrib.rnn.BasicRNNCell}
*   @{tf.contrib.rnn.BasicLSTMCell}
*   @{tf.contrib.rnn.GRUCell}
*   @{tf.contrib.rnn.LSTMCell}
*   @{tf.contrib.rnn.LayerNormBasicLSTMCell}

## Classes storing split `RNNCell` state

*   @{tf.contrib.rnn.LSTMStateTuple}

## Core RNN Cell wrappers (RNNCells that wrap other RNNCells)

*   @{tf.contrib.rnn.MultiRNNCell}
*   @{tf.contrib.rnn.LSTMBlockWrapper}
*   @{tf.contrib.rnn.DropoutWrapper}
*   @{tf.contrib.rnn.EmbeddingWrapper}
*   @{tf.contrib.rnn.InputProjectionWrapper}
*   @{tf.contrib.rnn.OutputProjectionWrapper}
*   @{tf.contrib.rnn.DeviceWrapper}
*   @{tf.contrib.rnn.ResidualWrapper}

### Block RNNCells
*   @{tf.contrib.rnn.LSTMBlockCell}
*   @{tf.contrib.rnn.GRUBlockCell}

### Fused RNNCells
*   @{tf.contrib.rnn.FusedRNNCell}
*   @{tf.contrib.rnn.FusedRNNCellAdaptor}
*   @{tf.contrib.rnn.TimeReversedFusedRNN}
*   @{tf.contrib.rnn.LSTMBlockFusedCell}

### LSTM-like cells
*   @{tf.contrib.rnn.CoupledInputForgetGateLSTMCell}
*   @{tf.contrib.rnn.TimeFreqLSTMCell}
*   @{tf.contrib.rnn.GridLSTMCell}

### RNNCell wrappers
*   @{tf.contrib.rnn.AttentionCellWrapper}
*   @{tf.contrib.rnn.CompiledWrapper}


## Recurrent Neural Networks

TensorFlow provides a number of methods for constructing Recurrent Neural
Networks.

*   @{tf.contrib.rnn.static_rnn}
*   @{tf.contrib.rnn.static_state_saving_rnn}
*   @{tf.contrib.rnn.static_bidirectional_rnn}
*   @{tf.contrib.rnn.stack_bidirectional_dynamic_rnn}
