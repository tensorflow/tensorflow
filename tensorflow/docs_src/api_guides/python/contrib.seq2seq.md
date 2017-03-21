# Seq2seq Library (contrib)
[TOC]

Module for constructing seq2seq models and dynamic decoding.  Builds on top of
libraries in @{tf.contrib.rnn}.

This library is composed of two primary components:

*   New attention wrappers for @{tf.contrib.rnn.RNNCell} objects.
*   A new object-oriented dynamic decoding framework.

## Attention

Attention wrappers are `RNNCell` objects that wrap other `RNNCell` objects and
implement attention.  The form of attention is determined by a subclass of
@{tf.contrib.seq2seq.AttentionMechanism}.  These subclasses describe the form
of attention (e.g. additive vs. multiplicative) to use when creating the
wrapper.  An instance of an `AttentionMechanism` is constructed with a
`memory` tensor, from which lookup keys and values tensors are created.

### Attention Mechanisms

The two basic attention mechanisms are:
*   @{tf.contrib.seq2seq.BahdanauAttention} (additive attention,
    [ref.](https://arxiv.org/abs/1409.0473))
*   @{tf.contrib.seq2seq.LuongAttention} (multiplicative attention,
    [ref.](https://arxiv.org/abs/1508.04025))

The `memory` tensor passed the attention mechanism's constructor is expected to
be shaped `[batch_size, memory_max_time, memory_depth]`; and often an additional
`memory_sequence_length` vector is accepted.  If provided, the `memory`
tensors' rows are masked with zeros past their true sequence lengths.

Attention mechanisms also have a concept of depth, usually determined as a
construction parameter `num_units`.  For some kinds of attention (like
`BahdanauAttention`), both queries and memory are projected to tensors of depth
`num_units`.  For other kinds (like `LuongAttention`), `num_units` should match
the depth of the queries; and the `memory` tensor will be projected to this
depth.

### Attention Wrappers

The basic attention wrapper is @{tf.contrib.seq2seq.DynamicAttentionWrapper}.
This wrapper accepts an `RNNCell` instance, an instance of `AttentionMechanism`,
and an attention depth parameter (`attention_size`); as well as several
optional arguments that allow one to customize intermediate calculations.

At each time step, the basic calculation performed by this wrapper is:

```python
cell_inputs = concat([inputs, prev_state.attention], -1)
cell_output, next_cell_state = cell(cell_inputs, prev_state.cell_state)
score = attention_mechanism(cell_output)
alignments = softmax(score)
context = matmul(alignments, attention_mechanism.values)
attention = tf.layers.Dense(attention_size)(concat([cell_output, context], 1))
next_state = DynamicAttentionWrapperState(
  cell_state=next_cell_state,
  attention=attention)
output = attention
return output, next_state
```

In practice, a number of the intermediate calculations are configurable.
For example, the initial concatenation of `inputs` and `prev_state.attention`
can be replaced with another mixing function.  The function `softmax` can
be replaced with alternative options when calculating `alignments` from the
`score`.  Finally, the outputs returned by the wrapper can be configured to
be the value `cell_output` instead of `attention`.

The benefit of using a `DynamicAttentionWrapper` is that it plays nicely with
other wrappers and the dynamic decoder described below.  For example, one can
write:

```python
cell = tf.contrib.rnn.DeviceWrapper(LSTMCell(512), "/gpu:0")
attention_mechanism = tf.contrib.seq2seq.LuongAttention(512, encoder_outputs)
attn_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(
  cell, attention_mechanism, attention_size=256)
attn_cell = tf.contrib.rnn.DeviceWrapper(attn_cell, "/gpu:1")
top_cell = tf.contrib.rnn.DeviceWrapper(LSTMCell(512), "/gpu:1")
multi_cell = MultiRNNCell([attn_cell, top_cell])
```

The `multi_rnn` cell will perform the bottom layer calculations on GPU 0;
attention calculations will be performed on GPU 1 and immediately passed
up to the top layer which is also calculated on GPU 1.  The attention is
also passed forward in time to the next time step and copied to GPU 0 for the
next time step of `cell`.  (*Note*: This is just an example of use,
not a suggested device partitioning strategy.)

## Dynamic Decoding

### Decoder base class and functions
*   @{tf.contrib.seq2seq.Decoder}
*   @{tf.contrib.seq2seq.dynamic_decode}

### Basic Decoder
*   @{tf.contrib.seq2seq.BasicDecoderOutput}
*   @{tf.contrib.seq2seq.BasicDecoder}

### Decoder Helpers
*   @{tf.contrib.seq2seq.Helper}
*   @{tf.contrib.seq2seq.CustomHelper}
*   @{tf.contrib.seq2seq.GreedyEmbeddingHelper}
*   @{tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper}
*   @{tf.contrib.seq2seq.ScheduledOutputTrainingHelper}
*   @{tf.contrib.seq2seq.TrainingHelper}
