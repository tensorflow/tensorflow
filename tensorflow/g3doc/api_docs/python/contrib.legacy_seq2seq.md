<!-- This file is machine generated: DO NOT EDIT! -->

# Sequence to Sequence (contrib)
[TOC]

Deprecated library for creating sequence-to-sequence models in TensorFlow.

- - -

### `tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs, initial_state, attention_states, cell, output_size=None, num_heads=1, loop_function=None, dtype=None, scope=None, initial_state_attention=False)` {#attention_decoder}

RNN decoder with attention for the sequence-to-sequence model.

In this context "attention" means that, during decoding, the RNN can look up
information in the additional tensor attention_states, and it does this by
focusing on a few entries from the tensor. This model has proven to yield
especially good results in a number of sequence-to-sequence tasks. This
implementation is based on http://arxiv.org/abs/1412.7449 (see below for
details). It is recommended for complex sequence-to-sequence tasks.

##### Args:


*  <b>`decoder_inputs`</b>: A list of 2D Tensors [batch_size x input_size].
*  <b>`initial_state`</b>: 2D Tensor [batch_size x cell.state_size].
*  <b>`attention_states`</b>: 3D Tensor [batch_size x attn_length x attn_size].
*  <b>`cell`</b>: core_rnn_cell.RNNCell defining the cell function and size.
*  <b>`output_size`</b>: Size of the output vectors; if None, we use cell.output_size.
*  <b>`num_heads`</b>: Number of attention heads that read from attention_states.
*  <b>`loop_function`</b>: If not None, this function will be applied to i-th output
    in order to generate i+1-th input, and decoder_inputs will be ignored,
    except for the first element ("GO" symbol). This can be used for decoding,
    but also for training to emulate http://arxiv.org/abs/1506.03099.
    Signature -- loop_function(prev, i) = next
      * prev is a 2D Tensor of shape [batch_size x output_size],
      * i is an integer, the step number (when advanced control is needed),
      * next is a 2D Tensor of shape [batch_size x input_size].
*  <b>`dtype`</b>: The dtype to use for the RNN initial state (default: tf.float32).
*  <b>`scope`</b>: VariableScope for the created subgraph; default: "attention_decoder".
*  <b>`initial_state_attention`</b>: If False (default), initial attentions are zero.
    If True, initialize the attentions from the initial state and attention
    states -- useful when we wish to resume decoding from a previously
    stored decoder state and attention states.

##### Returns:

  A tuple of the form (outputs, state), where:

*  <b>`outputs`</b>: A list of the same length as decoder_inputs of 2D Tensors of
      shape [batch_size x output_size]. These represent the generated outputs.
      Output i is computed from input i (which is either the i-th element
      of decoder_inputs or loop_function(output {i-1}, i)) as follows.
      First, we run the cell on a combination of the input and previous
      attention masks:
        cell_output, new_state = cell(linear(input, prev_attn), prev_state).
      Then, we calculate new attention masks:
        new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
      and then we calculate the output:
        output = linear(cell_output, new_attn).
*  <b>`state`</b>: The state of each decoder cell the final time-step.
      It is a 2D Tensor of shape [batch_size x cell.state_size].

##### Raises:


*  <b>`ValueError`</b>: when num_heads is not positive, there are no inputs, shapes
    of attention_states are not set, or input size cannot be inferred
    from the input.


- - -

### `tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, dtype=tf.float32, scope=None)` {#basic_rnn_seq2seq}

Basic RNN sequence-to-sequence model.

This model first runs an RNN to encode encoder_inputs into a state vector,
then runs decoder, initialized with the last encoder state, on decoder_inputs.
Encoder and decoder use the same RNN cell type, but don't share parameters.

##### Args:


*  <b>`encoder_inputs`</b>: A list of 2D Tensors [batch_size x input_size].
*  <b>`decoder_inputs`</b>: A list of 2D Tensors [batch_size x input_size].
*  <b>`cell`</b>: core_rnn_cell.RNNCell defining the cell function and size.
*  <b>`dtype`</b>: The dtype of the initial state of the RNN cell (default: tf.float32).
*  <b>`scope`</b>: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".

##### Returns:

  A tuple of the form (outputs, state), where:

*  <b>`outputs`</b>: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x output_size] containing the generated outputs.
*  <b>`state`</b>: The state of each decoder cell in the final time-step.
      It is a 2D Tensor of shape [batch_size x cell.state_size].


- - -

### `tf.contrib.legacy_seq2seq.embedding_attention_decoder(decoder_inputs, initial_state, attention_states, cell, num_symbols, embedding_size, num_heads=1, output_size=None, output_projection=None, feed_previous=False, update_embedding_for_previous=True, dtype=None, scope=None, initial_state_attention=False)` {#embedding_attention_decoder}

RNN decoder with embedding and attention and a pure-decoding option.

##### Args:


*  <b>`decoder_inputs`</b>: A list of 1D batch-sized int32 Tensors (decoder inputs).
*  <b>`initial_state`</b>: 2D Tensor [batch_size x cell.state_size].
*  <b>`attention_states`</b>: 3D Tensor [batch_size x attn_length x attn_size].
*  <b>`cell`</b>: core_rnn_cell.RNNCell defining the cell function.
*  <b>`num_symbols`</b>: Integer, how many symbols come into the embedding.
*  <b>`embedding_size`</b>: Integer, the length of the embedding vector for each symbol.
*  <b>`num_heads`</b>: Number of attention heads that read from attention_states.
*  <b>`output_size`</b>: Size of the output vectors; if None, use output_size.
*  <b>`output_projection`</b>: None or a pair (W, B) of output projection weights and
    biases; W has shape [output_size x num_symbols] and B has shape
    [num_symbols]; if provided and feed_previous=True, each fed previous
    output will first be multiplied by W and added B.
*  <b>`feed_previous`</b>: Boolean; if True, only the first of decoder_inputs will be
    used (the "GO" symbol), and all other decoder inputs will be generated by:
      next = embedding_lookup(embedding, argmax(previous_output)),
    In effect, this implements a greedy decoder. It can also be used
    during training to emulate http://arxiv.org/abs/1506.03099.
    If False, decoder_inputs are used as given (the standard decoder case).
*  <b>`update_embedding_for_previous`</b>: Boolean; if False and feed_previous=True,
    only the embedding for the first symbol of decoder_inputs (the "GO"
    symbol) will be updated by back propagation. Embeddings for the symbols
    generated from the decoder itself remain unchanged. This parameter has
    no effect if feed_previous=False.
*  <b>`dtype`</b>: The dtype to use for the RNN initial states (default: tf.float32).
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to
    "embedding_attention_decoder".
*  <b>`initial_state_attention`</b>: If False (default), initial attentions are zero.
    If True, initialize the attentions from the initial state and attention
    states -- useful when we wish to resume decoding from a previously
    stored decoder state and attention states.

##### Returns:

  A tuple of the form (outputs, state), where:

*  <b>`outputs`</b>: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x output_size] containing the generated outputs.
*  <b>`state`</b>: The state of each decoder cell at the final time-step.
      It is a 2D Tensor of shape [batch_size x cell.state_size].

##### Raises:


*  <b>`ValueError`</b>: When output_projection has the wrong shape.


- - -

### `tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell, num_encoder_symbols, num_decoder_symbols, embedding_size, num_heads=1, output_projection=None, feed_previous=False, dtype=None, scope=None, initial_state_attention=False)` {#embedding_attention_seq2seq}

Embedding sequence-to-sequence model with attention.

This model first embeds encoder_inputs by a newly created embedding (of shape
[num_encoder_symbols x input_size]). Then it runs an RNN to encode
embedded encoder_inputs into a state vector. It keeps the outputs of this
RNN at every step to use for attention later. Next, it embeds decoder_inputs
by another newly created embedding (of shape [num_decoder_symbols x
input_size]). Then it runs attention decoder, initialized with the last
encoder state, on embedded decoder_inputs and attending to encoder outputs.

Warning: when output_projection is None, the size of the attention vectors
and variables will be made proportional to num_decoder_symbols, can be large.

##### Args:


*  <b>`encoder_inputs`</b>: A list of 1D int32 Tensors of shape [batch_size].
*  <b>`decoder_inputs`</b>: A list of 1D int32 Tensors of shape [batch_size].
*  <b>`cell`</b>: core_rnn_cell.RNNCell defining the cell function and size.
*  <b>`num_encoder_symbols`</b>: Integer; number of symbols on the encoder side.
*  <b>`num_decoder_symbols`</b>: Integer; number of symbols on the decoder side.
*  <b>`embedding_size`</b>: Integer, the length of the embedding vector for each symbol.
*  <b>`num_heads`</b>: Number of attention heads that read from attention_states.
*  <b>`output_projection`</b>: None or a pair (W, B) of output projection weights and
    biases; W has shape [output_size x num_decoder_symbols] and B has
    shape [num_decoder_symbols]; if provided and feed_previous=True, each
    fed previous output will first be multiplied by W and added B.
*  <b>`feed_previous`</b>: Boolean or scalar Boolean Tensor; if True, only the first
    of decoder_inputs will be used (the "GO" symbol), and all other decoder
    inputs will be taken from previous outputs (as in embedding_rnn_decoder).
    If False, decoder_inputs are used as given (the standard decoder case).
*  <b>`dtype`</b>: The dtype of the initial RNN state (default: tf.float32).
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to
    "embedding_attention_seq2seq".
*  <b>`initial_state_attention`</b>: If False (default), initial attentions are zero.
    If True, initialize the attentions from the initial state and attention
    states.

##### Returns:

  A tuple of the form (outputs, state), where:

*  <b>`outputs`</b>: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x num_decoder_symbols] containing the generated
      outputs.
*  <b>`state`</b>: The state of each decoder cell at the final time-step.
      It is a 2D Tensor of shape [batch_size x cell.state_size].


- - -

### `tf.contrib.legacy_seq2seq.embedding_rnn_decoder(decoder_inputs, initial_state, cell, num_symbols, embedding_size, output_projection=None, feed_previous=False, update_embedding_for_previous=True, scope=None)` {#embedding_rnn_decoder}

RNN decoder with embedding and a pure-decoding option.

##### Args:


*  <b>`decoder_inputs`</b>: A list of 1D batch-sized int32 Tensors (decoder inputs).
*  <b>`initial_state`</b>: 2D Tensor [batch_size x cell.state_size].
*  <b>`cell`</b>: core_rnn_cell.RNNCell defining the cell function.
*  <b>`num_symbols`</b>: Integer, how many symbols come into the embedding.
*  <b>`embedding_size`</b>: Integer, the length of the embedding vector for each symbol.
*  <b>`output_projection`</b>: None or a pair (W, B) of output projection weights and
    biases; W has shape [output_size x num_symbols] and B has
    shape [num_symbols]; if provided and feed_previous=True, each fed
    previous output will first be multiplied by W and added B.
*  <b>`feed_previous`</b>: Boolean; if True, only the first of decoder_inputs will be
    used (the "GO" symbol), and all other decoder inputs will be generated by:
      next = embedding_lookup(embedding, argmax(previous_output)),
    In effect, this implements a greedy decoder. It can also be used
    during training to emulate http://arxiv.org/abs/1506.03099.
    If False, decoder_inputs are used as given (the standard decoder case).
*  <b>`update_embedding_for_previous`</b>: Boolean; if False and feed_previous=True,
    only the embedding for the first symbol of decoder_inputs (the "GO"
    symbol) will be updated by back propagation. Embeddings for the symbols
    generated from the decoder itself remain unchanged. This parameter has
    no effect if feed_previous=False.
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to
    "embedding_rnn_decoder".

##### Returns:

  A tuple of the form (outputs, state), where:

*  <b>`outputs`</b>: A list of the same length as decoder_inputs of 2D Tensors. The
      output is of shape [batch_size x cell.output_size] when
      output_projection is not None (and represents the dense representation
      of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
      when output_projection is None.
*  <b>`state`</b>: The state of each decoder cell in each time-step. This is a list
      with length len(decoder_inputs) -- one item for each time-step.
      It is a 2D Tensor of shape [batch_size x cell.state_size].

##### Raises:


*  <b>`ValueError`</b>: When output_projection has the wrong shape.


- - -

### `tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, num_encoder_symbols, num_decoder_symbols, embedding_size, output_projection=None, feed_previous=False, dtype=None, scope=None)` {#embedding_rnn_seq2seq}

Embedding RNN sequence-to-sequence model.

This model first embeds encoder_inputs by a newly created embedding (of shape
[num_encoder_symbols x input_size]). Then it runs an RNN to encode
embedded encoder_inputs into a state vector. Next, it embeds decoder_inputs
by another newly created embedding (of shape [num_decoder_symbols x
input_size]). Then it runs RNN decoder, initialized with the last
encoder state, on embedded decoder_inputs.

##### Args:


*  <b>`encoder_inputs`</b>: A list of 1D int32 Tensors of shape [batch_size].
*  <b>`decoder_inputs`</b>: A list of 1D int32 Tensors of shape [batch_size].
*  <b>`cell`</b>: core_rnn_cell.RNNCell defining the cell function and size.
*  <b>`num_encoder_symbols`</b>: Integer; number of symbols on the encoder side.
*  <b>`num_decoder_symbols`</b>: Integer; number of symbols on the decoder side.
*  <b>`embedding_size`</b>: Integer, the length of the embedding vector for each symbol.
*  <b>`output_projection`</b>: None or a pair (W, B) of output projection weights and
    biases; W has shape [output_size x num_decoder_symbols] and B has
    shape [num_decoder_symbols]; if provided and feed_previous=True, each
    fed previous output will first be multiplied by W and added B.
*  <b>`feed_previous`</b>: Boolean or scalar Boolean Tensor; if True, only the first
    of decoder_inputs will be used (the "GO" symbol), and all other decoder
    inputs will be taken from previous outputs (as in embedding_rnn_decoder).
    If False, decoder_inputs are used as given (the standard decoder case).
*  <b>`dtype`</b>: The dtype of the initial state for both the encoder and encoder
    rnn cells (default: tf.float32).
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to
    "embedding_rnn_seq2seq"

##### Returns:

  A tuple of the form (outputs, state), where:

*  <b>`outputs`</b>: A list of the same length as decoder_inputs of 2D Tensors. The
      output is of shape [batch_size x cell.output_size] when
      output_projection is not None (and represents the dense representation
      of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
      when output_projection is None.
*  <b>`state`</b>: The state of each decoder cell in each time-step. This is a list
      with length len(decoder_inputs) -- one item for each time-step.
      It is a 2D Tensor of shape [batch_size x cell.state_size].


- - -

### `tf.contrib.legacy_seq2seq.embedding_tied_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, num_symbols, embedding_size, num_decoder_symbols=None, output_projection=None, feed_previous=False, dtype=None, scope=None)` {#embedding_tied_rnn_seq2seq}

Embedding RNN sequence-to-sequence model with tied (shared) parameters.

This model first embeds encoder_inputs by a newly created embedding (of shape
[num_symbols x input_size]). Then it runs an RNN to encode embedded
encoder_inputs into a state vector. Next, it embeds decoder_inputs using
the same embedding. Then it runs RNN decoder, initialized with the last
encoder state, on embedded decoder_inputs. The decoder output is over symbols
from 0 to num_decoder_symbols - 1 if num_decoder_symbols is none; otherwise it
is over 0 to num_symbols - 1.

##### Args:


*  <b>`encoder_inputs`</b>: A list of 1D int32 Tensors of shape [batch_size].
*  <b>`decoder_inputs`</b>: A list of 1D int32 Tensors of shape [batch_size].
*  <b>`cell`</b>: core_rnn_cell.RNNCell defining the cell function and size.
*  <b>`num_symbols`</b>: Integer; number of symbols for both encoder and decoder.
*  <b>`embedding_size`</b>: Integer, the length of the embedding vector for each symbol.
*  <b>`num_decoder_symbols`</b>: Integer; number of output symbols for decoder. If
    provided, the decoder output is over symbols 0 to num_decoder_symbols - 1.
    Otherwise, decoder output is over symbols 0 to num_symbols - 1. Note that
    this assumes that the vocabulary is set up such that the first
    num_decoder_symbols of num_symbols are part of decoding.
*  <b>`output_projection`</b>: None or a pair (W, B) of output projection weights and
    biases; W has shape [output_size x num_symbols] and B has
    shape [num_symbols]; if provided and feed_previous=True, each
    fed previous output will first be multiplied by W and added B.
*  <b>`feed_previous`</b>: Boolean or scalar Boolean Tensor; if True, only the first
    of decoder_inputs will be used (the "GO" symbol), and all other decoder
    inputs will be taken from previous outputs (as in embedding_rnn_decoder).
    If False, decoder_inputs are used as given (the standard decoder case).
*  <b>`dtype`</b>: The dtype to use for the initial RNN states (default: tf.float32).
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to
    "embedding_tied_rnn_seq2seq".

##### Returns:

  A tuple of the form (outputs, state), where:

*  <b>`outputs`</b>: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x output_symbols] containing the generated
      outputs where output_symbols = num_decoder_symbols if
      num_decoder_symbols is not None otherwise output_symbols = num_symbols.
*  <b>`state`</b>: The state of each decoder cell at the final time-step.
      It is a 2D Tensor of shape [batch_size x cell.state_size].

##### Raises:


*  <b>`ValueError`</b>: When output_projection has the wrong shape.


- - -

### `tf.contrib.legacy_seq2seq.model_with_buckets(encoder_inputs, decoder_inputs, targets, weights, buckets, seq2seq, softmax_loss_function=None, per_example_loss=False, name=None)` {#model_with_buckets}

Create a sequence-to-sequence model with support for bucketing.

The seq2seq argument is a function that defines a sequence-to-sequence model,
e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(
    x, y, core_rnn_cell.GRUCell(24))

##### Args:


*  <b>`encoder_inputs`</b>: A list of Tensors to feed the encoder; first seq2seq input.
*  <b>`decoder_inputs`</b>: A list of Tensors to feed the decoder; second seq2seq input.
*  <b>`targets`</b>: A list of 1D batch-sized int32 Tensors (desired output sequence).
*  <b>`weights`</b>: List of 1D batch-sized float-Tensors to weight the targets.
*  <b>`buckets`</b>: A list of pairs of (input size, output size) for each bucket.
*  <b>`seq2seq`</b>: A sequence-to-sequence model function; it takes 2 input that
    agree with encoder_inputs and decoder_inputs, and returns a pair
    consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
*  <b>`softmax_loss_function`</b>: Function (inputs-batch, labels-batch) -> loss-batch
    to be used instead of the standard softmax (the default if this is None).
*  <b>`per_example_loss`</b>: Boolean. If set, the returned loss will be a batch-sized
    tensor of losses for each sequence in the batch. If unset, it will be
    a scalar with the averaged loss from all examples.
*  <b>`name`</b>: Optional name for this operation, defaults to "model_with_buckets".

##### Returns:

  A tuple of the form (outputs, losses), where:

*  <b>`outputs`</b>: The outputs for each bucket. Its j'th element consists of a list
      of 2D Tensors. The shape of output tensors can be either
      [batch_size x output_size] or [batch_size x num_decoder_symbols]
      depending on the seq2seq model used.
*  <b>`losses`</b>: List of scalar Tensors, representing losses for each bucket, or,
      if per_example_loss is set, a list of 1D batch-sized float Tensors.

##### Raises:


*  <b>`ValueError`</b>: If length of encoder_inputsut, targets, or weights is smaller
    than the largest (last) bucket.


- - -

### `tf.contrib.legacy_seq2seq.one2many_rnn_seq2seq(encoder_inputs, decoder_inputs_dict, cell, num_encoder_symbols, num_decoder_symbols_dict, embedding_size, feed_previous=False, dtype=None, scope=None)` {#one2many_rnn_seq2seq}

One-to-many RNN sequence-to-sequence model (multi-task).

This is a multi-task sequence-to-sequence model with one encoder and multiple
decoders. Reference to multi-task sequence-to-sequence learning can be found
here: http://arxiv.org/abs/1511.06114

##### Args:


*  <b>`encoder_inputs`</b>: A list of 1D int32 Tensors of shape [batch_size].
*  <b>`decoder_inputs_dict`</b>: A dictionany mapping decoder name (string) to
    the corresponding decoder_inputs; each decoder_inputs is a list of 1D
    Tensors of shape [batch_size]; num_decoders is defined as
    len(decoder_inputs_dict).
*  <b>`cell`</b>: core_rnn_cell.RNNCell defining the cell function and size.
*  <b>`num_encoder_symbols`</b>: Integer; number of symbols on the encoder side.
*  <b>`num_decoder_symbols_dict`</b>: A dictionary mapping decoder name (string) to an
    integer specifying number of symbols for the corresponding decoder;
    len(num_decoder_symbols_dict) must be equal to num_decoders.
*  <b>`embedding_size`</b>: Integer, the length of the embedding vector for each symbol.
*  <b>`feed_previous`</b>: Boolean or scalar Boolean Tensor; if True, only the first of
    decoder_inputs will be used (the "GO" symbol), and all other decoder
    inputs will be taken from previous outputs (as in embedding_rnn_decoder).
    If False, decoder_inputs are used as given (the standard decoder case).
*  <b>`dtype`</b>: The dtype of the initial state for both the encoder and encoder
    rnn cells (default: tf.float32).
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to
    "one2many_rnn_seq2seq"

##### Returns:

  A tuple of the form (outputs_dict, state_dict), where:

*  <b>`outputs_dict`</b>: A mapping from decoder name (string) to a list of the same
      length as decoder_inputs_dict[name]; each element in the list is a 2D
      Tensors with shape [batch_size x num_decoder_symbol_list[name]]
      containing the generated outputs.
*  <b>`state_dict`</b>: A mapping from decoder name (string) to the final state of the
      corresponding decoder RNN; it is a 2D Tensor of shape
      [batch_size x cell.state_size].


- - -

### `tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None, scope=None)` {#rnn_decoder}

RNN decoder for the sequence-to-sequence model.

##### Args:


*  <b>`decoder_inputs`</b>: A list of 2D Tensors [batch_size x input_size].
*  <b>`initial_state`</b>: 2D Tensor with shape [batch_size x cell.state_size].
*  <b>`cell`</b>: core_rnn_cell.RNNCell defining the cell function and size.
*  <b>`loop_function`</b>: If not None, this function will be applied to the i-th output
    in order to generate the i+1-st input, and decoder_inputs will be ignored,
    except for the first element ("GO" symbol). This can be used for decoding,
    but also for training to emulate http://arxiv.org/abs/1506.03099.
    Signature -- loop_function(prev, i) = next
      * prev is a 2D Tensor of shape [batch_size x output_size],
      * i is an integer, the step number (when advanced control is needed),
      * next is a 2D Tensor of shape [batch_size x input_size].
*  <b>`scope`</b>: VariableScope for the created subgraph; defaults to "rnn_decoder".

##### Returns:

  A tuple of the form (outputs, state), where:

*  <b>`outputs`</b>: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x output_size] containing generated outputs.
*  <b>`state`</b>: The state of each cell at the final time-step.
      It is a 2D Tensor of shape [batch_size x cell.state_size].
      (Note that in some cases, like basic RNN cell or GRU cell, outputs and
       states can be the same. They are different for LSTM cells though.)


- - -

### `tf.contrib.legacy_seq2seq.sequence_loss(logits, targets, weights, average_across_timesteps=True, average_across_batch=True, softmax_loss_function=None, name=None)` {#sequence_loss}

Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

##### Args:


*  <b>`logits`</b>: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
*  <b>`targets`</b>: List of 1D batch-sized int32 Tensors of the same length as logits.
*  <b>`weights`</b>: List of 1D batch-sized float-Tensors of the same length as logits.
*  <b>`average_across_timesteps`</b>: If set, divide the returned cost by the total
    label weight.
*  <b>`average_across_batch`</b>: If set, divide the returned cost by the batch size.
*  <b>`softmax_loss_function`</b>: Function (inputs-batch, labels-batch) -> loss-batch
    to be used instead of the standard softmax (the default if this is None).
*  <b>`name`</b>: Optional name for this operation, defaults to "sequence_loss".

##### Returns:

  A scalar float Tensor: The average log-perplexity per symbol (weighted).

##### Raises:


*  <b>`ValueError`</b>: If len(logits) is different from len(targets) or len(weights).


- - -

### `tf.contrib.legacy_seq2seq.sequence_loss_by_example(logits, targets, weights, average_across_timesteps=True, softmax_loss_function=None, name=None)` {#sequence_loss_by_example}

Weighted cross-entropy loss for a sequence of logits (per example).

##### Args:


*  <b>`logits`</b>: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
*  <b>`targets`</b>: List of 1D batch-sized int32 Tensors of the same length as logits.
*  <b>`weights`</b>: List of 1D batch-sized float-Tensors of the same length as logits.
*  <b>`average_across_timesteps`</b>: If set, divide the returned cost by the total
    label weight.
*  <b>`softmax_loss_function`</b>: Function (labels-batch, inputs-batch) -> loss-batch
    to be used instead of the standard softmax (the default if this is None).
*  <b>`name`</b>: Optional name for this operation, default: "sequence_loss_by_example".

##### Returns:

  1D batch-sized float Tensor: The log-perplexity for each sequence.

##### Raises:


*  <b>`ValueError`</b>: If len(logits) is different from len(targets) or len(weights).


- - -

### `tf.contrib.legacy_seq2seq.tied_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, loop_function=None, dtype=tf.float32, scope=None)` {#tied_rnn_seq2seq}

RNN sequence-to-sequence model with tied encoder and decoder parameters.

This model first runs an RNN to encode encoder_inputs into a state vector, and
then runs decoder, initialized with the last encoder state, on decoder_inputs.
Encoder and decoder use the same RNN cell and share parameters.

##### Args:


*  <b>`encoder_inputs`</b>: A list of 2D Tensors [batch_size x input_size].
*  <b>`decoder_inputs`</b>: A list of 2D Tensors [batch_size x input_size].
*  <b>`cell`</b>: core_rnn_cell.RNNCell defining the cell function and size.
*  <b>`loop_function`</b>: If not None, this function will be applied to i-th output
    in order to generate i+1-th input, and decoder_inputs will be ignored,
    except for the first element ("GO" symbol), see rnn_decoder for details.
*  <b>`dtype`</b>: The dtype of the initial state of the rnn cell (default: tf.float32).
*  <b>`scope`</b>: VariableScope for the created subgraph; default: "tied_rnn_seq2seq".

##### Returns:

  A tuple of the form (outputs, state), where:

*  <b>`outputs`</b>: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x output_size] containing the generated outputs.
*  <b>`state`</b>: The state of each decoder cell in each time-step. This is a list
      with length len(decoder_inputs) -- one item for each time-step.
      It is a 2D Tensor of shape [batch_size x cell.state_size].


