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

