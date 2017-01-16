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

