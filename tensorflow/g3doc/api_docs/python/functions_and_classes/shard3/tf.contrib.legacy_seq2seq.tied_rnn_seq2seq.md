### `tf.contrib.legacy_seq2seq.tied_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, loop_function=None, dtype=tf.float32, scope=None)` {#tied_rnn_seq2seq}

RNN sequence-to-sequence model with tied encoder and decoder parameters.

This model first runs an RNN to encode encoder_inputs into a state vector, and
then runs decoder, initialized with the last encoder state, on decoder_inputs.
Encoder and decoder use the same RNN cell and share parameters.

##### Args:


*  <b>`encoder_inputs`</b>: A list of 2D Tensors [batch_size x input_size].
*  <b>`decoder_inputs`</b>: A list of 2D Tensors [batch_size x input_size].
*  <b>`cell`</b>: rnn_cell.RNNCell defining the cell function and size.
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

