"""Library for creating sequence-to-sequence models."""

import tensorflow.python.platform

import tensorflow as tf

from tensorflow.models.rnn import linear
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell


def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None,
                scope=None):
  """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: if not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/pdf/1506.03099v2.pdf.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x cell.output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x cell.input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x cell.output_size] containing generated outputs.
    states: The state of each cell in each time-step. This is a list with
      length len(decoder_inputs) -- one item for each time-step.
      Each item is a 2D Tensor of shape [batch_size x cell.state_size].
      (Note that in some cases, like basic RNN cell or GRU cell, outputs and
       states can be the same. They are different for LSTM cells though.)
  """
  with tf.variable_scope(scope or "rnn_decoder"):
    states = [initial_state]
    outputs = []
    prev = None
    for i in xrange(len(decoder_inputs)):
      inp = decoder_inputs[i]
      if loop_function is not None and prev is not None:
        with tf.variable_scope("loop_function", reuse=True):
          # We do not propagate gradients over the loop function.
          inp = tf.stop_gradient(loop_function(prev, i))
      if i > 0:
        tf.get_variable_scope().reuse_variables()
      output, new_state = cell(inp, states[-1])
      outputs.append(output)
      states.append(new_state)
      if loop_function is not None:
        prev = tf.stop_gradient(output)
  return outputs, states


def basic_rnn_seq2seq(
    encoder_inputs, decoder_inputs, cell, dtype=tf.float32, scope=None):
  """Basic RNN sequence-to-sequence model.

  This model first runs an RNN to encode encoder_inputs into a state vector, and
  then runs decoder, initialized with the last encoder state, on decoder_inputs.
  Encoder and decoder use the same RNN cell type, but don't share parameters.

  Args:
    encoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
    decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x cell.output_size] containing the generated outputs.
    states: The state of each decoder cell in each time-step. This is a list
      with length len(decoder_inputs) -- one item for each time-step.
      Each item is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with tf.variable_scope(scope or "basic_rnn_seq2seq"):
    _, enc_states = rnn.rnn(cell, encoder_inputs, dtype=dtype)
    return rnn_decoder(decoder_inputs, enc_states[-1], cell)


def tied_rnn_seq2seq(encoder_inputs, decoder_inputs, cell,
                     loop_function=None, dtype=tf.float32, scope=None):
  """RNN sequence-to-sequence model with tied encoder and decoder parameters.

  This model first runs an RNN to encode encoder_inputs into a state vector, and
  then runs decoder, initialized with the last encoder state, on decoder_inputs.
  Encoder and decoder use the same RNN cell and share parameters.

  Args:
    encoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
    decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: if not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol), see rnn_decoder for details.
    dtype: The dtype of the initial state of the rnn cell (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "tied_rnn_seq2seq".

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x cell.output_size] containing the generated outputs.
    states: The state of each decoder cell in each time-step. This is a list
      with length len(decoder_inputs) -- one item for each time-step.
      Each item is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with tf.variable_scope("combined_tied_rnn_seq2seq"):
    scope = scope or "tied_rnn_seq2seq"
    _, enc_states = rnn.rnn(
        cell, encoder_inputs, dtype=dtype, scope=scope)
    tf.get_variable_scope().reuse_variables()
    return rnn_decoder(decoder_inputs, enc_states[-1], cell,
                       loop_function=loop_function, scope=scope)


def embedding_rnn_decoder(decoder_inputs, initial_state, cell, num_symbols,
                          output_projection=None, feed_previous=False,
                          scope=None):
  """RNN decoder with embedding and a pure-decoding option.

  Args:
    decoder_inputs: a list of 1D batch-sized int32-Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function.
    num_symbols: integer, how many symbols come into the embedding.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [cell.output_size x num_symbols] and B has
      shape [num_symbols]; if provided and feed_previous=True, each fed
      previous output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/pdf/1506.03099v2.pdf.
      If False, decoder_inputs are used as given (the standard decoder case).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_decoder".

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x cell.output_size] containing the generated outputs.
    states: The state of each decoder cell in each time-step. This is a list
      with length len(decoder_inputs) -- one item for each time-step.
      Each item is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: when output_projection has the wrong shape.
  """
  if output_projection is not None:
    proj_weights = tf.convert_to_tensor(output_projection[0], dtype=tf.float32)
    proj_weights.get_shape().assert_is_compatible_with([cell.output_size,
                                                        num_symbols])
    proj_biases = tf.convert_to_tensor(output_projection[1], dtype=tf.float32)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with tf.variable_scope(scope or "embedding_rnn_decoder"):
    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [num_symbols, cell.input_size])

    def extract_argmax_and_embed(prev, _):
      """Loop_function that extracts the symbol from prev and embeds it."""
      if output_projection is not None:
        prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
      prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
      return tf.nn.embedding_lookup(embedding, prev_symbol)

    loop_function = None
    if feed_previous:
      loop_function = extract_argmax_and_embed

    emb_inp = [tf.nn.embedding_lookup(embedding, i) for i in decoder_inputs]
    return rnn_decoder(emb_inp, initial_state, cell,
                       loop_function=loop_function)


def embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, cell,
                          num_encoder_symbols, num_decoder_symbols,
                          output_projection=None, feed_previous=False,
                          dtype=tf.float32, scope=None):
  """Embedding RNN sequence-to-sequence model.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x cell.input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  cell.input_size]). Then it runs RNN decoder, initialized with the last
  encoder state, on embedded decoder_inputs.

  Args:
    encoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
    decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: integer; number of symbols on the encoder side.
    num_decoder_symbols: integer; number of symbols on the decoder side.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [cell.output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial state for both the encoder and encoder
      rnn cells (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_seq2seq"

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x num_decoder_symbols] containing the generated outputs.
    states: The state of each decoder cell in each time-step. This is a list
      with length len(decoder_inputs) -- one item for each time-step.
      Each item is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with tf.variable_scope(scope or "embedding_rnn_seq2seq"):
    # Encoder.
    encoder_cell = rnn_cell.EmbeddingWrapper(cell, num_encoder_symbols)
    _, encoder_states = rnn.rnn(encoder_cell, encoder_inputs, dtype=dtype)

    # Decoder.
    if output_projection is None:
      cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)

    if isinstance(feed_previous, bool):
      return embedding_rnn_decoder(decoder_inputs, encoder_states[-1], cell,
                                   num_decoder_symbols, output_projection,
                                   feed_previous)
    else:  # If feed_previous is a Tensor, we construct 2 graphs and use cond.
      outputs1, states1 = embedding_rnn_decoder(
          decoder_inputs, encoder_states[-1], cell, num_decoder_symbols,
          output_projection, True)
      tf.get_variable_scope().reuse_variables()
      outputs2, states2 = embedding_rnn_decoder(
          decoder_inputs, encoder_states[-1], cell, num_decoder_symbols,
          output_projection, False)

      outputs = tf.control_flow_ops.cond(feed_previous,
                                         lambda: outputs1, lambda: outputs2)
      states = tf.control_flow_ops.cond(feed_previous,
                                        lambda: states1, lambda: states2)
      return outputs, states


def embedding_tied_rnn_seq2seq(encoder_inputs, decoder_inputs, cell,
                               num_symbols, output_projection=None,
                               feed_previous=False, dtype=tf.float32,
                               scope=None):
  """Embedding RNN sequence-to-sequence model with tied (shared) parameters.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_symbols x cell.input_size]). Then it runs an RNN to encode embedded
  encoder_inputs into a state vector. Next, it embeds decoder_inputs using
  the same embedding. Then it runs RNN decoder, initialized with the last
  encoder state, on embedded decoder_inputs.

  Args:
    encoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
    decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    num_symbols: integer; number of symbols for both encoder and decoder.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [cell.output_size x num_symbols] and B has
      shape [num_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype to use for the initial RNN states (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_tied_rnn_seq2seq".

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x num_decoder_symbols] containing the generated outputs.
    states: The state of each decoder cell in each time-step. This is a list
      with length len(decoder_inputs) -- one item for each time-step.
      Each item is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: when output_projection has the wrong shape.
  """
  if output_projection is not None:
    proj_weights = tf.convert_to_tensor(output_projection[0], dtype=dtype)
    proj_weights.get_shape().assert_is_compatible_with([cell.output_size,
                                                        num_symbols])
    proj_biases = tf.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with tf.variable_scope(scope or "embedding_tied_rnn_seq2seq"):
    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [num_symbols, cell.input_size])

    emb_encoder_inputs = [tf.nn.embedding_lookup(embedding, x)
                          for x in encoder_inputs]
    emb_decoder_inputs = [tf.nn.embedding_lookup(embedding, x)
                          for x in decoder_inputs]

    def extract_argmax_and_embed(prev, _):
      """Loop_function that extracts the symbol from prev and embeds it."""
      if output_projection is not None:
        prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
      prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
      return tf.nn.embedding_lookup(embedding, prev_symbol)

    if output_projection is None:
      cell = rnn_cell.OutputProjectionWrapper(cell, num_symbols)

    if isinstance(feed_previous, bool):
      loop_function = extract_argmax_and_embed if feed_previous else None
      return tied_rnn_seq2seq(emb_encoder_inputs, emb_decoder_inputs, cell,
                              loop_function=loop_function, dtype=dtype)
    else:  # If feed_previous is a Tensor, we construct 2 graphs and use cond.
      outputs1, states1 = tied_rnn_seq2seq(
          emb_encoder_inputs, emb_decoder_inputs, cell,
          loop_function=extract_argmax_and_embed, dtype=dtype)
      tf.get_variable_scope().reuse_variables()
      outputs2, states2 = tied_rnn_seq2seq(
          emb_encoder_inputs, emb_decoder_inputs, cell, dtype=dtype)

      outputs = tf.control_flow_ops.cond(feed_previous,
                                         lambda: outputs1, lambda: outputs2)
      states = tf.control_flow_ops.cond(feed_previous,
                                        lambda: states1, lambda: states2)
      return outputs, states


def attention_decoder(decoder_inputs, initial_state, attention_states, cell,
                      output_size=None, num_heads=1, loop_function=None,
                      dtype=tf.float32, scope=None):
  """RNN decoder with attention for the sequence-to-sequence model.

  Args:
    decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    output_size: size of the output vectors; if None, we use cell.output_size.
    num_heads: number of attention heads that read from attention_states.
    loop_function: if not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/pdf/1506.03099v2.pdf.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x cell.output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x cell.input_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors of shape
      [batch_size x output_size]. These represent the generated outputs.
      Output i is computed from input i (which is either i-th decoder_inputs or
      loop_function(output {i-1}, i)) as follows. First, we run the cell
      on a combination of the input and previous attention masks:
        cell_output, new_state = cell(linear(input, prev_attn), prev_state).
      Then, we calculate new attention masks:
        new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
      and then we calculate the output:
        output = linear(cell_output, new_attn).
    states: The state of each decoder cell in each time-step. This is a list
      with length len(decoder_inputs) -- one item for each time-step.
      Each item is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, or shapes
      of attention_states are not set.
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if not attention_states.get_shape()[1:2].is_fully_defined():
    raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size

  with tf.variable_scope(scope or "attention_decoder"):
    batch_size = tf.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    attn_size = attention_states.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = tf.reshape(attention_states, [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []
    attention_vec_size = attn_size  # Size of query vectors for attention.
    for a in xrange(num_heads):
      k = tf.get_variable("AttnW_%d" % a, [1, 1, attn_size, attention_vec_size])
      hidden_features.append(tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(tf.get_variable("AttnV_%d" % a, [attention_vec_size]))

    states = [initial_state]

    def attention(query):
      """Put attention masks on hidden using hidden_features and query."""
      ds = []  # Results of attention reads will be stored here.
      for a in xrange(num_heads):
        with tf.variable_scope("Attention_%d" % a):
          y = linear.linear(query, attention_vec_size, True)
          y = tf.reshape(y, [-1, 1, 1, attention_vec_size])
          # Attention mask is a softmax of v^T * tanh(...).
          s = tf.reduce_sum(v[a] * tf.tanh(hidden_features[a] + y), [2, 3])
          a = tf.nn.softmax(s)
          # Now calculate the attention-weighted vector d.
          d = tf.reduce_sum(tf.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                            [1, 2])
          ds.append(tf.reshape(d, [-1, attn_size]))
      return ds

    outputs = []
    prev = None
    batch_attn_size = tf.pack([batch_size, attn_size])
    attns = [tf.zeros(batch_attn_size, dtype=dtype)
             for _ in xrange(num_heads)]
    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size])
    for i in xrange(len(decoder_inputs)):
      if i > 0:
        tf.get_variable_scope().reuse_variables()
      inp = decoder_inputs[i]
      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and prev is not None:
        with tf.variable_scope("loop_function", reuse=True):
          inp = tf.stop_gradient(loop_function(prev, i))
      # Merge input and previous attentions into one vector of the right size.
      x = linear.linear([inp] + attns, cell.input_size, True)
      # Run the RNN.
      cell_output, new_state = cell(x, states[-1])
      states.append(new_state)
      # Run the attention mechanism.
      attns = attention(new_state)
      with tf.variable_scope("AttnOutputProjection"):
        output = linear.linear([cell_output] + attns, output_size, True)
      if loop_function is not None:
        # We do not propagate gradients over the loop function.
        prev = tf.stop_gradient(output)
      outputs.append(output)

  return outputs, states


def embedding_attention_decoder(decoder_inputs, initial_state, attention_states,
                                cell, num_symbols, num_heads=1,
                                output_size=None, output_projection=None,
                                feed_previous=False, dtype=tf.float32,
                                scope=None):
  """RNN decoder with embedding and attention and a pure-decoding option.

  Args:
    decoder_inputs: a list of 1D batch-sized int32-Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function.
    num_symbols: integer, how many symbols come into the embedding.
    num_heads: number of attention heads that read from attention_states.
    output_size: size of the output vectors; if None, use cell.output_size.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has shape
      [num_symbols]; if provided and feed_previous=True, each fed previous
      output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/pdf/1506.03099v2.pdf.
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype to use for the RNN initial states (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_decoder".

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x output_size] containing the generated outputs.
    states: The state of each decoder cell in each time-step. This is a list
      with length len(decoder_inputs) -- one item for each time-step.
      Each item is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: when output_projection has the wrong shape.
  """
  if output_size is None:
    output_size = cell.output_size
  if output_projection is not None:
    proj_weights = tf.convert_to_tensor(output_projection[0], dtype=dtype)
    proj_weights.get_shape().assert_is_compatible_with([cell.output_size,
                                                        num_symbols])
    proj_biases = tf.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with tf.variable_scope(scope or "embedding_attention_decoder"):
    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [num_symbols, cell.input_size])

    def extract_argmax_and_embed(prev, _):
      """Loop_function that extracts the symbol from prev and embeds it."""
      if output_projection is not None:
        prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
      prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
      emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
      return emb_prev

    loop_function = None
    if feed_previous:
      loop_function = extract_argmax_and_embed

    emb_inp = [tf.nn.embedding_lookup(embedding, i) for i in decoder_inputs]
    return attention_decoder(
        emb_inp, initial_state, attention_states, cell, output_size=output_size,
        num_heads=num_heads, loop_function=loop_function)


def embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell,
                                num_encoder_symbols, num_decoder_symbols,
                                num_heads=1, output_projection=None,
                                feed_previous=False, dtype=tf.float32,
                                scope=None):
  """Embedding sequence-to-sequence model with attention.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x cell.input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. It keeps the outputs of this
  RNN at every step to use for attention later. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  cell.input_size]). Then it runs attention decoder, initialized with the last
  encoder state, on embedded decoder_inputs and attending to encoder outputs.

  Args:
    encoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
    decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: integer; number of symbols on the encoder side.
    num_decoder_symbols: integer; number of symbols on the decoder side.
    num_heads: number of attention heads that read from attention_states.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [cell.output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial RNN state (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_seq2seq".

  Returns:
    outputs: A list of the same length as decoder_inputs of 2D Tensors with
      shape [batch_size x num_decoder_symbols] containing the generated outputs.
    states: The state of each decoder cell in each time-step. This is a list
      with length len(decoder_inputs) -- one item for each time-step.
      Each item is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with tf.variable_scope(scope or "embedding_attention_seq2seq"):
    # Encoder.
    encoder_cell = rnn_cell.EmbeddingWrapper(cell, num_encoder_symbols)
    encoder_outputs, encoder_states = rnn.rnn(
        encoder_cell, encoder_inputs, dtype=dtype)

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [tf.reshape(e, [-1, 1, cell.output_size])
                  for e in encoder_outputs]
    attention_states = tf.concat(1, top_states)

    # Decoder.
    output_size = None
    if output_projection is None:
      cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
      output_size = num_decoder_symbols

    if isinstance(feed_previous, bool):
      return embedding_attention_decoder(
          decoder_inputs, encoder_states[-1], attention_states, cell,
          num_decoder_symbols, num_heads, output_size, output_projection,
          feed_previous)
    else:  # If feed_previous is a Tensor, we construct 2 graphs and use cond.
      outputs1, states1 = embedding_attention_decoder(
          decoder_inputs, encoder_states[-1], attention_states, cell,
          num_decoder_symbols, num_heads, output_size, output_projection, True)
      tf.get_variable_scope().reuse_variables()
      outputs2, states2 = embedding_attention_decoder(
          decoder_inputs, encoder_states[-1], attention_states, cell,
          num_decoder_symbols, num_heads, output_size, output_projection, False)

      outputs = tf.control_flow_ops.cond(feed_previous,
                                         lambda: outputs1, lambda: outputs2)
      states = tf.control_flow_ops.cond(feed_previous,
                                        lambda: states1, lambda: states2)
      return outputs, states


def sequence_loss_by_example(logits, targets, weights, num_decoder_symbols,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: list of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: list of 1D batch-sized int32-Tensors of the same length as logits.
    weights: list of 1D batch-sized float-Tensors of the same length as logits.
    num_decoder_symbols: integer, number of decoder symbols (output classes).
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: the log-perplexity for each sequence.

  Raises:
    ValueError: if len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with tf.op_scope(logits + targets + weights, name,
                   "sequence_loss_by_example"):
    batch_size = tf.shape(targets[0])[0]
    log_perp_list = []
    length = batch_size * num_decoder_symbols
    for i in xrange(len(logits)):
      if softmax_loss_function is None:
        # TODO(lukaszkaiser): There is no SparseCrossEntropy in TensorFlow, so
        # we need to first cast targets into a dense representation, and as
        # SparseToDense does not accept batched inputs, we need to do this by
        # re-indexing and re-sizing. When TensorFlow adds SparseCrossEntropy,
        # rewrite this method.
        indices = targets[i] + num_decoder_symbols * tf.range(0, batch_size)
        with tf.device("/cpu:0"):  # Sparse-to-dense must happen on CPU for now.
          dense = tf.sparse_to_dense(indices, tf.expand_dims(length, 0), 1.0,
                                     0.0)
        target = tf.reshape(dense, [-1, num_decoder_symbols])
        crossent = tf.nn.softmax_cross_entropy_with_logits(
            logits[i], target, name="SequenceLoss/CrossEntropy{0}".format(i))
      else:
        crossent = softmax_loss_function(logits[i], targets[i])
      log_perp_list.append(crossent * weights[i])
    log_perps = tf.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = tf.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss(logits, targets, weights, num_decoder_symbols,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: list of 2D Tensors os shape [batch_size x num_decoder_symbols].
    targets: list of 1D batch-sized int32-Tensors of the same length as logits.
    weights: list of 1D batch-sized float-Tensors of the same length as logits.
    num_decoder_symbols: integer, number of decoder symbols (output classes).
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: the average log-perplexity per symbol (weighted).

  Raises:
    ValueError: if len(logits) is different from len(targets) or len(weights).
  """
  with tf.op_scope(logits + targets + weights, name, "sequence_loss"):
    cost = tf.reduce_sum(sequence_loss_by_example(
        logits, targets, weights, num_decoder_symbols,
        average_across_timesteps=average_across_timesteps,
        softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = tf.shape(targets[0])[0]
      return cost / tf.cast(batch_size, tf.float32)
    else:
      return cost


def model_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                       buckets, num_decoder_symbols, seq2seq,
                       softmax_loss_function=None, name=None):
  """Create a sequence-to-sequence model with support for bucketing.

  The seq2seq argument is a function that defines a sequence-to-sequence model,
  e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))

  Args:
    encoder_inputs: a list of Tensors to feed the encoder; first seq2seq input.
    decoder_inputs: a list of Tensors to feed the decoder; second seq2seq input.
    targets: a list of 1D batch-sized int32-Tensors (desired output sequence).
    weights: list of 1D batch-sized float-Tensors to weight the targets.
    buckets: a list of pairs of (input size, output size) for each bucket.
    num_decoder_symbols: integer, number of decoder symbols (output classes).
    seq2seq: a sequence-to-sequence model function; it takes 2 input that
      agree with encoder_inputs and decoder_inputs, and returns a pair
      consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
    softmax_loss_function: function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: optional name for this operation, defaults to "model_with_buckets".

  Returns:
    outputs: The outputs for each bucket. Its j'th element consists of a list
      of 2D Tensors of shape [batch_size x num_decoder_symbols] (j'th outputs).
    losses: List of scalar Tensors, representing losses for each bucket.
  Raises:
    ValueError: if length of encoder_inputsut, targets, or weights is smaller
      than the largest (last) bucket.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  with tf.op_scope(all_inputs, name, "model_with_buckets"):
    for j in xrange(len(buckets)):
      if j > 0:
        tf.get_variable_scope().reuse_variables()
      bucket_encoder_inputs = [encoder_inputs[i]
                               for i in xrange(buckets[j][0])]
      bucket_decoder_inputs = [decoder_inputs[i]
                               for i in xrange(buckets[j][1])]
      bucket_outputs, _ = seq2seq(bucket_encoder_inputs,
                                  bucket_decoder_inputs)
      outputs.append(bucket_outputs)

      bucket_targets = [targets[i] for i in xrange(buckets[j][1])]
      bucket_weights = [weights[i] for i in xrange(buckets[j][1])]
      losses.append(sequence_loss(
          outputs[-1], bucket_targets, bucket_weights, num_decoder_symbols,
          softmax_loss_function=softmax_loss_function))

  return outputs, losses
