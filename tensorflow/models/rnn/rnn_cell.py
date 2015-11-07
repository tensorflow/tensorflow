"""Module for constructing RNN Cells."""

import math

import tensorflow as tf

from tensorflow.models.rnn import linear


class RNNCell(object):
  """Abstract object representing an RNN cell.

  An RNN cell, in the most abstract setting, is anything that has
  a state -- a vector of floats of size self.state_size -- and performs some
  operation that takes inputs of size self.input_size. This operation
  results in an output of size self.output_size and a new state.

  This module provides a number of basic commonly used RNN cells, such as
  LSTM (Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number
  of operators that allow add dropouts, projections, or embeddings for inputs.
  Constructing multi-layer cells is supported by a super-class, MultiRNNCell,
  defined later. Every RNNCell must have the properties below and and
  implement __call__ with the following signature.
  """

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: 2D Tensor with shape [batch_size x self.input_size].
      state: 2D Tensor with shape [batch_size x self.state_size].
      scope: VariableScope for the created subgraph; defaults to class name.

    Returns:
      A pair containing:
      - Output: A 2D Tensor with shape [batch_size x self.output_size]
      - New state: A 2D Tensor with shape [batch_size x self.state_size].
    """
    raise NotImplementedError("Abstract method")

  @property
  def input_size(self):
    """Integer: size of inputs accepted by this cell."""
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  @property
  def state_size(self):
    """Integer: size of state used by this cell."""
    raise NotImplementedError("Abstract method")

  def zero_state(self, batch_size, dtype):
    """Return state tensor (shape [batch_size x state_size]) filled with 0.

    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.

    Returns:
      A 2D Tensor of shape [batch_size x state_size] filled with zeros.
    """
    zeros = tf.zeros(tf.pack([batch_size, self.state_size]), dtype=dtype)
    # The reshape below is a no-op, but it allows shape inference of shape[1].
    return tf.reshape(zeros, [-1, self.state_size])


class BasicRNNCell(RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units):
    self._num_units = num_units

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = tanh(W * input + U * state + B)."""
    with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
      output = tf.tanh(linear.linear([inputs, state], self._num_units, True))
    return output, output


class GRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units):
    self._num_units = num_units

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with tf.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not udpate.
        r, u = tf.split(1, 2, linear.linear([inputs, state],
                                            2 * self._num_units, True, 1.0))
        r, u = tf.sigmoid(r), tf.sigmoid(u)
      with tf.variable_scope("Candidate"):
        c = tf.tanh(linear.linear([inputs, r * state], self._num_units, True))
      new_h = u * state + (1 - u) * c
    return new_h, new_h


class BasicLSTMCell(RNNCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/pdf/1409.2329v5.pdf.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  Biases of the forget gate are initialized by default to 1 in order to reduce
  the scale of forgetting in the beginning of the training.
  """

  def __init__(self, num_units, forget_bias=1.0):
    self._num_units = num_units
    self._forget_bias = forget_bias

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return 2 * self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      c, h = tf.split(1, 2, state)
      concat = linear.linear([inputs, h], 4 * self._num_units, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(1, 4, concat)

      new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
      new_h = tf.tanh(new_c) * tf.sigmoid(o)

    return new_h, tf.concat(1, [new_c, new_h])


class LSTMCell(RNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.

  This implementation is based on:

    https://research.google.com/pubs/archive/43905.pdf

  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.

  It uses peep-hole connections, optional cell clipping, and an optional
  projection layer.
  """

  def __init__(self, num_units, input_size,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None,
               num_unit_shards=1, num_proj_shards=1):
    """Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell
      input_size: int, The dimensionality of the inputs into the LSTM cell
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      num_unit_shards: How to split the weight matrix.  If >1, the weight
        matrix is stored across num_unit_shards.
        Note that num_unit_shards must evenly divide num_units * 4.
      num_proj_shards: How to split the projection matrix.  If >1, the
        projection matrix is stored across num_proj_shards.
        Note that num_proj_shards must evenly divide num_proj
              (if num_proj is not None).

    Raises:
      ValueError: if num_unit_shards doesn't divide 4 * num_units or
        num_proj_shards doesn't divide num_proj
    """
    self._num_units = num_units
    self._input_size = input_size
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards

    if (num_units * 4) % num_unit_shards != 0:
      raise ValueError("num_unit_shards must evently divide 4 * num_units")
    if num_proj and num_proj % num_proj_shards != 0:
      raise ValueError("num_proj_shards must evently divide num_proj")

    if num_proj:
      self._state_size = num_units + num_proj
      self._output_size = num_proj
    else:
      self._state_size = 2 * num_units
      self._output_size = num_units

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def __call__(self, input_, state, scope=None):
    """Run one step of LSTM.

    Args:
      input_: input Tensor, 2D, batch x num_units.
      state: state Tensor, 2D, batch x state_size.
      scope: VariableScope for the created subgraph; defaults to "LSTMCell".

    Returns:
      A tuple containing:
      - A 2D, batch x output_dim, Tensor representing the output of the LSTM
        after reading "input_" when previous state was "state".
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - A 2D, batch x state_size, Tensor representing the new state of LSTM
        after reading "input_" when previous state was "state".
    """
    num_proj = self._num_units if self._num_proj is None else self._num_proj

    c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
    m_prev = tf.slice(state, [0, self._num_units], [-1, num_proj])

    dtype = input_.dtype

    unit_shard_size = (4 * self._num_units) / self._num_unit_shards

    with tf.variable_scope(scope or type(self).__name__):  # "LSTMCell"
      w = tf.concat(
          1, [tf.get_variable("W_%d" % i,
                              shape=[self.input_size + num_proj,
                                     unit_shard_size],
                              initializer=self._initializer,
                              dtype=dtype)
              for i in range(self._num_unit_shards)])

      b = tf.get_variable(
          "B", shape=[4 * self._num_units],
          initializer=tf.zeros_initializer, dtype=dtype)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      cell_inputs = tf.concat(1, [input_, m_prev])
      i, j, f, o = tf.split(1, 4, tf.nn.bias_add(tf.matmul(cell_inputs, w), b))

      # Diagonal connections
      if self._use_peepholes:
        w_f_diag = tf.get_variable(
            "W_F_diag", shape=[self._num_units], dtype=dtype)
        w_i_diag = tf.get_variable(
            "W_I_diag", shape=[self._num_units], dtype=dtype)
        w_o_diag = tf.get_variable(
            "W_O_diag", shape=[self._num_units], dtype=dtype)

      if self._use_peepholes:
        c = (tf.sigmoid(f + 1 + w_f_diag * c_prev) * c_prev +
             tf.sigmoid(i + w_i_diag * c_prev) * tf.tanh(j))
      else:
        c = (tf.sigmoid(f + 1) * c_prev + tf.sigmoid(i) * tf.tanh(j))

      if self._cell_clip is not None:
        c = tf.clip_by_value(c, -self._cell_clip, self._cell_clip)

      if self._use_peepholes:
        m = tf.sigmoid(o + w_o_diag * c) * tf.tanh(c)
      else:
        m = tf.sigmoid(o) * tf.tanh(c)

      if self._num_proj is not None:
        proj_shard_size = self._num_proj / self._num_proj_shards
        w_proj = tf.concat(
            1, [tf.get_variable("W_P_%d" % i,
                                shape=[self._num_units, proj_shard_size],
                                initializer=self._initializer, dtype=dtype)
                for i in range(self._num_proj_shards)])
        # TODO(ebrevdo), use matmulsum
        m = tf.matmul(m, w_proj)

    return m, tf.concat(1, [c, m])


class OutputProjectionWrapper(RNNCell):
  """Operator adding an output projection to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your outputs in time,
  do the projection on this batch-concated sequence, then split it
  if needed or directly feed into a softmax.
  """

  def __init__(self, cell, output_size):
    """Create a cell with output projection.

    Args:
      cell: an RNNCell, a projection to output_size is added to it.
      output_size: integer, the size of the output after projection.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if output_size is not positive.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not RNNCell.")
    if output_size < 1:
      raise ValueError("Parameter output_size must be > 0: %d." % output_size)
    self._cell = cell
    self._output_size = output_size

  @property
  def input_size(self):
    return self._cell.input_size

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._cell.state_size

  def __call__(self, inputs, state, scope=None):
    """Run the cell and output projection on inputs, starting from state."""
    output, res_state = self._cell(inputs, state)
    # Default scope: "OutputProjectionWrapper"
    with tf.variable_scope(scope or type(self).__name__):
      projected = linear.linear(output, self._output_size, True)
    return projected, res_state


class InputProjectionWrapper(RNNCell):
  """Operator adding an input projection to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your inputs in time,
  do the projection on this batch-concated sequence, then split it.
  """

  def __init__(self, cell, input_size):
    """Create a cell with input projection.

    Args:
      cell: an RNNCell, a projection of inputs is added before it.
      input_size: integer, the size of the inputs before projection.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if input_size is not positive.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not RNNCell.")
    if input_size < 1:
      raise ValueError("Parameter input_size must be > 0: %d." % input_size)
    self._cell = cell
    self._input_size = input_size

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._cell.output_size

  @property
  def state_size(self):
    return self._cell.state_size

  def __call__(self, inputs, state, scope=None):
    """Run the input projection and then the cell."""
    # Default scope: "InputProjectionWrapper"
    with tf.variable_scope(scope or type(self).__name__):
      projected = linear.linear(inputs, self._cell.input_size, True)
    return self._cell(projected, state)


class DropoutWrapper(RNNCell):
  """Operator adding dropout to inputs and outputs of the given cell."""

  def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
               seed=None):
    """Create a cell with added input and/or output dropout.

    Dropout is never used on the state.

    Args:
      cell: an RNNCell, a projection to output_size is added to it.
      input_keep_prob: unit Tensor or float between 0 and 1, input keep
        probability; if it is float and 1, no input dropout will be added.
      output_keep_prob: unit Tensor or float between 0 and 1, output keep
        probability; if it is float and 1, no output dropout will be added.
      seed: (optional) integer, the randomness seed.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if keep_prob is not between 0 and 1.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")
    if (isinstance(input_keep_prob, float) and
        not (input_keep_prob >= 0.0 and input_keep_prob <= 1.0)):
      raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d"
                       % input_keep_prob)
    if (isinstance(output_keep_prob, float) and
        not (output_keep_prob >= 0.0 and output_keep_prob <= 1.0)):
      raise ValueError("Parameter input_keep_prob must be between 0 and 1: %d"
                       % output_keep_prob)
    self._cell = cell
    self._input_keep_prob = input_keep_prob
    self._output_keep_prob = output_keep_prob
    self._seed = seed

  @property
  def input_size(self):
    return self._cell.input_size

  @property
  def output_size(self):
    return self._cell.output_size

  @property
  def state_size(self):
    return self._cell.state_size

  def __call__(self, inputs, state):
    """Run the cell with the declared dropouts."""
    if (not isinstance(self._input_keep_prob, float) or
        self._input_keep_prob < 1):
      inputs = tf.nn.dropout(inputs, self._input_keep_prob, seed=self._seed)
    output, new_state = self._cell(inputs, state)
    if (not isinstance(self._output_keep_prob, float) or
        self._output_keep_prob < 1):
      output = tf.nn.dropout(output, self._output_keep_prob, seed=self._seed)
    return output, new_state


class EmbeddingWrapper(RNNCell):
  """Operator adding input embedding to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your inputs in time,
  do the embedding on this batch-concated sequence, then split it and
  feed into your RNN.
  """

  def __init__(self, cell, embedding_classes=0, embedding=None,
               initializer=None):
    """Create a cell with an added input embedding.

    Args:
      cell: an RNNCell, an embedding will be put before its inputs.
      embedding_classes: integer, how many symbols will be embedded.
      embedding: Variable, the embedding to use; if None, a new embedding
        will be created; if set, then embedding_classes is not required.
      initializer: an initializer to use when creating the embedding;
        if None, the initializer from variable scope or a default one is used.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if embedding_classes is not positive.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not RNNCell.")
    if embedding_classes < 1 and embedding is None:
      raise ValueError("Pass embedding or embedding_classes must be > 0: %d."
                       % embedding_classes)
    if embedding_classes > 0 and embedding is not None:
      if embedding.size[0] != embedding_classes:
        raise ValueError("You declared embedding_classes=%d but passed an "
                         "embedding for %d classes." % (embedding.size[0],
                                                        embedding_classes))
      if embedding.size[1] != cell.input_size:
        raise ValueError("You passed embedding with output size %d and a cell"
                         " that accepts size %d." % (embedding.size[1],
                                                     cell.input_size))
    self._cell = cell
    self._embedding_classes = embedding_classes
    self._embedding = embedding
    self._initializer = initializer

  @property
  def input_size(self):
    return 1

  @property
  def output_size(self):
    return self._cell.output_size

  @property
  def state_size(self):
    return self._cell.state_size

  def __call__(self, inputs, state, scope=None):
    """Run the cell on embedded inputs."""
    with tf.variable_scope(scope or type(self).__name__):  # "EmbeddingWrapper"
      with tf.device("/cpu:0"):
        if self._embedding:
          embedding = self._embedding
        else:
          if self._initializer:
            initializer = self._initializer
          elif tf.get_variable_scope().initializer:
            initializer = tf.get_variable_scope().initializer
          else:
            # Default initializer for embeddings should have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)
          embedding = tf.get_variable("embedding", [self._embedding_classes,
                                                    self._cell.input_size],
                                      initializer=initializer)
        embedded = tf.nn.embedding_lookup(embedding, tf.reshape(inputs, [-1]))
    return self._cell(embedded, state)


class MultiRNNCell(RNNCell):
  """RNN cell composed sequentially of multiple simple cells."""

  def __init__(self, cells):
    """Create a RNN cell composed sequentially of a number of RNNCells.

    Args:
      cells: list of RNNCells that will be composed in this order.

    Raises:
      ValueError: if cells is empty (not allowed) or if their sizes don't match.
    """
    if not cells:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")
    for i in xrange(len(cells) - 1):
      if cells[i + 1].input_size != cells[i].output_size:
        raise ValueError("In MultiRNNCell, the input size of each next"
                         " cell must match the output size of the previous one."
                         " Mismatched output size in cell %d." % i)
    self._cells = cells

  @property
  def input_size(self):
    return self._cells[0].input_size

  @property
  def output_size(self):
    return self._cells[-1].output_size

  @property
  def state_size(self):
    return sum([cell.state_size for cell in self._cells])

  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    with tf.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
      cur_state_pos = 0
      cur_inp = inputs
      new_states = []
      for i, cell in enumerate(self._cells):
        with tf.variable_scope("Cell%d" % i):
          cur_state = tf.slice(state, [0, cur_state_pos], [-1, cell.state_size])
          cur_state_pos += cell.state_size
          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)
    return cur_inp, tf.concat(1, new_states)
