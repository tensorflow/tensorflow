# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh


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
    zeros = array_ops.zeros(
        array_ops.pack([batch_size, self.state_size]), dtype=dtype)
    zeros.set_shape([None, self.state_size])
    return zeros


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
    with vs.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
      output = tanh(linear([inputs, state], self._num_units, True))
    return output, output


class GRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None):
    self._num_units = num_units
    self._input_size = num_units if input_size is None else input_size

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with vs.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        r, u = array_ops.split(1, 2, linear([inputs, state],
                                            2 * self._num_units, True, 1.0))
        r, u = sigmoid(r), sigmoid(u)
      with vs.variable_scope("Candidate"):
        c = tanh(linear([inputs, r * state], self._num_units, True))
      new_h = u * state + (1 - u) * c
    return new_h, new_h


class BasicLSTMCell(RNNCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full LSTMCell that follows.
  """

  def __init__(self, num_units, forget_bias=1.0, input_size=None):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: int, The dimensionality of the inputs into the LSTM cell,
        by default equal to num_units.
    """
    self._num_units = num_units
    self._input_size = num_units if input_size is None else input_size
    self._forget_bias = forget_bias

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return 2 * self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      c, h = array_ops.split(1, 2, state)
      concat = linear([inputs, h], 4 * self._num_units, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = array_ops.split(1, 4, concat)

      new_c = c * sigmoid(f + self._forget_bias) + sigmoid(i) * tanh(j)
      new_h = tanh(new_c) * sigmoid(o)

    return new_h, array_ops.concat(1, [new_c, new_h])


def _get_concat_variable(name, shape, dtype, num_shards):
  """Get a sharded variable concatenated into one tensor."""
  sharded_variable = _get_sharded_variable(name, shape, dtype, num_shards)
  if len(sharded_variable) == 1:
    return sharded_variable[0]

  concat_name = name + "/concat"
  concat_full_name = vs.get_variable_scope().name + "/" + concat_name + ":0"
  for value in ops.get_collection(ops.GraphKeys.CONCATENATED_VARIABLES):
    if value.name == concat_full_name:
      return value

  concat_variable = array_ops.concat(0, sharded_variable, name=concat_name)
  ops.add_to_collection(ops.GraphKeys.CONCATENATED_VARIABLES,
                        concat_variable)
  return concat_variable


def _get_sharded_variable(name, shape, dtype, num_shards):
  """Get a list of sharded variables with the given dtype."""
  if num_shards > shape[0]:
    raise ValueError("Too many shards: shape=%s, num_shards=%d" %
                     (shape, num_shards))
  unit_shard_size = int(math.floor(shape[0] / num_shards))
  remaining_rows = shape[0] - unit_shard_size * num_shards

  shards = []
  for i in range(num_shards):
    current_size = unit_shard_size
    if i < remaining_rows:
      current_size += 1
    shards.append(vs.get_variable(name + "_%d" % i, [current_size, shape[1]],
                                  dtype=dtype))
  return shards


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
      num_proj_shards: How to split the projection matrix.  If >1, the
        projection matrix is stored across num_proj_shards.
    """
    self._num_units = num_units
    self._input_size = input_size
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards

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

    c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
    m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

    dtype = input_.dtype

    with vs.variable_scope(scope or type(self).__name__,
                           initializer=self._initializer):  # "LSTMCell"
      concat_w = _get_concat_variable(
          "W", [self.input_size + num_proj, 4 * self._num_units],
          dtype, self._num_unit_shards)

      b = vs.get_variable(
          "B", shape=[4 * self._num_units],
          initializer=array_ops.zeros_initializer, dtype=dtype)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      cell_inputs = array_ops.concat(1, [input_, m_prev])
      lstm_matrix = nn_ops.bias_add(math_ops.matmul(cell_inputs, concat_w), b)
      i, j, f, o = array_ops.split(1, 4, lstm_matrix)

      # Diagonal connections
      if self._use_peepholes:
        w_f_diag = vs.get_variable(
            "W_F_diag", shape=[self._num_units], dtype=dtype)
        w_i_diag = vs.get_variable(
            "W_I_diag", shape=[self._num_units], dtype=dtype)
        w_o_diag = vs.get_variable(
            "W_O_diag", shape=[self._num_units], dtype=dtype)

      if self._use_peepholes:
        c = (sigmoid(f + 1 + w_f_diag * c_prev) * c_prev +
             sigmoid(i + w_i_diag * c_prev) * tanh(j))
      else:
        c = (sigmoid(f + 1) * c_prev + sigmoid(i) * tanh(j))

      if self._cell_clip is not None:
        c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)

      if self._use_peepholes:
        m = sigmoid(o + w_o_diag * c) * tanh(c)
      else:
        m = sigmoid(o) * tanh(c)

      if self._num_proj is not None:
        concat_w_proj = _get_concat_variable(
            "W_P", [self._num_units, self._num_proj],
            dtype, self._num_proj_shards)

        m = math_ops.matmul(m, concat_w_proj)

    return m, array_ops.concat(1, [c, m])


class OutputProjectionWrapper(RNNCell):
  """Operator adding an output projection to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your outputs in time,
  do the projection on this batch-concatenated sequence, then split it
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
    with vs.variable_scope(scope or type(self).__name__):
      projected = linear(output, self._output_size, True)
    return projected, res_state


class InputProjectionWrapper(RNNCell):
  """Operator adding an input projection to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your inputs in time,
  do the projection on this batch-concatenated sequence, then split it.
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
    with vs.variable_scope(scope or type(self).__name__):
      projected = linear(inputs, self._cell.input_size, True)
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

  def __call__(self, inputs, state, scope=None):
    """Run the cell with the declared dropouts."""
    if (not isinstance(self._input_keep_prob, float) or
        self._input_keep_prob < 1):
      inputs = nn_ops.dropout(inputs, self._input_keep_prob, seed=self._seed)
    output, new_state = self._cell(inputs, state)
    if (not isinstance(self._output_keep_prob, float) or
        self._output_keep_prob < 1):
      output = nn_ops.dropout(output, self._output_keep_prob, seed=self._seed)
    return output, new_state


class EmbeddingWrapper(RNNCell):
  """Operator adding input embedding to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your inputs in time,
  do the embedding on this batch-concatenated sequence, then split it and
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
    with vs.variable_scope(scope or type(self).__name__):  # "EmbeddingWrapper"
      with ops.device("/cpu:0"):
        if self._embedding:
          embedding = self._embedding
        else:
          if self._initializer:
            initializer = self._initializer
          elif vs.get_variable_scope().initializer:
            initializer = vs.get_variable_scope().initializer
          else:
            # Default initializer for embeddings should have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)
          embedding = vs.get_variable("embedding", [self._embedding_classes,
                                                    self._cell.input_size],
                                      initializer=initializer)
        embedded = embedding_ops.embedding_lookup(
            embedding, array_ops.reshape(inputs, [-1]))
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
    with vs.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
      cur_state_pos = 0
      cur_inp = inputs
      new_states = []
      for i, cell in enumerate(self._cells):
        with vs.variable_scope("Cell%d" % i):
          cur_state = array_ops.slice(
              state, [0, cur_state_pos], [-1, cell.state_size])
          cur_state_pos += cell.state_size
          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)
    return cur_inp, array_ops.concat(1, new_states)


def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  assert args
  if not isinstance(args, (list, tuple)):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with vs.variable_scope(scope or "Linear"):
    matrix = vs.get_variable("Matrix", [total_arg_size, output_size])
    if len(args) == 1:
      res = math_ops.matmul(args[0], matrix)
    else:
      res = math_ops.matmul(array_ops.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = vs.get_variable(
        "Bias", [output_size],
        initializer=init_ops.constant_initializer(bias_start))
  return res + bias_term
