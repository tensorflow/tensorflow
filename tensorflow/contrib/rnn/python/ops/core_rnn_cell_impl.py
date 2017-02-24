# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Module implementing RNN Cells."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


class BasicRNNCell(RNNCell):
  """The most basic RNN cell."""

  def __init__(self, num_units, input_size=None, activation=tanh):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
    with vs.variable_scope(scope or "basic_rnn_cell"):
      output = self._activation(
          _linear([inputs, state], self._num_units, True))
    return output, output


class GRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None, activation=tanh):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope(scope or "gru_cell"):
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        r, u = array_ops.split(
            value=_linear(
                [inputs, state], 2 * self._num_units, True, 1.0),
            num_or_size_splits=2,
            axis=1)
        r, u = sigmoid(r), sigmoid(u)
      with vs.variable_scope("candidate"):
        c = self._activation(_linear([inputs, r * state],
                                     self._num_units, True))
      new_h = u * state + (1 - u) * c
    return new_h, new_h


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(c, h)`, in that order.

  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c, h) = self
    if not c.dtype == h.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c.dtype), str(h.dtype)))
    return c.dtype


class BasicLSTMCell(RNNCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full LSTMCell that follows.
  """

  def __init__(self, num_units, forget_bias=1.0, input_size=None,
               state_is_tuple=True, activation=tanh):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.
    """
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with vs.variable_scope(scope or "basic_lstm_cell"):
      # Parameters of gates are concatenated into one multiply for efficiency.
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)
      concat = _linear([inputs, h], 4 * self._num_units, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

      new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * sigmoid(o)

      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c, new_h)
      else:
        new_state = array_ops.concat([new_c, new_h], 1)
      return new_h, new_state


class LSTMCell(RNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.

  The default non-peephole implementation is based on:

    http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

  S. Hochreiter and J. Schmidhuber.
  "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

  The peephole implementation is based on:

    https://research.google.com/pubs/archive/43905.pdf

  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.

  The class uses optional peep-hole connections, optional cell clipping, and
  an optional projection layer.
  """

  def __init__(self, num_units, input_size=None,
               use_peepholes=False, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=None, num_proj_shards=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=tanh):
    """Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell
      input_size: Deprecated and unused.
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within
        `[-proj_clip, proj_clip]`.
      num_unit_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      num_proj_shards: Deprecated, will be removed by Jan. 2017.
        Use a variable_scope partitioner instead.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  This latter behavior will soon be deprecated.
      activation: Activation function of the inner states.
    """
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      logging.warn(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)

    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._num_unit_shards = num_unit_shards
    self._num_proj_shards = num_proj_shards
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation

    if num_proj:
      self._state_size = (
          LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def __call__(self, inputs, state, scope=None):
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.
      scope: VariableScope for the created subgraph; defaults to "lstm_cell".

    Returns:
      A tuple containing:

      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    num_proj = self._num_units if self._num_proj is None else self._num_proj

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
    with vs.variable_scope(scope or "lstm_cell",
                           initializer=self._initializer) as unit_scope:
      if self._num_unit_shards is not None:
        unit_scope.set_partitioner(
            partitioned_variables.fixed_size_partitioner(
                self._num_unit_shards))
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      lstm_matrix = _linear([inputs, m_prev], 4 * self._num_units, bias=True)
      i, j, f, o = array_ops.split(
          value=lstm_matrix, num_or_size_splits=4, axis=1)
      # Diagonal connections
      if self._use_peepholes:
        with vs.variable_scope(unit_scope) as projection_scope:
          if self._num_unit_shards is not None:
            projection_scope.set_partitioner(None)
          w_f_diag = vs.get_variable(
              "w_f_diag", shape=[self._num_units], dtype=dtype)
          w_i_diag = vs.get_variable(
              "w_i_diag", shape=[self._num_units], dtype=dtype)
          w_o_diag = vs.get_variable(
              "w_o_diag", shape=[self._num_units], dtype=dtype)

      if self._use_peepholes:
        c = (sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev +
             sigmoid(i + w_i_diag * c_prev) * self._activation(j))
      else:
        c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
             self._activation(j))

      if self._cell_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
        # pylint: enable=invalid-unary-operand-type
      if self._use_peepholes:
        m = sigmoid(o + w_o_diag * c) * self._activation(c)
      else:
        m = sigmoid(o) * self._activation(c)

      if self._num_proj is not None:
        with vs.variable_scope("projection") as proj_scope:
          if self._num_proj_shards is not None:
            proj_scope.set_partitioner(
                partitioned_variables.fixed_size_partitioner(
                    self._num_proj_shards))
          m = _linear(m, self._num_proj, bias=False)

        if self._proj_clip is not None:
          # pylint: disable=invalid-unary-operand-type
          m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
          # pylint: enable=invalid-unary-operand-type

    new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                 array_ops.concat([c, m], 1))
    return m, new_state


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
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._output_size

  def __call__(self, inputs, state, scope=None):
    """Run the cell and output projection on inputs, starting from state."""
    output, res_state = self._cell(inputs, state)
    # Default scope: "OutputProjectionWrapper"
    with vs.variable_scope(scope or "output_projection_wrapper"):
      projected = _linear(output, self._output_size, True)
    return projected, res_state


class InputProjectionWrapper(RNNCell):
  """Operator adding an input projection to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your inputs in time,
  do the projection on this batch-concatenated sequence, then split it.
  """

  def __init__(self, cell, num_proj, input_size=None):
    """Create a cell with input projection.

    Args:
      cell: an RNNCell, a projection of inputs is added before it.
      num_proj: Python integer.  The dimension to project to.
      input_size: Deprecated and unused.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not RNNCell.")
    self._cell = cell
    self._num_proj = num_proj

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    """Run the input projection and then the cell."""
    # Default scope: "InputProjectionWrapper"
    with vs.variable_scope(scope or "input_projection_wrapper"):
      projected = _linear(inputs, self._num_proj, True)
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
      raise ValueError("Parameter output_keep_prob must be between 0 and 1: %d"
                       % output_keep_prob)
    self._cell = cell
    self._input_keep_prob = input_keep_prob
    self._output_keep_prob = output_keep_prob
    self._seed = seed

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    """Run the cell with the declared dropouts."""
    if (not isinstance(self._input_keep_prob, float) or
        self._input_keep_prob < 1):
      inputs = nn_ops.dropout(inputs, self._input_keep_prob, seed=self._seed)
    output, new_state = self._cell(inputs, state, scope)
    if (not isinstance(self._output_keep_prob, float) or
        self._output_keep_prob < 1):
      output = nn_ops.dropout(output, self._output_keep_prob, seed=self._seed)
    return output, new_state


class ResidualWrapper(RNNCell):
  """RNNCell wrapper that ensures cell inputs are added to the outputs."""

  def __init__(self, cell):
    """Constructs a `ResidualWrapper` for `cell`.

    Args:
      cell: An instance of `RNNCell`.
    """
    self._cell = cell

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    """Run the cell and add its inputs to its outputs.

    Args:
      inputs: cell inputs.
      state: cell state.
      scope: optional cell scope.

    Returns:
      Tuple of cell outputs and new state.

    Raises:
      TypeError: If cell inputs and outputs have different structure (type).
      ValueError: If cell inputs and outputs have different structure (value).
    """
    outputs, new_state = self._cell(inputs, state, scope=scope)
    nest.assert_same_structure(inputs, outputs)
    # Ensure shapes match
    def assert_shape_match(inp, out):
      inp.get_shape().assert_is_compatible_with(out.get_shape())
    nest.map_structure(assert_shape_match, inputs, outputs)
    res_outputs = nest.map_structure(
        lambda inp, out: inp + out, inputs, outputs)
    return (res_outputs, new_state)


class DeviceWrapper(RNNCell):
  """Operator that ensures an RNNCell runs on a particular device."""

  def __init__(self, cell, device):
    """Construct a `DeviceWrapper` for `cell` with device `device`.

    Ensures the wrapped `cell` is called with `tf.device(device)`.

    Args:
      cell: An instance of `RNNCell`.
      device: A device string or function, for passing to `tf.device`.
    """
    self._cell = cell
    self._device = device

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    """Run the cell on specified device."""
    with ops.device(self._device):
      return self._cell(inputs, state, scope=scope)


class EmbeddingWrapper(RNNCell):
  """Operator adding input embedding to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your inputs in time,
  do the embedding on this batch-concatenated sequence, then split it and
  feed into your RNN.
  """

  def __init__(self, cell, embedding_classes, embedding_size, initializer=None):
    """Create a cell with an added input embedding.

    Args:
      cell: an RNNCell, an embedding will be put before its inputs.
      embedding_classes: integer, how many symbols will be embedded.
      embedding_size: integer, the size of the vectors we embed into.
      initializer: an initializer to use when creating the embedding;
        if None, the initializer from variable scope or a default one is used.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if embedding_classes is not positive.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not RNNCell.")
    if embedding_classes <= 0 or embedding_size <= 0:
      raise ValueError("Both embedding_classes and embedding_size must be > 0: "
                       "%d, %d." % (embedding_classes, embedding_size))
    self._cell = cell
    self._embedding_classes = embedding_classes
    self._embedding_size = embedding_size
    self._initializer = initializer

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    """Run the cell on embedded inputs."""
    with vs.variable_scope(scope or "embedding_wrapper"):  # "EmbeddingWrapper"
      with ops.device("/cpu:0"):
        if self._initializer:
          initializer = self._initializer
        elif vs.get_variable_scope().initializer:
          initializer = vs.get_variable_scope().initializer
        else:
          # Default initializer for embeddings should have variance=1.
          sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
          initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)

        if type(state) is tuple:
          data_type = state[0].dtype
        else:
          data_type = state.dtype

        embedding = vs.get_variable(
            "embedding", [self._embedding_classes, self._embedding_size],
            initializer=initializer,
            dtype=data_type)
        embedded = embedding_ops.embedding_lookup(
            embedding, array_ops.reshape(inputs, [-1]))
    return self._cell(embedded, state)


class MultiRNNCell(RNNCell):
  """RNN cell composed sequentially of multiple simple cells."""

  def __init__(self, cells, state_is_tuple=True):
    """Create a RNN cell composed sequentially of a number of RNNCells.

    Args:
      cells: list of RNNCells that will be composed in this order.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  If False, the states are all
        concatenated along the column axis.  This latter behavior will soon be
        deprecated.

    Raises:
      ValueError: if cells is empty (not allowed), or at least one of the cells
        returns a state tuple but the flag `state_is_tuple` is `False`.
    """
    if not cells:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")
    if not nest.is_sequence(cells):
      raise TypeError(
          "cells must be a list or tuple, but saw: %s." % cells)

    self._cells = cells
    self._state_is_tuple = state_is_tuple
    if not state_is_tuple:
      if any(nest.is_sequence(c.state_size) for c in self._cells):
        raise ValueError("Some cells return tuples of states, but the flag "
                         "state_is_tuple is not set.  State sizes are: %s"
                         % str([c.state_size for c in self._cells]))

  @property
  def state_size(self):
    if self._state_is_tuple:
      return tuple(cell.state_size for cell in self._cells)
    else:
      return sum([cell.state_size for cell in self._cells])

  @property
  def output_size(self):
    return self._cells[-1].output_size

  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    with vs.variable_scope(scope or "multi_rnn_cell"):
      cur_state_pos = 0
      cur_inp = inputs
      new_states = []
      for i, cell in enumerate(self._cells):
        with vs.variable_scope("cell_%d" % i):
          if self._state_is_tuple:
            if not nest.is_sequence(state):
              raise ValueError(
                  "Expected state to be a tuple of length %d, but received: %s"
                  % (len(self.state_size), state))
            cur_state = state[i]
          else:
            cur_state = array_ops.slice(
                state, [0, cur_state_pos], [-1, cell.state_size])
            cur_state_pos += cell.state_size
          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)
    new_states = (tuple(new_states) if self._state_is_tuple else
                  array_ops.concat(new_states, 1))
    return cur_inp, new_states


class _SlimRNNCell(RNNCell):
  """A simple wrapper for slim.rnn_cells."""

  def __init__(self, cell_fn):
    """Create a SlimRNNCell from a cell_fn.

    Args:
      cell_fn: a function which takes (inputs, state, scope) and produces the
        outputs and the new_state. Additionally when called with inputs=None and
        state=None it should return (initial_outputs, initial_state).

    Raises:
      TypeError: if cell_fn is not callable
      ValueError: if cell_fn cannot produce a valid initial state.
    """
    if not callable(cell_fn):
      raise TypeError("cell_fn %s needs to be callable", cell_fn)
    self._cell_fn = cell_fn
    self._cell_name = cell_fn.func.__name__
    init_output, init_state = self._cell_fn(None, None)
    output_shape = init_output.get_shape()
    state_shape = init_state.get_shape()
    self._output_size = output_shape.with_rank(2)[1].value
    self._state_size = state_shape.with_rank(2)[1].value
    if self._output_size is None:
      raise ValueError("Initial output created by %s has invalid shape %s" %
                       (self._cell_name, output_shape))
    if self._state_size is None:
      raise ValueError("Initial state created by %s has invalid shape %s" %
                       (self._cell_name, state_shape))

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def __call__(self, inputs, state, scope=None):
    scope = scope or self._cell_name
    output, state = self._cell_fn(inputs, state, scope=scope)
    return output, state


def _linear(args, output_size, bias, bias_start=0.0):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        "weights", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = vs.get_variable(
          "biases", [output_size],
          dtype=dtype,
          initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    return nn_ops.bias_add(res, biases)
