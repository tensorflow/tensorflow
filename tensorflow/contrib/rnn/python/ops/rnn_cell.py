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

"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

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
    shards.append(vs.get_variable(name + "_%d" % i, [current_size] + shape[1:],
                                  dtype=dtype))
  return shards


class CoupledInputForgetGateLSTMCell(rnn_cell.RNNCell):
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

  The coupling of input and forget gate is based on:

    http://arxiv.org/pdf/1503.04069.pdf

  Greff et al. "LSTM: A Search Space Odyssey"

  The class uses optional peep-hole connections, and an optional projection
  layer.
  """

  def __init__(self, num_units, use_peepholes=False,
               initializer=None, num_proj=None, proj_clip=None,
               num_unit_shards=1, num_proj_shards=1,
               forget_bias=1.0, state_is_tuple=False,
               activation=math_ops.tanh):
    """Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
      provided, then the projected values are clipped elementwise to within
      `[-proj_clip, proj_clip]`.
      num_unit_shards: How to split the weight matrix.  If >1, the weight
        matrix is stored across num_unit_shards.
      num_proj_shards: How to split the projection matrix.  If >1, the
        projection matrix is stored across num_proj_shards.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  By default (False), they are concatenated
        along the column axis.  This default behavior will soon be deprecated.
      activation: Activation function of the inner states.
    """
    if not state_is_tuple:
      logging.warn(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use state_is_tuple=True." % self)
    self._num_units = num_units
    self._use_peepholes = use_peepholes
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
          rnn_cell.LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (
          rnn_cell.LSTMStateTuple(num_units, num_units)
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
      scope: VariableScope for the created subgraph; defaults to "LSTMCell".

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
    sigmoid = math_ops.sigmoid

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
    with vs.variable_scope(scope or type(self).__name__,
                           initializer=self._initializer):  # "LSTMCell"
      concat_w = _get_concat_variable(
          "W", [input_size.value + num_proj, 3 * self._num_units],
          dtype, self._num_unit_shards)

      b = vs.get_variable(
          "B", shape=[3 * self._num_units],
          initializer=init_ops.zeros_initializer, dtype=dtype)

      # j = new_input, f = forget_gate, o = output_gate
      cell_inputs = array_ops.concat(1, [inputs, m_prev])
      lstm_matrix = nn_ops.bias_add(math_ops.matmul(cell_inputs, concat_w), b)
      j, f, o = array_ops.split(1, 3, lstm_matrix)

      # Diagonal connections
      if self._use_peepholes:
        w_f_diag = vs.get_variable(
            "W_F_diag", shape=[self._num_units], dtype=dtype)
        w_o_diag = vs.get_variable(
            "W_O_diag", shape=[self._num_units], dtype=dtype)

      if self._use_peepholes:
        f_act = sigmoid(f + self._forget_bias + w_f_diag * c_prev)
      else:
        f_act = sigmoid(f + self._forget_bias)
      c = (f_act * c_prev + (1 - f_act) * self._activation(j))

      if self._use_peepholes:
        m = sigmoid(o + w_o_diag * c) * self._activation(c)
      else:
        m = sigmoid(o) * self._activation(c)

      if self._num_proj is not None:
        concat_w_proj = _get_concat_variable(
            "W_P", [self._num_units, self._num_proj],
            dtype, self._num_proj_shards)

        m = math_ops.matmul(m, concat_w_proj)
        if self._proj_clip is not None:
          # pylint: disable=invalid-unary-operand-type
          m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
          # pylint: enable=invalid-unary-operand-type

    new_state = (rnn_cell.LSTMStateTuple(c, m) if self._state_is_tuple
                 else array_ops.concat(1, [c, m]))
    return m, new_state


class TimeFreqLSTMCell(rnn_cell.RNNCell):
  """Time-Frequency Long short-term memory unit (LSTM) recurrent network cell.

  This implementation is based on:

    Tara N. Sainath and Bo Li
    "Modeling Time-Frequency Patterns with LSTM vs. Convolutional Architectures
    for LVCSR Tasks." submitted to INTERSPEECH, 2016.

  It uses peep-hole connections and optional cell clipping.
  """

  def __init__(self, num_units, use_peepholes=False,
               cell_clip=None, initializer=None,
               num_unit_shards=1, forget_bias=1.0,
               feature_size=None, frequency_skip=None):
    """Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell
      use_peepholes: bool, set True to enable diagonal/peephole connections.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_unit_shards: int, How to split the weight matrix.  If >1, the weight
        matrix is stored across num_unit_shards.
      forget_bias: float, Biases of the forget gate are initialized by default
        to 1 in order to reduce the scale of forgetting at the beginning
        of the training.
      feature_size: int, The size of the input feature the LSTM spans over.
      frequency_skip: int, The amount the LSTM filter is shifted by in
        frequency.
    """
    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_unit_shards = num_unit_shards
    self._forget_bias = forget_bias
    self._feature_size = feature_size
    self._frequency_skip = frequency_skip
    self._state_size = 2 * num_units
    self._output_size = num_units

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def __call__(self, inputs, state, scope=None):
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: state Tensor, 2D, batch x state_size.
      scope: VariableScope for the created subgraph; defaults to
        "TimeFreqLSTMCell".

    Returns:
      A tuple containing:
      - A 2D, batch x output_dim, Tensor representing the output of the LSTM
        after reading "inputs" when previous state was "state".
        Here output_dim is num_units.
      - A 2D, batch x state_size, Tensor representing the new state of LSTM
        after reading "inputs" when previous state was "state".
    Raises:
      ValueError: if an input_size was specified and the provided inputs have
        a different dimension.
    """
    sigmoid = math_ops.sigmoid
    tanh = math_ops.tanh

    freq_inputs = self._make_tf_features(inputs)
    dtype = inputs.dtype
    actual_input_size = freq_inputs[0].get_shape().as_list()[1]
    with vs.variable_scope(scope or type(self).__name__,
                           initializer=self._initializer):  # "TimeFreqLSTMCell"
      concat_w = _get_concat_variable(
          "W", [actual_input_size + 2*self._num_units, 4 * self._num_units],
          dtype, self._num_unit_shards)
      b = vs.get_variable(
          "B", shape=[4 * self._num_units],
          initializer=init_ops.zeros_initializer, dtype=dtype)

      # Diagonal connections
      if self._use_peepholes:
        w_f_diag = vs.get_variable(
            "W_F_diag", shape=[self._num_units], dtype=dtype)
        w_i_diag = vs.get_variable(
            "W_I_diag", shape=[self._num_units], dtype=dtype)
        w_o_diag = vs.get_variable(
            "W_O_diag", shape=[self._num_units], dtype=dtype)

      # initialize the first freq state to be zero
      m_prev_freq = array_ops.zeros([int(inputs.get_shape()[0]),
                                     self._num_units], dtype)
      for fq in range(len(freq_inputs)):
        c_prev = array_ops.slice(state, [0, 2*fq*self._num_units],
                                 [-1, self._num_units])
        m_prev = array_ops.slice(state, [0, (2*fq+1)*self._num_units],
                                 [-1, self._num_units])
        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        cell_inputs = array_ops.concat(1, [freq_inputs[fq], m_prev,
                                           m_prev_freq])
        lstm_matrix = nn_ops.bias_add(math_ops.matmul(cell_inputs, concat_w), b)
        i, j, f, o = array_ops.split(1, 4, lstm_matrix)

        if self._use_peepholes:
          c = (sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev +
               sigmoid(i + w_i_diag * c_prev) * tanh(j))
        else:
          c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) * tanh(j))

        if self._cell_clip is not None:
          # pylint: disable=invalid-unary-operand-type
          c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
          # pylint: enable=invalid-unary-operand-type

        if self._use_peepholes:
          m = sigmoid(o + w_o_diag * c) * tanh(c)
        else:
          m = sigmoid(o) * tanh(c)
        m_prev_freq = m
        if fq == 0:
          state_out = array_ops.concat(1, [c, m])
          m_out = m
        else:
          state_out = array_ops.concat(1, [state_out, c, m])
          m_out = array_ops.concat(1, [m_out, m])
    return m_out, state_out

  def _make_tf_features(self, input_feat):
    """Make the frequency features.

    Args:
      input_feat: input Tensor, 2D, batch x num_units.

    Returns:
      A list of frequency features, with each element containing:
      - A 2D, batch x output_dim, Tensor representing the time-frequency feature
        for that frequency index. Here output_dim is feature_size.
    Raises:
      ValueError: if input_size cannot be inferred from static shape inference.
    """
    input_size = input_feat.get_shape().with_rank(2)[-1].value
    if input_size is None:
      raise ValueError("Cannot infer input_size from static shape inference.")
    num_feats = int((input_size - self._feature_size) / (
        self._frequency_skip)) + 1
    freq_inputs = []
    for f in range(num_feats):
      cur_input = array_ops.slice(input_feat, [0, f*self._frequency_skip],
                                  [-1, self._feature_size])
      freq_inputs.append(cur_input)
    return freq_inputs


class GridLSTMCell(rnn_cell.RNNCell):
  """Grid Long short-term memory unit (LSTM) recurrent network cell.

  The default is based on:
    Nal Kalchbrenner, Ivo Danihelka and Alex Graves
    "Grid Long Short-Term Memory," Proc. ICLR 2016.
    http://arxiv.org/abs/1507.01526

  When peephole connections are used, the implementation is based on:
    Tara N. Sainath and Bo Li
    "Modeling Time-Frequency Patterns with LSTM vs. Convolutional Architectures
    for LVCSR Tasks." submitted to INTERSPEECH, 2016.

  The code uses optional peephole connections, shared_weights and cell clipping.
  """

  def __init__(self, num_units, use_peepholes=False,
               share_time_frequency_weights=False,
               cell_clip=None, initializer=None,
               num_unit_shards=1, forget_bias=1.0,
               feature_size=None, frequency_skip=None,
               num_frequency_blocks=1,
               couple_input_forget_gates=False,
               state_is_tuple=False):
    """Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell
      use_peepholes: bool, default False. Set True to enable diagonal/peephole
        connections.
      share_time_frequency_weights: bool, default False. Set True to enable
        shared cell weights between time and frequency LSTMs.
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_unit_shards: int, How to split the weight matrix.  If >1, the weight
        matrix is stored across num_unit_shards.
      forget_bias: float, Biases of the forget gate are initialized by default
        to 1 in order to reduce the scale of forgetting at the beginning
        of the training.
      feature_size: int, The size of the input feature the LSTM spans over.
      frequency_skip: int, The amount the LSTM filter is shifted by in
        frequency.
      num_frequency_blocks: int, The total number of frequency blocks needed to
        cover the whole input feature.
      couple_input_forget_gates: bool, Whether to couple the input and forget
        gates, i.e. f_gate = 1.0 - i_gate, to reduce model parameters and
        computation cost.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  By default (False), they are concatenated
        along the column axis.  This default behavior will soon be deprecated.
    """
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._share_time_frequency_weights = share_time_frequency_weights
    self._couple_input_forget_gates = couple_input_forget_gates
    self._state_is_tuple = state_is_tuple
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_unit_shards = num_unit_shards
    self._forget_bias = forget_bias
    self._feature_size = feature_size
    self._frequency_skip = frequency_skip
    self._num_frequency_blocks = int(num_frequency_blocks)
    if state_is_tuple:
      state_names = ""
      for freq_index in range(self._num_frequency_blocks):
        name_prefix = "state_f%02d" % freq_index
        state_names += ("%s_c, %s_m," % (name_prefix, name_prefix))
      self._state_tuple_type = collections.namedtuple(
          "GridLSTMStateTuple", state_names.strip(','))
      self._state_size = self._state_tuple_type(
              *([num_units, num_units] * self._num_frequency_blocks))
    else:
      self._state_tuple_type = None
      self._state_size = num_units * self._num_frequency_blocks * 2
    self._output_size = num_units * self._num_frequency_blocks * 2

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  @property
  def state_tuple_type(self):
    return self._state_tuple_type

  def __call__(self, inputs, state, scope=None):
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: state Tensor, 2D, batch x state_size.
      scope: VariableScope for the created subgraph; defaults to "LSTMCell".

    Returns:
      A tuple containing:
      - A 2D, batch x output_dim, Tensor representing the output of the LSTM
        after reading "inputs" when previous state was "state".
        Here output_dim is num_units.
      - A 2D, batch x state_size, Tensor representing the new state of LSTM
        after reading "inputs" when previous state was "state".
    Raises:
      ValueError: if an input_size was specified and the provided inputs have
        a different dimension.
    """
    sigmoid = math_ops.sigmoid
    tanh = math_ops.tanh
    num_gates = 3 if self._couple_input_forget_gates else 4

    freq_inputs = self._make_tf_features(inputs)
    dtype = inputs.dtype
    actual_input_size = freq_inputs[0].get_shape().as_list()[1]
    with vs.variable_scope(scope or type(self).__name__,
                           initializer=self._initializer):  # "GridLSTMCell"
      concat_w_f = _get_concat_variable(
          "W_f", [actual_input_size + 2 * self._num_units,
                  num_gates * self._num_units],
          dtype, self._num_unit_shards)
      b_f = vs.get_variable(
          "B_f", shape=[num_gates * self._num_units],
          initializer=init_ops.zeros_initializer, dtype=dtype)
      if not self._share_time_frequency_weights:
        concat_w_t = _get_concat_variable(
            "W_t", [actual_input_size + 2 * self._num_units,
                    num_gates * self._num_units],
            dtype, self._num_unit_shards)
        b_t = vs.get_variable(
            "B_t", shape=[num_gates * self._num_units],
            initializer=init_ops.zeros_initializer, dtype=dtype)

      if self._use_peepholes:
        # Diagonal connections
        if not self._couple_input_forget_gates:
          w_f_diag_freqf = vs.get_variable(
              "W_F_diag_freqf", shape=[self._num_units], dtype=dtype)
          w_f_diag_freqt = vs.get_variable(
              "W_F_diag_freqt", shape=[self._num_units], dtype=dtype)
        w_i_diag_freqf = vs.get_variable(
            "W_I_diag_freqf", shape=[self._num_units], dtype=dtype)
        w_i_diag_freqt = vs.get_variable(
            "W_I_diag_freqt", shape=[self._num_units], dtype=dtype)
        w_o_diag_freqf = vs.get_variable(
            "W_O_diag_freqf", shape=[self._num_units], dtype=dtype)
        w_o_diag_freqt = vs.get_variable(
            "W_O_diag_freqt", shape=[self._num_units], dtype=dtype)
        if not self._share_time_frequency_weights:
          if not self._couple_input_forget_gates:
            w_f_diag_timef = vs.get_variable(
                "W_F_diag_timef", shape=[self._num_units], dtype=dtype)
            w_f_diag_timet = vs.get_variable(
                "W_F_diag_timet", shape=[self._num_units], dtype=dtype)
          w_i_diag_timef = vs.get_variable(
              "W_I_diag_timef", shape=[self._num_units], dtype=dtype)
          w_i_diag_timet = vs.get_variable(
              "W_I_diag_timet", shape=[self._num_units], dtype=dtype)
          w_o_diag_timef = vs.get_variable(
              "W_O_diag_timef", shape=[self._num_units], dtype=dtype)
          w_o_diag_timet = vs.get_variable(
              "W_O_diag_timet", shape=[self._num_units], dtype=dtype)

      # initialize the first freq state to be zero
      m_prev_freq = array_ops.zeros(
          [int(inputs.get_shape()[0]), self._num_units], dtype)
      c_prev_freq = array_ops.zeros(
          [int(inputs.get_shape()[0]), self._num_units], dtype)
      for freq_index in range(len(freq_inputs)):
        if self._state_is_tuple:
          name_prefix = "state_f%02d" % freq_index
          c_prev_time = getattr(state, name_prefix + "_c")
          m_prev_time = getattr(state, name_prefix + "_m")
        else:
          c_prev_time = array_ops.slice(
              state, [0, 2 * freq_index * self._num_units],
              [-1, self._num_units])
          m_prev_time = array_ops.slice(
              state, [0, (2 * freq_index + 1) * self._num_units],
              [-1, self._num_units])

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        cell_inputs = array_ops.concat(1, [freq_inputs[freq_index], m_prev_time,
                                           m_prev_freq])

        # F-LSTM
        lstm_matrix_freq = nn_ops.bias_add(math_ops.matmul(cell_inputs,
                                                           concat_w_f), b_f)
        if self._couple_input_forget_gates:
          i_freq, j_freq, o_freq = array_ops.split(1, num_gates,
                                                   lstm_matrix_freq)
          f_freq = None
        else:
          i_freq, j_freq, f_freq, o_freq = array_ops.split(1, num_gates,
                                                           lstm_matrix_freq)
        # T-LSTM
        if self._share_time_frequency_weights:
          i_time = i_freq
          j_time = j_freq
          f_time = f_freq
          o_time = o_freq
        else:
          lstm_matrix_time = nn_ops.bias_add(math_ops.matmul(cell_inputs,
                                                             concat_w_t), b_t)
          if self._couple_input_forget_gates:
            i_time, j_time, o_time = array_ops.split(1, num_gates,
                                                     lstm_matrix_time)
            f_time = None
          else:
            i_time, j_time, f_time, o_time = array_ops.split(1, 4,
                                                             lstm_matrix_time)

        # F-LSTM c_freq
        # input gate activations
        if self._use_peepholes:
          i_freq_g = sigmoid(i_freq +
                             w_i_diag_freqf * c_prev_freq +
                             w_i_diag_freqt * c_prev_time)
        else:
          i_freq_g = sigmoid(i_freq)
        # forget gate activations
        if self._couple_input_forget_gates:
          f_freq_g = 1.0 - i_freq_g
        else:
          if self._use_peepholes:
            f_freq_g = sigmoid(f_freq + self._forget_bias +
                               w_f_diag_freqf * c_prev_freq +
                               w_f_diag_freqt * c_prev_time)
          else:
            f_freq_g = sigmoid(f_freq + self._forget_bias)
        # cell state
        c_freq = f_freq_g * c_prev_freq + i_freq_g * tanh(j_freq)
        if self._cell_clip is not None:
          # pylint: disable=invalid-unary-operand-type
          c_freq = clip_ops.clip_by_value(c_freq, -self._cell_clip,
                                          self._cell_clip)
          # pylint: enable=invalid-unary-operand-type

        # T-LSTM c_freq
        # input gate activations
        if self._use_peepholes:
          if self._share_time_frequency_weights:
            i_time_g = sigmoid(i_time +
                               w_i_diag_freqf * c_prev_freq +
                               w_i_diag_freqt * c_prev_time)
          else:
            i_time_g = sigmoid(i_time +
                               w_i_diag_timef * c_prev_freq +
                               w_i_diag_timet * c_prev_time)
        else:
          i_time_g = sigmoid(i_time)
        # forget gate activations
        if self._couple_input_forget_gates:
          f_time_g = 1.0 - i_time_g
        else:
          if self._use_peepholes:
            if self._share_time_frequency_weights:
              f_time_g = sigmoid(f_time + self._forget_bias +
                                 w_f_diag_freqf * c_prev_freq +
                                 w_f_diag_freqt * c_prev_time)
            else:
              f_time_g = sigmoid(f_time + self._forget_bias +
                                 w_f_diag_timef * c_prev_freq +
                                 w_f_diag_timet * c_prev_time)
          else:
            f_time_g = sigmoid(f_time + self._forget_bias)
        # cell state
        c_time = f_time_g * c_prev_time + i_time_g * tanh(j_time)
        if self._cell_clip is not None:
          # pylint: disable=invalid-unary-operand-type
          c_time = clip_ops.clip_by_value(c_time, -self._cell_clip,
                                          self._cell_clip)
          # pylint: enable=invalid-unary-operand-type

        # F-LSTM m_freq
        if self._use_peepholes:
          m_freq = sigmoid(o_freq +
                           w_o_diag_freqf * c_freq +
                           w_o_diag_freqt * c_time) * tanh(c_freq)
        else:
          m_freq = sigmoid(o_freq) * tanh(c_freq)

        # T-LSTM m_time
        if self._use_peepholes:
          if self._share_time_frequency_weights:
            m_time = sigmoid(o_time +
                             w_o_diag_freqf * c_freq +
                             w_o_diag_freqt * c_time) * tanh(c_time)
          else:
            m_time = sigmoid(o_time +
                             w_o_diag_timef * c_freq +
                             w_o_diag_timet * c_time) * tanh(c_time)
        else:
          m_time = sigmoid(o_time) * tanh(c_time)

        m_prev_freq = m_freq
        c_prev_freq = c_freq
        # Concatenate the outputs for T-LSTM and F-LSTM for each shift
        if freq_index == 0:
          state_out_lst = [c_time, m_time]
          m_out_lst = [m_time, m_freq]
        else:
          state_out_lst.extend([c_time, m_time])
          m_out_lst.extend([m_time, m_freq])
      if self._state_is_tuple:
        state_out = self._state_tuple_type(*state_out_lst)
      else:
        state_out = array_ops.concat(1, state_out_lst)
      # Outputs are always concated as it is never used separately.
      m_out = array_ops.concat(1, m_out_lst)
    return m_out, state_out

  def _make_tf_features(self, input_feat):
    """Make the frequency features.

    Args:
      input_feat: input Tensor, 2D, batch x num_units.

    Returns:
      A list of frequency features, with each element containing:
      - A 2D, batch x output_dim, Tensor representing the time-frequency feature
        for that frequency index. Here output_dim is feature_size.
    Raises:
      ValueError: if input_size cannot be inferred from static shape inference.
    """
    input_size = input_feat.get_shape().with_rank(2)[-1].value
    if input_size is None:
      raise ValueError("Cannot infer input_size from static shape inference.")
    num_feats = int((input_size - self._feature_size) / (
        self._frequency_skip)) + 1
    if num_feats != self._num_frequency_blocks:
      raise ValueError(
          "Invalid num_frequency_blocks, requires %d but gets %d, please check"
          " the input size and filter config are correct." % (
              self._num_frequency_blocks, num_feats))
    freq_inputs = []
    for f in range(num_feats):
      cur_input = array_ops.slice(input_feat, [0, f*self._frequency_skip],
                                  [-1, self._feature_size])
      freq_inputs.append(cur_input)
    return freq_inputs


# pylint: disable=protected-access
_linear = rnn_cell._linear
# pylint: enable=protected-access


class AttentionCellWrapper(rnn_cell.RNNCell):
  """Basic attention cell wrapper.

  Implementation based on https://arxiv.org/pdf/1601.06733.pdf.
  """

  def __init__(self, cell, attn_length, attn_size=None, attn_vec_size=None,
               input_size=None, state_is_tuple=False):
    """Create a cell with attention.

    Args:
      cell: an RNNCell, an attention is added to it.
      attn_length: integer, the size of an attention window.
      attn_size: integer, the size of an attention vector. Equal to
          cell.output_size by default.
      attn_vec_size: integer, the number of convolutional features calculated
          on attention state and a size of the hidden layer built from
          base cell state. Equal attn_size to by default.
      input_size: integer, the size of a hidden linear layer,
          built from inputs and attention. Derived from the input tensor
          by default.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  By default (False), the states are all
        concatenated along the column axis.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if cell returns a state tuple but the flag
          `state_is_tuple` is `False` or if attn_length is zero or less.
    """
    if not isinstance(cell, rnn_cell.RNNCell):
      raise TypeError("The parameter cell is not RNNCell.")
    if nest.is_sequence(cell.state_size) and not state_is_tuple:
      raise ValueError("Cell returns tuple of states, but the flag "
                       "state_is_tuple is not set. State size is: %s"
                       % str(cell.state_size))
    if attn_length <= 0:
      raise ValueError("attn_length should be greater than zero, got %s"
                       % str(attn_length))
    if not state_is_tuple:
      logging.warn(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use state_is_tuple=True." % self)
    if attn_size is None:
      attn_size = cell.output_size
    if attn_vec_size is None:
      attn_vec_size = attn_size
    self._state_is_tuple = state_is_tuple
    self._cell = cell
    self._attn_vec_size = attn_vec_size
    self._input_size = input_size
    self._attn_size = attn_size
    self._attn_length = attn_length

  @property
  def state_size(self):
    size = (self._cell.state_size, self._attn_size,
            self._attn_size * self._attn_length)
    if self._state_is_tuple:
      return size
    else:
      return sum(list(size))

  @property
  def output_size(self):
    return self._attn_size

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell with attention (LSTMA)."""
    with vs.variable_scope(scope or type(self).__name__):
      if self._state_is_tuple:
        state, attns, attn_states = state
      else:
        states = state
        state = array_ops.slice(states, [0, 0], [-1, self._cell.state_size])
        attns = array_ops.slice(
            states, [0, self._cell.state_size], [-1, self._attn_size])
        attn_states = array_ops.slice(
            states, [0, self._cell.state_size + self._attn_size],
            [-1, self._attn_size * self._attn_length])
      attn_states = array_ops.reshape(attn_states,
                                      [-1, self._attn_length, self._attn_size])
      input_size = self._input_size
      if input_size is None:
        input_size = inputs.get_shape().as_list()[1]
      inputs = _linear([inputs, attns], input_size, True)
      lstm_output, new_state = self._cell(inputs, state)
      if self._state_is_tuple:
        new_state_cat = array_ops.concat(1, nest.flatten(new_state))
      else:
        new_state_cat = new_state
      new_attns, new_attn_states = self._attention(new_state_cat, attn_states)
      with vs.variable_scope("AttnOutputProjection"):
        output = _linear([lstm_output, new_attns], self._attn_size, True)
      new_attn_states = array_ops.concat(1, [new_attn_states,
                                             array_ops.expand_dims(output, 1)])
      new_attn_states = array_ops.reshape(
          new_attn_states, [-1, self._attn_length * self._attn_size])
      new_state = (new_state, new_attns, new_attn_states)
      if not self._state_is_tuple:
        new_state = array_ops.concat(1, list(new_state))
      return output, new_state

  def _attention(self, query, attn_states):
    conv2d = nn_ops.conv2d
    reduce_sum = math_ops.reduce_sum
    softmax = nn_ops.softmax
    tanh = math_ops.tanh

    with vs.variable_scope("Attention"):
      k = vs.get_variable("AttnW", [1, 1, self._attn_size, self._attn_vec_size])
      v = vs.get_variable("AttnV", [self._attn_vec_size])
      hidden = array_ops.reshape(attn_states,
                                 [-1, self._attn_length, 1, self._attn_size])
      hidden_features = conv2d(hidden, k, [1, 1, 1, 1], "SAME")
      y = _linear(query, self._attn_vec_size, True)
      y = array_ops.reshape(y, [-1, 1, 1, self._attn_vec_size])
      s = reduce_sum(v * tanh(hidden_features + y), [2, 3])
      a = softmax(s)
      d = reduce_sum(
          array_ops.reshape(a, [-1, self._attn_length, 1, 1]) * hidden, [1, 2])
      new_attns = array_ops.reshape(d, [-1, self._attn_size])
      new_attn_states = array_ops.slice(attn_states, [0, 1, 0], [-1, -1, -1])
      return new_attns, new_attn_states


class LayerNormBasicLSTMCell(rnn_cell.RNNCell):
  """LSTM unit with layer normalization and recurrent dropout.

  This class adds layer normalization and recurrent dropout to a
  basic LSTM unit. Layer normalization implementation is based on:

    https://arxiv.org/abs/1607.06450.

  "Layer Normalization"
  Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

  and is applied before the internal nonlinearities.
  Recurrent dropout is base on:

    https://arxiv.org/abs/1603.05118

  "Recurrent Dropout without Memory Loss"
  Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.
  """

  def __init__(self, num_units, forget_bias=1.0,
               input_size=None, activation=math_ops.tanh,
               layer_norm=True, norm_gain=1.0, norm_shift=0.0,
               dropout_keep_prob=1.0, dropout_prob_seed=None):
    """Initializes the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      activation: Activation function of the inner states.
      layer_norm: If `True`, layer normalization will be applied.
      norm_gain: float, The layer normalization gain initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      norm_shift: float, The layer normalization shift initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      dropout_keep_prob: unit Tensor or float between 0 and 1 representing the
        recurrent dropout probability value. If float and 1.0, no dropout will
        be applied.
      dropout_prob_seed: (optional) integer, the randomness seed.
    """

    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)

    self._num_units = num_units
    self._activation = activation
    self._forget_bias = forget_bias
    self._keep_prob = dropout_keep_prob
    self._seed = dropout_prob_seed
    self._layer_norm = layer_norm
    self._g = norm_gain
    self._b = norm_shift

  @property
  def state_size(self):
    return rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def _norm(self, inp, scope):
    with vs.variable_scope(scope) as scope:
      shape = inp.get_shape()[-1:]
      gamma_init = init_ops.constant_initializer(self._g)
      beta_init = init_ops.constant_initializer(self._b)
      gamma = vs.get_variable("gamma", shape=shape, initializer=gamma_init)  # pylint: disable=unused-variable
      beta = vs.get_variable("beta", shape=shape, initializer=beta_init)  # pylint: disable=unused-variable
      normalized = layers.layer_norm(inp, reuse=True, scope=scope)
      return normalized

  def _linear(self, args, scope="linear"):
    out_size = 4 * self._num_units
    proj_size = args.get_shape()[-1]
    with vs.variable_scope(scope) as scope:
      weights = vs.get_variable("weights", [proj_size, out_size])
      out = math_ops.matmul(args, weights)
      if not self._layer_norm:
        bias = vs.get_variable("b", [out_size])
        out += bias
      return out

  def __call__(self, inputs, state, scope=None):
    """LSTM cell with layer normalization and recurrent dropout."""

    with vs.variable_scope(scope or type(self).__name__) as scope:  # LayerNormBasicLSTMCell  # pylint: disable=unused-variables
      c, h = state
      args = array_ops.concat(1, [inputs, h])
      concat = self._linear(args)

      i, j, f, o = array_ops.split(1, 4, concat)
      if self._layer_norm:
        i = self._norm(i, "input")
        j = self._norm(j, "transform")
        f = self._norm(f, "forget")
        o = self._norm(o, "output")

      g = self._activation(j)
      if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
        g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)

      new_c = (c * math_ops.sigmoid(f + self._forget_bias)
               + math_ops.sigmoid(i) * g)
      if self._layer_norm:
        new_c = self._norm(new_c, "state")
      new_h = self._activation(new_c) * math_ops.sigmoid(o)

      new_state = rnn_cell.LSTMStateTuple(new_c, new_h)
      return new_h, new_state
