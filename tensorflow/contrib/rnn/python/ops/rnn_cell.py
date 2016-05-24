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

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh


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
          initializer=array_ops.zeros_initializer, dtype=dtype)

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
               feature_size=None, frequency_skip=None):
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
    """
    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._share_time_frequency_weights = share_time_frequency_weights
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

    freq_inputs = self._make_tf_features(inputs)
    dtype = inputs.dtype
    actual_input_size = freq_inputs[0].get_shape().as_list()[1]
    with vs.variable_scope(scope or type(self).__name__,
                           initializer=self._initializer):  # "GridLSTMCell"
      concat_w_f = _get_concat_variable(
          "W_f", [actual_input_size + 2*self._num_units, 4 * self._num_units],
          dtype, self._num_unit_shards)
      b_f = vs.get_variable(
          "B_f", shape=[4 * self._num_units],
          initializer=array_ops.zeros_initializer, dtype=dtype)
      if not self._share_time_frequency_weights:
        concat_w_t = _get_concat_variable(
            "W_t", [actual_input_size + 2*self._num_units, 4 * self._num_units],
            dtype, self._num_unit_shards)
        b_t = vs.get_variable(
            "B_t", shape=[4 * self._num_units],
            initializer=array_ops.zeros_initializer, dtype=dtype)

      if self._use_peepholes:
        # Diagonal connections
        w_f_diag_freqf = vs.get_variable(
            "W_F_diag_freqf", shape=[self._num_units], dtype=dtype)
        w_i_diag_freqf = vs.get_variable(
            "W_I_diag_freqf", shape=[self._num_units], dtype=dtype)
        w_o_diag_freqf = vs.get_variable(
            "W_O_diag_freqf", shape=[self._num_units], dtype=dtype)
        w_f_diag_freqt = vs.get_variable(
            "W_F_diag_freqt", shape=[self._num_units], dtype=dtype)
        w_i_diag_freqt = vs.get_variable(
            "W_I_diag_freqt", shape=[self._num_units], dtype=dtype)
        w_o_diag_freqt = vs.get_variable(
            "W_O_diag_freqt", shape=[self._num_units], dtype=dtype)
        if not self._share_time_frequency_weights:
          w_f_diag_timef = vs.get_variable(
              "W_F_diag_timef", shape=[self._num_units], dtype=dtype)
          w_i_diag_timef = vs.get_variable(
              "W_I_diag_timef", shape=[self._num_units], dtype=dtype)
          w_o_diag_timef = vs.get_variable(
              "W_O_diag_timef", shape=[self._num_units], dtype=dtype)
          w_f_diag_timet = vs.get_variable(
              "W_F_diag_timet", shape=[self._num_units], dtype=dtype)
          w_i_diag_timet = vs.get_variable(
              "W_I_diag_timet", shape=[self._num_units], dtype=dtype)
          w_o_diag_timet = vs.get_variable(
              "W_O_diag_timet", shape=[self._num_units], dtype=dtype)

      # initialize the first freq state to be zero
      m_prev_freq = array_ops.zeros([int(inputs.get_shape()[0]),
                                     self._num_units], dtype)
      c_prev_freq = array_ops.zeros([int(inputs.get_shape()[0]),
                                     self._num_units], dtype)
      for freq_index in range(len(freq_inputs)):
        c_prev_time = array_ops.slice(state, [0, 2 * freq_index *
                                              self._num_units],
                                      [-1, self._num_units])
        m_prev_time = array_ops.slice(state, [0, (2 * freq_index + 1) *
                                              self._num_units],
                                      [-1, self._num_units])

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        cell_inputs = array_ops.concat(1, [freq_inputs[freq_index], m_prev_time,
                                           m_prev_freq])

        # F-LSTM
        lstm_matrix_freq = nn_ops.bias_add(math_ops.matmul(cell_inputs,
                                                           concat_w_f), b_f)
        i_freq, j_freq, f_freq, o_freq = array_ops.split(1, 4, lstm_matrix_freq)
        # T-LSTM
        if self._share_time_frequency_weights:
          i_time = i_freq
          j_time = j_freq
          f_time = f_freq
          o_time = o_freq
        else:
          lstm_matrix_time = nn_ops.bias_add(math_ops.matmul(cell_inputs,
                                                             concat_w_t), b_t)
          i_time, j_time, f_time, o_time = array_ops.split(1, 4,
                                                           lstm_matrix_time)

        # F-LSTM c_freq
        if self._use_peepholes:
          c_freq = (sigmoid(f_freq + self._forget_bias + w_f_diag_freqf * (
              c_prev_freq) + w_f_diag_freqt * c_prev_time) * c_prev_freq +
                    sigmoid(i_freq + w_i_diag_freqf * c_prev_freq + (
                        w_i_diag_freqt * c_prev_time)) * tanh(j_freq))
        else:
          c_freq = (sigmoid(f_freq + self._forget_bias) * c_prev_freq +
                    sigmoid(i_freq) * tanh(j_freq))
        if self._cell_clip is not None:
          # pylint: disable=invalid-unary-operand-type
          c_freq = clip_ops.clip_by_value(c_freq, -self._cell_clip,
                                          self._cell_clip)
          # pylint: enable=invalid-unary-operand-type

        # T-LSTM c_freq
        if self._use_peepholes:
          if self._share_time_frequency_weights:
            c_time = sigmoid(f_time + self._forget_bias + w_f_diag_freqf * (
                c_prev_freq + w_f_diag_freqt * c_prev_time)) * c_prev_time + (
                    sigmoid(i_time + w_i_diag_freqf * c_prev_freq + (
                        w_i_diag_freqt * c_prev_time)) * tanh(j_time))
          else:
            c_time = sigmoid(f_time + self._forget_bias + w_f_diag_timef * (
                c_prev_time + w_f_diag_timet * c_prev_time)) * c_prev_time + (
                    sigmoid(i_time + w_i_diag_timef * c_prev_freq + (
                        w_i_diag_timet * c_prev_time)) * tanh(j_time))
        else:
          c_time = (sigmoid(f_time + self._forget_bias) * c_prev_time +
                    sigmoid(i_time) * tanh(j_time))

        if self._cell_clip is not None:
          # pylint: disable=invalid-unary-operand-type
          c_time = clip_ops.clip_by_value(c_freq, -self._cell_clip,
                                          self._cell_clip)
          # pylint: enable=invalid-unary-operand-type

        # F-LSTM m_freq
        if self._use_peepholes:
          m_freq = sigmoid(o_freq + w_o_diag_freqf * c_freq +
                           w_o_diag_freqt * c_time) * tanh(c_freq)
        else:
          m_freq = sigmoid(o_freq) * tanh(c_freq)

        # T-LSTM m_time
        if self._use_peepholes:
          if self._share_time_frequency_weights:
            m_time = sigmoid(o_time + w_o_diag_freqf * c_freq +
                             w_o_diag_freqt * c_time) * tanh(c_time)
          else:
            m_time = sigmoid(o_time + w_o_diag_timef * c_freq +
                             w_o_diag_timet * c_time) * tanh(c_time)
        else:
          m_time = sigmoid(o_time) * tanh(c_time)

        m_prev_freq = m_freq
        c_prev_freq = c_freq
        # Concatenate the outputs for T-LSTM and F-LSTM for each shift
        if freq_index == 0:
          state_out = array_ops.concat(1, [c_time, m_time])
          m_out = array_ops.concat(1, [m_time, m_freq])
        else:
          state_out = array_ops.concat(1, [state_out, c_time, m_time])
          m_out = array_ops.concat(1, [m_out, m_time, m_freq])
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
