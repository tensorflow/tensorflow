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

from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import partitioned_variables
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

  concat_variable = array_ops.concat(sharded_variable, 0, name=concat_name)
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


def _norm(g, b, inp, scope):
  shape = inp.get_shape()[-1:]
  gamma_init = init_ops.constant_initializer(g)
  beta_init = init_ops.constant_initializer(b)
  with vs.variable_scope(scope):
    # Initialize beta and gamma for use by layer_norm.
    vs.get_variable("gamma", shape=shape, initializer=gamma_init)
    vs.get_variable("beta", shape=shape, initializer=beta_init)
  normalized = layers.layer_norm(inp, reuse=True, scope=scope)
  return normalized


class CoupledInputForgetGateLSTMCell(rnn_cell_impl.RNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.

  The default non-peephole implementation is based on:

    http://www.bioinf.jku.at/publications/older/2604.pdf

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
  Layer normalization implementation is based on:

    https://arxiv.org/abs/1607.06450.

  "Layer Normalization"
  Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

  and is applied before the internal nonlinearities.

  """

  def __init__(self,
               num_units,
               use_peepholes=False,
               initializer=None,
               num_proj=None,
               proj_clip=None,
               num_unit_shards=1,
               num_proj_shards=1,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=math_ops.tanh,
               reuse=None,
               layer_norm=False,
               norm_gain=1.0,
               norm_shift=0.0):
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
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      layer_norm: If `True`, layer normalization will be applied.
      norm_gain: float, The layer normalization gain initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      norm_shift: float, The layer normalization shift initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
    """
    super(CoupledInputForgetGateLSTMCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      logging.warn(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use state_is_tuple=True.", self)
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
    self._reuse = reuse
    self._layer_norm = layer_norm
    self._norm_gain = norm_gain
    self._norm_shift = norm_shift

    if num_proj:
      self._state_size = (rnn_cell_impl.LSTMStateTuple(num_units, num_proj)
                          if state_is_tuple else num_units + num_proj)
      self._output_size = num_proj
    else:
      self._state_size = (rnn_cell_impl.LSTMStateTuple(num_units, num_units)
                          if state_is_tuple else 2 * num_units)
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def call(self, inputs, state):
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.

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
    concat_w = _get_concat_variable(
        "W", [input_size.value + num_proj, 3 * self._num_units],
        dtype, self._num_unit_shards)

    b = vs.get_variable(
        "B",
        shape=[3 * self._num_units],
        initializer=init_ops.zeros_initializer(),
        dtype=dtype)

    # j = new_input, f = forget_gate, o = output_gate
    cell_inputs = array_ops.concat([inputs, m_prev], 1)
    lstm_matrix = math_ops.matmul(cell_inputs, concat_w)

    # If layer nomalization is applied, do not add bias
    if not self._layer_norm:
      lstm_matrix = nn_ops.bias_add(lstm_matrix, b)

    j, f, o = array_ops.split(value=lstm_matrix, num_or_size_splits=3, axis=1)

    # Apply layer normalization
    if self._layer_norm:
      j = _norm(self._norm_gain, self._norm_shift, j, "transform")
      f = _norm(self._norm_gain, self._norm_shift, f, "forget")
      o = _norm(self._norm_gain, self._norm_shift, o, "output")

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

    # Apply layer normalization
    if self._layer_norm:
      c = _norm(self._norm_gain, self._norm_shift, c, "state")

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

    new_state = (rnn_cell_impl.LSTMStateTuple(c, m)
                 if self._state_is_tuple else array_ops.concat([c, m], 1))
    return m, new_state


class TimeFreqLSTMCell(rnn_cell_impl.RNNCell):
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
               feature_size=None, frequency_skip=None,
               reuse=None):
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
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(TimeFreqLSTMCell, self).__init__(_reuse=reuse)
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
    self._reuse = reuse

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def call(self, inputs, state):
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: state Tensor, 2D, batch x state_size.

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

    concat_w = _get_concat_variable(
        "W", [actual_input_size + 2*self._num_units, 4 * self._num_units],
        dtype, self._num_unit_shards)

    b = vs.get_variable(
        "B",
        shape=[4 * self._num_units],
        initializer=init_ops.zeros_initializer(),
        dtype=dtype)

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
      cell_inputs = array_ops.concat([freq_inputs[fq], m_prev, m_prev_freq],
                                     1)
      lstm_matrix = nn_ops.bias_add(math_ops.matmul(cell_inputs, concat_w), b)
      i, j, f, o = array_ops.split(
          value=lstm_matrix, num_or_size_splits=4, axis=1)

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
        state_out = array_ops.concat([c, m], 1)
        m_out = m
      else:
        state_out = array_ops.concat([state_out, c, m], 1)
        m_out = array_ops.concat([m_out, m], 1)
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


class GridLSTMCell(rnn_cell_impl.RNNCell):
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
               num_frequency_blocks=None,
               start_freqindex_list=None,
               end_freqindex_list=None,
               couple_input_forget_gates=False,
               state_is_tuple=True,
               reuse=None):
    """Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell
      use_peepholes: (optional) bool, default False. Set True to enable
        diagonal/peephole connections.
      share_time_frequency_weights: (optional) bool, default False. Set True to
        enable shared cell weights between time and frequency LSTMs.
      cell_clip: (optional) A float value, default None, if provided the cell
        state is clipped by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices, default None.
      num_unit_shards: (optional) int, default 1, How to split the weight
        matrix. If > 1,the weight matrix is stored across num_unit_shards.
      forget_bias: (optional) float, default 1.0, The initial bias of the
        forget gates, used to reduce the scale of forgetting at the beginning
        of the training.
      feature_size: (optional) int, default None, The size of the input feature
        the LSTM spans over.
      frequency_skip: (optional) int, default None, The amount the LSTM filter
        is shifted by in frequency.
      num_frequency_blocks: [required] A list of frequency blocks needed to
        cover the whole input feature splitting defined by start_freqindex_list
        and end_freqindex_list.
      start_freqindex_list: [optional], list of ints, default None,  The
        starting frequency index for each frequency block.
      end_freqindex_list: [optional], list of ints, default None. The ending
        frequency index for each frequency block.
      couple_input_forget_gates: (optional) bool, default False, Whether to
        couple the input and forget gates, i.e. f_gate = 1.0 - i_gate, to reduce
        model parameters and computation cost.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  By default (False), they are concatenated
        along the column axis.  This default behavior will soon be deprecated.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    Raises:
      ValueError: if the num_frequency_blocks list is not specified
    """
    super(GridLSTMCell, self).__init__(_reuse=reuse)
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
    self._start_freqindex_list = start_freqindex_list
    self._end_freqindex_list = end_freqindex_list
    self._num_frequency_blocks = num_frequency_blocks
    self._total_blocks = 0
    self._reuse = reuse
    if self._num_frequency_blocks is None:
      raise ValueError("Must specify num_frequency_blocks")

    for block_index in range(len(self._num_frequency_blocks)):
      self._total_blocks += int(self._num_frequency_blocks[block_index])
    if state_is_tuple:
      state_names = ""
      for block_index in range(len(self._num_frequency_blocks)):
        for freq_index in range(self._num_frequency_blocks[block_index]):
          name_prefix = "state_f%02d_b%02d" % (freq_index, block_index)
          state_names += ("%s_c, %s_m," % (name_prefix, name_prefix))
      self._state_tuple_type = collections.namedtuple(
          "GridLSTMStateTuple", state_names.strip(","))
      self._state_size = self._state_tuple_type(
          *([num_units, num_units] * self._total_blocks))
    else:
      self._state_tuple_type = None
      self._state_size = num_units * self._total_blocks * 2
    self._output_size = num_units * self._total_blocks * 2

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  @property
  def state_tuple_type(self):
    return self._state_tuple_type

  def call(self, inputs, state):
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, 2D, [batch, feature_size].
      state: Tensor or tuple of Tensors, 2D, [batch, state_size], depends on the
        flag self._state_is_tuple.

    Returns:
      A tuple containing:
      - A 2D, [batch, output_dim], Tensor representing the output of the LSTM
        after reading "inputs" when previous state was "state".
        Here output_dim is num_units.
      - A 2D, [batch, state_size], Tensor representing the new state of LSTM
        after reading "inputs" when previous state was "state".
    Raises:
      ValueError: if an input_size was specified and the provided inputs have
        a different dimension.
    """
    batch_size = inputs.shape[0].value or array_ops.shape(inputs)[0]
    freq_inputs = self._make_tf_features(inputs)
    m_out_lst = []
    state_out_lst = []
    for block in range(len(freq_inputs)):
      m_out_lst_current, state_out_lst_current = self._compute(
          freq_inputs[block], block, state, batch_size,
          state_is_tuple=self._state_is_tuple)
      m_out_lst.extend(m_out_lst_current)
      state_out_lst.extend(state_out_lst_current)
    if self._state_is_tuple:
      state_out = self._state_tuple_type(*state_out_lst)
    else:
      state_out = array_ops.concat(state_out_lst, 1)
    m_out = array_ops.concat(m_out_lst, 1)
    return m_out, state_out

  def _compute(self, freq_inputs, block, state, batch_size,
               state_prefix="state",
               state_is_tuple=True):
    """Run the actual computation of one step LSTM.

    Args:
      freq_inputs: list of Tensors, 2D, [batch, feature_size].
      block: int, current frequency block index to process.
      state: Tensor or tuple of Tensors, 2D, [batch, state_size], it depends on
        the flag state_is_tuple.
      batch_size: int32, batch size.
      state_prefix: (optional) string, name prefix for states, defaults to
        "state".
      state_is_tuple: boolean, indicates whether the state is a tuple or Tensor.

    Returns:
      A tuple, containing:
      - A list of [batch, output_dim] Tensors, representing the output of the
        LSTM given the inputs and state.
      - A list of [batch, state_size] Tensors, representing the LSTM state
        values given the inputs and previous state.
    """
    sigmoid = math_ops.sigmoid
    tanh = math_ops.tanh
    num_gates = 3 if self._couple_input_forget_gates else 4
    dtype = freq_inputs[0].dtype
    actual_input_size = freq_inputs[0].get_shape().as_list()[1]

    concat_w_f = _get_concat_variable(
        "W_f_%d" % block, [actual_input_size + 2 * self._num_units,
                           num_gates * self._num_units],
        dtype, self._num_unit_shards)
    b_f = vs.get_variable(
        "B_f_%d" % block,
        shape=[num_gates * self._num_units],
        initializer=init_ops.zeros_initializer(),
        dtype=dtype)
    if not self._share_time_frequency_weights:
      concat_w_t = _get_concat_variable(
          "W_t_%d" % block, [actual_input_size + 2 * self._num_units,
                             num_gates * self._num_units],
          dtype, self._num_unit_shards)
      b_t = vs.get_variable(
          "B_t_%d" % block,
          shape=[num_gates * self._num_units],
          initializer=init_ops.zeros_initializer(),
          dtype=dtype)

    if self._use_peepholes:
      # Diagonal connections
      if not self._couple_input_forget_gates:
        w_f_diag_freqf = vs.get_variable(
            "W_F_diag_freqf_%d" % block, shape=[self._num_units], dtype=dtype)
        w_f_diag_freqt = vs.get_variable(
            "W_F_diag_freqt_%d"% block, shape=[self._num_units], dtype=dtype)
      w_i_diag_freqf = vs.get_variable(
          "W_I_diag_freqf_%d" % block, shape=[self._num_units], dtype=dtype)
      w_i_diag_freqt = vs.get_variable(
          "W_I_diag_freqt_%d" % block, shape=[self._num_units], dtype=dtype)
      w_o_diag_freqf = vs.get_variable(
          "W_O_diag_freqf_%d" % block, shape=[self._num_units], dtype=dtype)
      w_o_diag_freqt = vs.get_variable(
          "W_O_diag_freqt_%d" % block, shape=[self._num_units], dtype=dtype)
      if not self._share_time_frequency_weights:
        if not self._couple_input_forget_gates:
          w_f_diag_timef = vs.get_variable(
              "W_F_diag_timef_%d" % block, shape=[self._num_units], dtype=dtype)
          w_f_diag_timet = vs.get_variable(
              "W_F_diag_timet_%d" % block, shape=[self._num_units], dtype=dtype)
        w_i_diag_timef = vs.get_variable(
            "W_I_diag_timef_%d" % block, shape=[self._num_units], dtype=dtype)
        w_i_diag_timet = vs.get_variable(
            "W_I_diag_timet_%d" % block, shape=[self._num_units], dtype=dtype)
        w_o_diag_timef = vs.get_variable(
            "W_O_diag_timef_%d" % block, shape=[self._num_units], dtype=dtype)
        w_o_diag_timet = vs.get_variable(
            "W_O_diag_timet_%d" % block, shape=[self._num_units], dtype=dtype)

    # initialize the first freq state to be zero
    m_prev_freq = array_ops.zeros([batch_size, self._num_units], dtype)
    c_prev_freq = array_ops.zeros([batch_size, self._num_units], dtype)
    for freq_index in range(len(freq_inputs)):
      if state_is_tuple:
        name_prefix = "%s_f%02d_b%02d" % (state_prefix, freq_index, block)
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
      cell_inputs = array_ops.concat(
          [freq_inputs[freq_index], m_prev_time, m_prev_freq], 1)

      # F-LSTM
      lstm_matrix_freq = nn_ops.bias_add(math_ops.matmul(cell_inputs,
                                                         concat_w_f), b_f)
      if self._couple_input_forget_gates:
        i_freq, j_freq, o_freq = array_ops.split(
            value=lstm_matrix_freq, num_or_size_splits=num_gates, axis=1)
        f_freq = None
      else:
        i_freq, j_freq, f_freq, o_freq = array_ops.split(
            value=lstm_matrix_freq, num_or_size_splits=num_gates, axis=1)
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
          i_time, j_time, o_time = array_ops.split(
              value=lstm_matrix_time, num_or_size_splits=num_gates, axis=1)
          f_time = None
        else:
          i_time, j_time, f_time, o_time = array_ops.split(
              value=lstm_matrix_time, num_or_size_splits=num_gates, axis=1)

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

    return m_out_lst, state_out_lst

  def _make_tf_features(self, input_feat, slice_offset=0):
    """Make the frequency features.

    Args:
      input_feat: input Tensor, 2D, [batch, num_units].
      slice_offset: (optional) Python int, default 0, the slicing offset is only
        used for the backward processing in the BidirectionalGridLSTMCell. It
        specifies a different starting point instead of always 0 to enable the
        forward and backward processing look at different frequency blocks.

    Returns:
      A list of frequency features, with each element containing:
      - A 2D, [batch, output_dim], Tensor representing the time-frequency
        feature for that frequency index. Here output_dim is feature_size.
    Raises:
      ValueError: if input_size cannot be inferred from static shape inference.
    """
    input_size = input_feat.get_shape().with_rank(2)[-1].value
    if input_size is None:
      raise ValueError("Cannot infer input_size from static shape inference.")
    if slice_offset > 0:
      # Padding to the end
      inputs = array_ops.pad(
          input_feat, array_ops.constant([0, 0, 0, slice_offset], shape=[2, 2],
                                         dtype=dtypes.int32),
          "CONSTANT")
    elif slice_offset < 0:
      # Padding to the front
      inputs = array_ops.pad(
          input_feat, array_ops.constant([0, 0, -slice_offset, 0], shape=[2, 2],
                                         dtype=dtypes.int32),
          "CONSTANT")
      slice_offset = 0
    else:
      inputs = input_feat
    freq_inputs = []
    if not self._start_freqindex_list:
      if len(self._num_frequency_blocks) != 1:
        raise ValueError("Length of num_frequency_blocks"
                         " is not 1, but instead is %d",
                         len(self._num_frequency_blocks))
      num_feats = int((input_size - self._feature_size) / (
          self._frequency_skip)) + 1
      if num_feats != self._num_frequency_blocks[0]:
        raise ValueError(
            "Invalid num_frequency_blocks, requires %d but gets %d, please"
            " check the input size and filter config are correct." % (
                self._num_frequency_blocks[0], num_feats))
      block_inputs = []
      for f in range(num_feats):
        cur_input = array_ops.slice(
            inputs, [0, slice_offset + f * self._frequency_skip],
            [-1, self._feature_size])
        block_inputs.append(cur_input)
      freq_inputs.append(block_inputs)
    else:
      if len(self._start_freqindex_list) != len(self._end_freqindex_list):
        raise ValueError("Length of start and end freqindex_list"
                         " does not match %d %d",
                         len(self._start_freqindex_list),
                         len(self._end_freqindex_list))
      if len(self._num_frequency_blocks) != len(self._start_freqindex_list):
        raise ValueError("Length of num_frequency_blocks"
                         " is not equal to start_freqindex_list %d %d",
                         len(self._num_frequency_blocks),
                         len(self._start_freqindex_list))
      for b in range(len(self._start_freqindex_list)):
        start_index = self._start_freqindex_list[b]
        end_index = self._end_freqindex_list[b]
        cur_size = end_index - start_index
        block_feats = int((cur_size - self._feature_size) / (
            self._frequency_skip)) + 1
        if block_feats != self._num_frequency_blocks[b]:
          raise ValueError(
              "Invalid num_frequency_blocks, requires %d but gets %d, please"
              " check the input size and filter config are correct." % (
                  self._num_frequency_blocks[b], block_feats))
        block_inputs = []
        for f in range(block_feats):
          cur_input = array_ops.slice(
              inputs, [0, start_index + slice_offset + f *
                       self._frequency_skip],
              [-1, self._feature_size])
          block_inputs.append(cur_input)
        freq_inputs.append(block_inputs)
    return freq_inputs


class BidirectionalGridLSTMCell(GridLSTMCell):
  """Bidirectional GridLstm cell.

  The bidirection connection is only used in the frequency direction, which
  hence doesn't affect the time direction's real-time processing that is
  required for online recognition systems.
  The current implementation uses different weights for the two directions.
  """

  def __init__(self, num_units, use_peepholes=False,
               share_time_frequency_weights=False,
               cell_clip=None, initializer=None,
               num_unit_shards=1, forget_bias=1.0,
               feature_size=None, frequency_skip=None,
               num_frequency_blocks=None,
               start_freqindex_list=None,
               end_freqindex_list=None,
               couple_input_forget_gates=False,
               backward_slice_offset=0,
               reuse=None):
    """Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell
      use_peepholes: (optional) bool, default False. Set True to enable
        diagonal/peephole connections.
      share_time_frequency_weights: (optional) bool, default False. Set True to
        enable shared cell weights between time and frequency LSTMs.
      cell_clip: (optional) A float value, default None, if provided the cell
        state is clipped by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices, default None.
      num_unit_shards: (optional) int, default 1, How to split the weight
        matrix. If > 1,the weight matrix is stored across num_unit_shards.
      forget_bias: (optional) float, default 1.0, The initial bias of the
        forget gates, used to reduce the scale of forgetting at the beginning
        of the training.
      feature_size: (optional) int, default None, The size of the input feature
        the LSTM spans over.
      frequency_skip: (optional) int, default None, The amount the LSTM filter
        is shifted by in frequency.
      num_frequency_blocks: [required] A list of frequency blocks needed to
        cover the whole input feature splitting defined by start_freqindex_list
        and end_freqindex_list.
      start_freqindex_list: [optional], list of ints, default None,  The
        starting frequency index for each frequency block.
      end_freqindex_list: [optional], list of ints, default None. The ending
        frequency index for each frequency block.
      couple_input_forget_gates: (optional) bool, default False, Whether to
        couple the input and forget gates, i.e. f_gate = 1.0 - i_gate, to reduce
        model parameters and computation cost.
      backward_slice_offset: (optional) int32, default 0, the starting offset to
        slice the feature for backward processing.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(BidirectionalGridLSTMCell, self).__init__(
        num_units, use_peepholes, share_time_frequency_weights, cell_clip,
        initializer, num_unit_shards, forget_bias, feature_size, frequency_skip,
        num_frequency_blocks, start_freqindex_list, end_freqindex_list,
        couple_input_forget_gates, True, reuse)
    self._backward_slice_offset = int(backward_slice_offset)
    state_names = ""
    for direction in ["fwd", "bwd"]:
      for block_index in range(len(self._num_frequency_blocks)):
        for freq_index in range(self._num_frequency_blocks[block_index]):
          name_prefix = "%s_state_f%02d_b%02d" % (direction, freq_index,
                                                  block_index)
          state_names += ("%s_c, %s_m," % (name_prefix, name_prefix))
    self._state_tuple_type = collections.namedtuple(
        "BidirectionalGridLSTMStateTuple", state_names.strip(","))
    self._state_size = self._state_tuple_type(
        *([num_units, num_units] * self._total_blocks * 2))
    self._output_size = 2 * num_units * self._total_blocks * 2

  def call(self, inputs, state):
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, 2D, [batch, num_units].
      state: tuple of Tensors, 2D, [batch, state_size].

    Returns:
      A tuple containing:
      - A 2D, [batch, output_dim], Tensor representing the output of the LSTM
        after reading "inputs" when previous state was "state".
        Here output_dim is num_units.
      - A 2D, [batch, state_size], Tensor representing the new state of LSTM
        after reading "inputs" when previous state was "state".
    Raises:
      ValueError: if an input_size was specified and the provided inputs have
        a different dimension.
    """
    batch_size = inputs.shape[0].value or array_ops.shape(inputs)[0]
    fwd_inputs = self._make_tf_features(inputs)
    if self._backward_slice_offset:
      bwd_inputs = self._make_tf_features(inputs, self._backward_slice_offset)
    else:
      bwd_inputs = fwd_inputs

    # Forward processing
    with vs.variable_scope("fwd"):
      fwd_m_out_lst = []
      fwd_state_out_lst = []
      for block in range(len(fwd_inputs)):
        fwd_m_out_lst_current, fwd_state_out_lst_current = self._compute(
            fwd_inputs[block], block, state, batch_size,
            state_prefix="fwd_state", state_is_tuple=True)
        fwd_m_out_lst.extend(fwd_m_out_lst_current)
        fwd_state_out_lst.extend(fwd_state_out_lst_current)
    # Backward processing
    bwd_m_out_lst = []
    bwd_state_out_lst = []
    with vs.variable_scope("bwd"):
      for block in range(len(bwd_inputs)):
        # Reverse the blocks
        bwd_inputs_reverse = bwd_inputs[block][::-1]
        bwd_m_out_lst_current, bwd_state_out_lst_current = self._compute(
            bwd_inputs_reverse, block, state, batch_size,
            state_prefix="bwd_state", state_is_tuple=True)
        bwd_m_out_lst.extend(bwd_m_out_lst_current)
        bwd_state_out_lst.extend(bwd_state_out_lst_current)
    state_out = self._state_tuple_type(*(fwd_state_out_lst + bwd_state_out_lst))
    # Outputs are always concated as it is never used separately.
    m_out = array_ops.concat(fwd_m_out_lst + bwd_m_out_lst, 1)
    return m_out, state_out


# pylint: disable=protected-access
_Linear = core_rnn_cell._Linear  # pylint: disable=invalid-name
# pylint: enable=protected-access


class AttentionCellWrapper(rnn_cell_impl.RNNCell):
  """Basic attention cell wrapper.

  Implementation based on https://arxiv.org/abs/1409.0473.
  """

  def __init__(self, cell, attn_length, attn_size=None, attn_vec_size=None,
               input_size=None, state_is_tuple=True, reuse=None):
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
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if cell returns a state tuple but the flag
          `state_is_tuple` is `False` or if attn_length is zero or less.
    """
    super(AttentionCellWrapper, self).__init__(_reuse=reuse)
    if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
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
          "deprecated.  Use state_is_tuple=True.", self)
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
    self._reuse = reuse
    self._linear1 = None
    self._linear2 = None
    self._linear3 = None

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

  def call(self, inputs, state):
    """Long short-term memory cell with attention (LSTMA)."""
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
    if self._linear1 is None:
      self._linear1 = _Linear([inputs, attns], input_size, True)
    inputs = self._linear1([inputs, attns])
    cell_output, new_state = self._cell(inputs, state)
    if self._state_is_tuple:
      new_state_cat = array_ops.concat(nest.flatten(new_state), 1)
    else:
      new_state_cat = new_state
    new_attns, new_attn_states = self._attention(new_state_cat, attn_states)
    with vs.variable_scope("attn_output_projection"):
      if self._linear2 is None:
        self._linear2 = _Linear([cell_output, new_attns], self._attn_size, True)
      output = self._linear2([cell_output, new_attns])
    new_attn_states = array_ops.concat(
        [new_attn_states, array_ops.expand_dims(output, 1)], 1)
    new_attn_states = array_ops.reshape(
        new_attn_states, [-1, self._attn_length * self._attn_size])
    new_state = (new_state, new_attns, new_attn_states)
    if not self._state_is_tuple:
      new_state = array_ops.concat(list(new_state), 1)
    return output, new_state

  def _attention(self, query, attn_states):
    conv2d = nn_ops.conv2d
    reduce_sum = math_ops.reduce_sum
    softmax = nn_ops.softmax
    tanh = math_ops.tanh

    with vs.variable_scope("attention"):
      k = vs.get_variable(
          "attn_w", [1, 1, self._attn_size, self._attn_vec_size])
      v = vs.get_variable("attn_v", [self._attn_vec_size])
      hidden = array_ops.reshape(attn_states,
                                 [-1, self._attn_length, 1, self._attn_size])
      hidden_features = conv2d(hidden, k, [1, 1, 1, 1], "SAME")
      if self._linear3 is None:
        self._linear3 = _Linear(query, self._attn_vec_size, True)
      y = self._linear3(query)
      y = array_ops.reshape(y, [-1, 1, 1, self._attn_vec_size])
      s = reduce_sum(v * tanh(hidden_features + y), [2, 3])
      a = softmax(s)
      d = reduce_sum(
          array_ops.reshape(a, [-1, self._attn_length, 1, 1]) * hidden, [1, 2])
      new_attns = array_ops.reshape(d, [-1, self._attn_size])
      new_attn_states = array_ops.slice(attn_states, [0, 1, 0], [-1, -1, -1])
      return new_attns, new_attn_states


class HighwayWrapper(rnn_cell_impl.RNNCell):
  """RNNCell wrapper that adds highway connection on cell input and output.

  Based on:
    R. K. Srivastava, K. Greff, and J. Schmidhuber, "Highway networks",
    arXiv preprint arXiv:1505.00387, 2015.
    https://arxiv.org/abs/1505.00387
  """

  def __init__(self, cell,
               couple_carry_transform_gates=True,
               carry_bias_init=1.0):
    """Constructs a `HighwayWrapper` for `cell`.

    Args:
      cell: An instance of `RNNCell`.
      couple_carry_transform_gates: boolean, should the Carry and Transform gate
        be coupled.
      carry_bias_init: float, carry gates bias initialization.
    """
    self._cell = cell
    self._couple_carry_transform_gates = couple_carry_transform_gates
    self._carry_bias_init = carry_bias_init

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def _highway(self, inp, out):
    input_size = inp.get_shape().with_rank(2)[1].value
    carry_weight = vs.get_variable("carry_w", [input_size, input_size])
    carry_bias = vs.get_variable(
        "carry_b", [input_size],
        initializer=init_ops.constant_initializer(
            self._carry_bias_init))
    carry = math_ops.sigmoid(nn_ops.xw_plus_b(inp, carry_weight, carry_bias))
    if self._couple_carry_transform_gates:
      transform = 1 - carry
    else:
      transform_weight = vs.get_variable("transform_w",
                                         [input_size, input_size])
      transform_bias = vs.get_variable(
          "transform_b", [input_size],
          initializer=init_ops.constant_initializer(
              -self._carry_bias_init))
      transform = math_ops.sigmoid(nn_ops.xw_plus_b(inp,
                                                    transform_weight,
                                                    transform_bias))
    return inp * carry + out * transform

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
    res_outputs = nest.map_structure(self._highway, inputs, outputs)
    return (res_outputs, new_state)


class LayerNormBasicLSTMCell(rnn_cell_impl.RNNCell):
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
               dropout_keep_prob=1.0, dropout_prob_seed=None,
               reuse=None):
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
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(LayerNormBasicLSTMCell, self).__init__(_reuse=reuse)

    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)

    self._num_units = num_units
    self._activation = activation
    self._forget_bias = forget_bias
    self._keep_prob = dropout_keep_prob
    self._seed = dropout_prob_seed
    self._layer_norm = layer_norm
    self._norm_gain = norm_gain
    self._norm_shift = norm_shift
    self._reuse = reuse

  @property
  def state_size(self):
    return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def _norm(self, inp, scope, dtype=dtypes.float32):
    shape = inp.get_shape()[-1:]
    gamma_init = init_ops.constant_initializer(self._norm_gain)
    beta_init = init_ops.constant_initializer(self._norm_shift)
    with vs.variable_scope(scope):
      # Initialize beta and gamma for use by layer_norm.
      vs.get_variable("gamma", shape=shape, initializer=gamma_init, dtype=dtype)
      vs.get_variable("beta", shape=shape, initializer=beta_init, dtype=dtype)
    normalized = layers.layer_norm(inp, reuse=True, scope=scope)
    return normalized

  def _linear(self, args):
    out_size = 4 * self._num_units
    proj_size = args.get_shape()[-1]
    dtype = args.dtype
    weights = vs.get_variable("kernel", [proj_size, out_size], dtype=dtype)
    out = math_ops.matmul(args, weights)
    if not self._layer_norm:
      bias = vs.get_variable("bias", [out_size], dtype=dtype)
      out = nn_ops.bias_add(out, bias)
    return out

  def call(self, inputs, state):
    """LSTM cell with layer normalization and recurrent dropout."""
    c, h = state
    args = array_ops.concat([inputs, h], 1)
    concat = self._linear(args)
    dtype = args.dtype

    i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)
    if self._layer_norm:
      i = self._norm(i, "input", dtype=dtype)
      j = self._norm(j, "transform", dtype=dtype)
      f = self._norm(f, "forget", dtype=dtype)
      o = self._norm(o, "output", dtype=dtype)

    g = self._activation(j)
    if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
      g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)

    new_c = (c * math_ops.sigmoid(f + self._forget_bias)
             + math_ops.sigmoid(i) * g)
    if self._layer_norm:
      new_c = self._norm(new_c, "state", dtype=dtype)
    new_h = self._activation(new_c) * math_ops.sigmoid(o)

    new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
    return new_h, new_state


class NASCell(rnn_cell_impl.RNNCell):
  """Neural Architecture Search (NAS) recurrent network cell.

  This implements the recurrent cell from the paper:

    https://arxiv.org/abs/1611.01578

  Barret Zoph and Quoc V. Le.
  "Neural Architecture Search with Reinforcement Learning" Proc. ICLR 2017.

  The class uses an optional projection layer.
  """

  def __init__(self, num_units, num_proj=None,
               use_biases=False, reuse=None):
    """Initialize the parameters for a NAS cell.

    Args:
      num_units: int, The number of units in the NAS cell
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      use_biases: (optional) bool, If True then use biases within the cell. This
        is False by default.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(NASCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._num_proj = num_proj
    self._use_biases = use_biases
    self._reuse = reuse

    if num_proj is not None:
      self._state_size = rnn_cell_impl.LSTMStateTuple(num_units, num_proj)
      self._output_size = num_proj
    else:
      self._state_size = rnn_cell_impl.LSTMStateTuple(num_units, num_units)
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def call(self, inputs, state):
    """Run one step of NAS Cell.

    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: This must be a tuple of state Tensors, both `2-D`, with column
        sizes `c_state` and `m_state`.

    Returns:
      A tuple containing:
      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        NAS Cell after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of NAS Cell after reading `inputs`
        when the previous state was `state`.  Same type and shape(s) as `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    sigmoid = math_ops.sigmoid
    tanh = math_ops.tanh
    relu = nn_ops.relu

    num_proj = self._num_units if self._num_proj is None else self._num_proj

    (c_prev, m_prev) = state

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
    # Variables for the NAS cell. W_m is all matrices multiplying the
    # hiddenstate and W_inputs is all matrices multiplying the inputs.
    concat_w_m = vs.get_variable(
        "recurrent_kernel", [num_proj, 8 * self._num_units],
        dtype)
    concat_w_inputs = vs.get_variable(
        "kernel", [input_size.value, 8 * self._num_units],
        dtype)

    m_matrix = math_ops.matmul(m_prev, concat_w_m)
    inputs_matrix = math_ops.matmul(inputs, concat_w_inputs)

    if self._use_biases:
      b = vs.get_variable(
          "bias",
          shape=[8 * self._num_units],
          initializer=init_ops.zeros_initializer(),
          dtype=dtype)
      m_matrix = nn_ops.bias_add(m_matrix, b)

    # The NAS cell branches into 8 different splits for both the hiddenstate
    # and the input
    m_matrix_splits = array_ops.split(axis=1, num_or_size_splits=8,
                                      value=m_matrix)
    inputs_matrix_splits = array_ops.split(axis=1, num_or_size_splits=8,
                                           value=inputs_matrix)

    # First layer
    layer1_0 = sigmoid(inputs_matrix_splits[0] + m_matrix_splits[0])
    layer1_1 = relu(inputs_matrix_splits[1] + m_matrix_splits[1])
    layer1_2 = sigmoid(inputs_matrix_splits[2] + m_matrix_splits[2])
    layer1_3 = relu(inputs_matrix_splits[3] * m_matrix_splits[3])
    layer1_4 = tanh(inputs_matrix_splits[4] + m_matrix_splits[4])
    layer1_5 = sigmoid(inputs_matrix_splits[5] + m_matrix_splits[5])
    layer1_6 = tanh(inputs_matrix_splits[6] + m_matrix_splits[6])
    layer1_7 = sigmoid(inputs_matrix_splits[7] + m_matrix_splits[7])

    # Second layer
    l2_0 = tanh(layer1_0 * layer1_1)
    l2_1 = tanh(layer1_2 + layer1_3)
    l2_2 = tanh(layer1_4 * layer1_5)
    l2_3 = sigmoid(layer1_6 + layer1_7)

    # Inject the cell
    l2_0 = tanh(l2_0 + c_prev)

    # Third layer
    l3_0_pre = l2_0 * l2_1
    new_c = l3_0_pre  # create new cell
    l3_0 = l3_0_pre
    l3_1 = tanh(l2_2 + l2_3)

    # Final layer
    new_m = tanh(l3_0 * l3_1)

    # Projection layer if specified
    if self._num_proj is not None:
      concat_w_proj = vs.get_variable(
          "projection_weights", [self._num_units, self._num_proj],
          dtype)
      new_m = math_ops.matmul(new_m, concat_w_proj)

    new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_m)
    return new_m, new_state


class UGRNNCell(rnn_cell_impl.RNNCell):
  """Update Gate Recurrent Neural Network (UGRNN) cell.

  Compromise between a LSTM/GRU and a vanilla RNN.  There is only one
  gate, and that is to determine whether the unit should be
  integrating or computing instantaneously.  This is the recurrent
  idea of the feedforward Highway Network.

  This implements the recurrent cell from the paper:

    https://arxiv.org/abs/1611.09913

  Jasmine Collins, Jascha Sohl-Dickstein, and David Sussillo.
  "Capacity and Trainability in Recurrent Neural Networks" Proc. ICLR 2017.
  """

  def __init__(self, num_units, initializer=None, forget_bias=1.0,
               activation=math_ops.tanh, reuse=None):
    """Initialize the parameters for an UGRNN cell.

    Args:
      num_units: int, The number of units in the UGRNN cell
      initializer: (optional) The initializer to use for the weight matrices.
      forget_bias: (optional) float, default 1.0, The initial bias of the
        forget gate, used to reduce the scale of forgetting at the beginning
        of the training.
      activation: (optional) Activation function of the inner states.
        Default is `tf.tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(UGRNNCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._initializer = initializer
    self._forget_bias = forget_bias
    self._activation = activation
    self._reuse = reuse
    self._linear = None

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Run one step of UGRNN.

    Args:
      inputs: input Tensor, 2D, batch x input size.
      state: state Tensor, 2D, batch x num units.

    Returns:
      new_output: batch x num units, Tensor representing the output of the UGRNN
        after reading `inputs` when previous state was `state`. Identical to
        `new_state`.
      new_state: batch x num units, Tensor representing the state of the UGRNN
        after reading `inputs` when previous state was `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    sigmoid = math_ops.sigmoid

    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    with vs.variable_scope(vs.get_variable_scope(),
                           initializer=self._initializer):
      cell_inputs = array_ops.concat([inputs, state], 1)
      if self._linear is None:
        self._linear = _Linear(cell_inputs, 2 * self._num_units, True)
      rnn_matrix = self._linear(cell_inputs)

      [g_act, c_act] = array_ops.split(
          axis=1, num_or_size_splits=2, value=rnn_matrix)

      c = self._activation(c_act)
      g = sigmoid(g_act + self._forget_bias)
      new_state = g * state + (1.0 - g) * c
      new_output = new_state

    return new_output, new_state


class IntersectionRNNCell(rnn_cell_impl.RNNCell):
  """Intersection Recurrent Neural Network (+RNN) cell.

  Architecture with coupled recurrent gate as well as coupled depth
  gate, designed to improve information flow through stacked RNNs. As the
  architecture uses depth gating, the dimensionality of the depth
  output (y) also should not change through depth (input size == output size).
  To achieve this, the first layer of a stacked Intersection RNN projects
  the inputs to N (num units) dimensions. Therefore when initializing an
  IntersectionRNNCell, one should set `num_in_proj = N` for the first layer
  and use default settings for subsequent layers.

  This implements the recurrent cell from the paper:

    https://arxiv.org/abs/1611.09913

  Jasmine Collins, Jascha Sohl-Dickstein, and David Sussillo.
  "Capacity and Trainability in Recurrent Neural Networks" Proc. ICLR 2017.

  The Intersection RNN is built for use in deeply stacked
  RNNs so it may not achieve best performance with depth 1.
  """

  def __init__(self, num_units, num_in_proj=None,
               initializer=None, forget_bias=1.0,
               y_activation=nn_ops.relu, reuse=None):
    """Initialize the parameters for an +RNN cell.

    Args:
      num_units: int, The number of units in the +RNN cell
      num_in_proj: (optional) int, The input dimensionality for the RNN.
        If creating the first layer of an +RNN, this should be set to
        `num_units`. Otherwise, this should be set to `None` (default).
        If `None`, dimensionality of `inputs` should be equal to `num_units`,
        otherwise ValueError is thrown.
      initializer: (optional) The initializer to use for the weight matrices.
      forget_bias: (optional) float, default 1.0, The initial bias of the
        forget gates, used to reduce the scale of forgetting at the beginning
        of the training.
      y_activation: (optional) Activation function of the states passed
        through depth. Default is 'tf.nn.relu`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(IntersectionRNNCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._initializer = initializer
    self._forget_bias = forget_bias
    self._num_input_proj = num_in_proj
    self._y_activation = y_activation
    self._reuse = reuse
    self._linear1 = None
    self._linear2 = None

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Run one step of the Intersection RNN.

    Args:
      inputs: input Tensor, 2D, batch x input size.
      state: state Tensor, 2D, batch x num units.

    Returns:
      new_y: batch x num units, Tensor representing the output of the +RNN
        after reading `inputs` when previous state was `state`.
      new_state: batch x num units, Tensor representing the state of the +RNN
        after reading `inputs` when previous state was `state`.

    Raises:
      ValueError: If input size cannot be inferred from `inputs` via
        static shape inference.
      ValueError: If input size != output size (these must be equal when
        using the Intersection RNN).
    """
    sigmoid = math_ops.sigmoid
    tanh = math_ops.tanh

    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    with vs.variable_scope(vs.get_variable_scope(),
                           initializer=self._initializer):
      # read-in projections (should be used for first layer in deep +RNN
      # to transform size of inputs from I --> N)
      if input_size.value != self._num_units:
        if self._num_input_proj:
          with vs.variable_scope("in_projection"):
            if self._linear1 is None:
              self._linear1 = _Linear(inputs, self._num_units, True)
            inputs = self._linear1(inputs)
        else:
          raise ValueError("Must have input size == output size for "
                           "Intersection RNN. To fix, num_in_proj should "
                           "be set to num_units at cell init.")

      n_dim = i_dim = self._num_units
      cell_inputs = array_ops.concat([inputs, state], 1)
      if self._linear2 is None:
        self._linear2 = _Linear(cell_inputs, 2*n_dim + 2*i_dim, True)
      rnn_matrix = self._linear2(cell_inputs)

      gh_act = rnn_matrix[:, :n_dim]                           # b x n
      h_act = rnn_matrix[:, n_dim:2*n_dim]                     # b x n
      gy_act = rnn_matrix[:, 2*n_dim:2*n_dim+i_dim]            # b x i
      y_act = rnn_matrix[:, 2*n_dim+i_dim:2*n_dim+2*i_dim]     # b x i

      h = tanh(h_act)
      y = self._y_activation(y_act)
      gh = sigmoid(gh_act + self._forget_bias)
      gy = sigmoid(gy_act + self._forget_bias)

      new_state = gh * state + (1.0 - gh) * h  # passed thru time
      new_y = gy * inputs + (1.0 - gy) * y  # passed thru depth

    return new_y, new_state


_REGISTERED_OPS = None


class CompiledWrapper(rnn_cell_impl.RNNCell):
  """Wraps step execution in an XLA JIT scope."""

  def __init__(self, cell, compile_stateful=False):
    """Create CompiledWrapper cell.

    Args:
      cell: Instance of `RNNCell`.
      compile_stateful: Whether to compile stateful ops like initializers
        and random number generators (default: False).
    """
    self._cell = cell
    self._compile_stateful = compile_stateful

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def __call__(self, inputs, state, scope=None):
    if self._compile_stateful:
      compile_ops = True
    else:
      def compile_ops(node_def):
        global _REGISTERED_OPS
        if _REGISTERED_OPS is None:
          _REGISTERED_OPS = op_def_registry.get_registered_ops()
        return not _REGISTERED_OPS[node_def.op].is_stateful

    with jit.experimental_jit_scope(compile_ops=compile_ops):
      return self._cell(inputs, state, scope)


def _random_exp_initializer(minval,
                            maxval,
                            seed=None,
                            dtype=dtypes.float32):
  """Returns an exponential distribution initializer.

  Args:
    minval: float or a scalar float Tensor. With value > 0. Lower bound of the
        range of random values to generate.
    maxval: float or a scalar float Tensor. With value > minval. Upper bound of
        the range of random values to generate.
    seed: An integer. Used to create random seeds.
    dtype: The data type.

  Returns:
    An initializer that generates tensors with an exponential distribution.
  """

  def _initializer(shape, dtype=dtype, partition_info=None):
    del partition_info  # Unused.
    return math_ops.exp(
        random_ops.random_uniform(
            shape,
            math_ops.log(minval),
            math_ops.log(maxval),
            dtype,
            seed=seed))

  return _initializer


class PhasedLSTMCell(rnn_cell_impl.RNNCell):
  """Phased LSTM recurrent network cell.

  https://arxiv.org/pdf/1610.09513v1.pdf
  """

  def __init__(self,
               num_units,
               use_peepholes=False,
               leak=0.001,
               ratio_on=0.1,
               trainable_ratio_on=True,
               period_init_min=1.0,
               period_init_max=1000.0,
               reuse=None):
    """Initialize the Phased LSTM cell.

    Args:
      num_units: int, The number of units in the Phased LSTM cell.
      use_peepholes: bool, set True to enable peephole connections.
      leak: float or scalar float Tensor with value in [0, 1]. Leak applied
          during training.
      ratio_on: float or scalar float Tensor with value in [0, 1]. Ratio of the
          period during which the gates are open.
      trainable_ratio_on: bool, weather ratio_on is trainable.
      period_init_min: float or scalar float Tensor. With value > 0.
          Minimum value of the initialized period.
          The period values are initialized by drawing from the distribution:
          e^U(log(period_init_min), log(period_init_max))
          Where U(.,.) is the uniform distribution.
      period_init_max: float or scalar float Tensor.
          With value > period_init_min. Maximum value of the initialized period.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope. If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(PhasedLSTMCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._leak = leak
    self._ratio_on = ratio_on
    self._trainable_ratio_on = trainable_ratio_on
    self._period_init_min = period_init_min
    self._period_init_max = period_init_max
    self._reuse = reuse
    self._linear1 = None
    self._linear2 = None
    self._linear3 = None

  @property
  def state_size(self):
    return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def _mod(self, x, y):
    """Modulo function that propagates x gradients."""
    return array_ops.stop_gradient(math_ops.mod(x, y) - x) + x

  def _get_cycle_ratio(self, time, phase, period):
    """Compute the cycle ratio in the dtype of the time."""
    phase_casted = math_ops.cast(phase, dtype=time.dtype)
    period_casted = math_ops.cast(period, dtype=time.dtype)
    shifted_time = time - phase_casted
    cycle_ratio = self._mod(shifted_time, period_casted) / period_casted
    return math_ops.cast(cycle_ratio, dtype=dtypes.float32)

  def call(self, inputs, state):
    """Phased LSTM Cell.

    Args:
      inputs: A tuple of 2 Tensor.
         The first Tensor has shape [batch, 1], and type float32 or float64.
         It stores the time.
         The second Tensor has shape [batch, features_size], and type float32.
         It stores the features.
      state: rnn_cell_impl.LSTMStateTuple, state from previous timestep.

    Returns:
      A tuple containing:
      - A Tensor of float32, and shape [batch_size, num_units], representing the
        output of the cell.
      - A rnn_cell_impl.LSTMStateTuple, containing 2 Tensors of float32, shape
        [batch_size, num_units], representing the new state and the output.
    """
    (c_prev, h_prev) = state
    (time, x) = inputs

    in_mask_gates = [x, h_prev]
    if self._use_peepholes:
      in_mask_gates.append(c_prev)

    with vs.variable_scope("mask_gates"):
      if self._linear1 is None:
        self._linear1 = _Linear(in_mask_gates, 2 * self._num_units, True)

      mask_gates = math_ops.sigmoid(
          self._linear1(in_mask_gates))
      [input_gate, forget_gate] = array_ops.split(
          axis=1, num_or_size_splits=2, value=mask_gates)

    with vs.variable_scope("new_input"):
      if self._linear2 is None:
        self._linear2 = _Linear([x, h_prev], self._num_units, True)
      new_input = math_ops.tanh(self._linear2([x, h_prev]))

    new_c = (c_prev * forget_gate + input_gate * new_input)

    in_out_gate = [x, h_prev]
    if self._use_peepholes:
      in_out_gate.append(new_c)

    with vs.variable_scope("output_gate"):
      if self._linear3 is None:
        self._linear3 = _Linear(in_out_gate, self._num_units, True)
      output_gate = math_ops.sigmoid(self._linear3(in_out_gate))

    new_h = math_ops.tanh(new_c) * output_gate

    period = vs.get_variable(
        "period", [self._num_units],
        initializer=_random_exp_initializer(
            self._period_init_min, self._period_init_max))
    phase = vs.get_variable(
        "phase", [self._num_units],
        initializer=init_ops.random_uniform_initializer(
            0., period.initial_value))
    ratio_on = vs.get_variable(
        "ratio_on", [self._num_units],
        initializer=init_ops.constant_initializer(self._ratio_on),
        trainable=self._trainable_ratio_on)

    cycle_ratio = self._get_cycle_ratio(time, phase, period)

    k_up = 2 * cycle_ratio / ratio_on
    k_down = 2 - k_up
    k_closed = self._leak * cycle_ratio

    k = array_ops.where(cycle_ratio < ratio_on, k_down, k_closed)
    k = array_ops.where(cycle_ratio < 0.5 * ratio_on, k_up, k)

    new_c = k * new_c + (1 - k) * c_prev
    new_h = k * new_h + (1 - k) * h_prev

    new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)

    return new_h, new_state

class ConvLSTMCell(rnn_cell_impl.RNNCell):
  """Convolutional LSTM recurrent network cell.

  https://arxiv.org/pdf/1506.04214v1.pdf
  """

  def __init__(self,
               conv_ndims,
               input_shape,
               output_channels,
               kernel_shape,
               use_bias=True,
               skip_connection=False,
               forget_bias=1.0,
               initializers=None,
               name="conv_lstm_cell"):
    """Construct ConvLSTMCell.
    Args:
      conv_ndims: Convolution dimensionality (1, 2 or 3).
      input_shape: Shape of the input as int tuple, excluding the batch size.
      output_channels: int, number of output channels of the conv LSTM.
      kernel_shape: Shape of kernel as in tuple (of size 1,2 or 3).
      use_bias: Use bias in convolutions.
      skip_connection: If set to `True`, concatenate the input to the
      output of the conv LSTM. Default: `False`.
      forget_bias: Forget bias.
      name: Name of the module.
    Raises:
      ValueError: If `skip_connection` is `True` and stride is different from 1
        or if `input_shape` is incompatible with `conv_ndims`.
    """
    super(ConvLSTMCell, self).__init__(name=name)

    if conv_ndims != len(input_shape)-1:
      raise ValueError("Invalid input_shape {} for conv_ndims={}.".format(
          input_shape, conv_ndims))

    self._conv_ndims = conv_ndims
    self._input_shape = input_shape
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._use_bias = use_bias
    self._forget_bias = forget_bias
    self._skip_connection = skip_connection

    self._total_output_channels = output_channels
    if self._skip_connection:
      self._total_output_channels += self._input_shape[-1]

    state_size = tensor_shape.TensorShape(
        self._input_shape[:-1] + [self._output_channels])
    self._state_size = rnn_cell_impl.LSTMStateTuple(state_size, state_size)
    self._output_size = tensor_shape.TensorShape(self._input_shape[:-1]
                                                 + [self._total_output_channels])

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def call(self, inputs, state, scope=None):
    cell, hidden = state
    new_hidden = _conv([inputs, hidden],
                       self._kernel_shape,
                       4*self._output_channels,
                       self._use_bias)
    gates = array_ops.split(value=new_hidden,
                            num_or_size_splits=4,
                            axis=self._conv_ndims+1)

    input_gate, new_input, forget_gate, output_gate = gates
    new_cell = math_ops.sigmoid(forget_gate + self._forget_bias) * cell
    new_cell += math_ops.sigmoid(input_gate) * math_ops.tanh(new_input)
    output = math_ops.tanh(new_cell) * math_ops.sigmoid(output_gate)

    if self._skip_connection:
      output = array_ops.concat([output, inputs], axis=-1)
    new_state = rnn_cell_impl.LSTMStateTuple(new_cell, output)
    return output, new_state

class Conv1DLSTMCell(ConvLSTMCell):
  """1D Convolutional LSTM recurrent network cell.

  https://arxiv.org/pdf/1506.04214v1.pdf
  """
  def __init__(self, name="conv_1d_lstm_cell", **kwargs):
    """Construct Conv1DLSTM. See `ConvLSTMCell` for more details."""
    super(Conv1DLSTMCell, self).__init__(conv_ndims=1, **kwargs)

class Conv2DLSTMCell(ConvLSTMCell):
  """2D Convolutional LSTM recurrent network cell.

  https://arxiv.org/pdf/1506.04214v1.pdf
  """
  def __init__(self, name="conv_2d_lstm_cell", **kwargs):
    """Construct Conv2DLSTM. See `ConvLSTMCell` for more details."""
    super(Conv2DLSTMCell, self).__init__(conv_ndims=2, **kwargs)

class Conv3DLSTMCell(ConvLSTMCell):
  """3D Convolutional LSTM recurrent network cell.

  https://arxiv.org/pdf/1506.04214v1.pdf
  """
  def __init__(self, name="conv_3d_lstm_cell", **kwargs):
    """Construct Conv3DLSTM. See `ConvLSTMCell` for more details."""
    super(Conv3DLSTMCell, self).__init__(conv_ndims=3, **kwargs)


def _conv(args, filter_size, num_features, bias, bias_start=0.0):
  """convolution:
  Args:
    args: a Tensor or a list of Tensors of dimension 3D, 4D or 5D, 
    batch x n, Tensors.
    filter_size: int tuple of filter height and width.
    num_features: int, number of features.
    bias_start: starting value to initialize the bias; 0 by default.
  Returns:
    A 3D, 4D, or 5D Tensor with shape [batch ... num_features]
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

  # Calculate the total size of arguments on dimension 1.
  total_arg_size_depth = 0
  shapes = [a.get_shape().as_list() for a in args]
  shape_length = len(shapes[0])
  for shape in shapes:
    if len(shape) not in [3,4,5]:
      raise ValueError("Conv Linear expects 3D, 4D "
                       "or 5D arguments: %s" % str(shapes))
    if len(shape) != len(shapes[0]):
      raise ValueError("Conv Linear expects all args "
                       "to be of same Dimension: %s" % str(shapes))
    else:
      total_arg_size_depth += shape[-1]
  dtype = [a.dtype for a in args][0]

  # determine correct conv operation
  if   shape_length == 3:
    conv_op = nn_ops.conv1d
    strides = 1
  elif shape_length == 4:
    conv_op = nn_ops.conv2d
    strides = shape_length*[1]
  elif shape_length == 5:
    conv_op = nn_ops.conv3d
    strides = shape_length*[1]

  # Now the computation.
  kernel = vs.get_variable(
      "kernel",
      filter_size + [total_arg_size_depth, num_features],
      dtype=dtype)
  if len(args) == 1:
    res = conv_op(args[0],
                  kernel,
                  strides,
                  padding='SAME')
  else:
    res = conv_op(array_ops.concat(axis=shape_length-1, values=args),
                  kernel,
                  strides,
                  padding='SAME')
  if not bias:
    return res
  bias_term = vs.get_variable(
      "biases", [num_features],
      dtype=dtype,
      initializer=init_ops.constant_initializer(
          bias_start, dtype=dtype))
  return res + bias_term

class GLSTMCell(rnn_cell_impl.RNNCell):
  """Group LSTM cell (G-LSTM).

  The implementation is based on:

    https://arxiv.org/abs/1703.10722

  O. Kuchaiev and B. Ginsburg
  "Factorization Tricks for LSTM Networks", ICLR 2017 workshop.
  """

  def __init__(self, num_units, initializer=None, num_proj=None,
               number_of_groups=1, forget_bias=1.0, activation=math_ops.tanh,
               reuse=None):
    """Initialize the parameters of G-LSTM cell.

    Args:
      num_units: int, The number of units in the G-LSTM cell
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      number_of_groups: (optional) int, number of groups to use.
        If `number_of_groups` is 1, then it should be equivalent to LSTM cell
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training.
      activation: Activation function of the inner states.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already
        has the given variables, an error is raised.

    Raises:
      ValueError: If `num_units` or `num_proj` is not divisible by 
        `number_of_groups`.
    """
    super(GLSTMCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._initializer = initializer
    self._num_proj = num_proj
    self._forget_bias = forget_bias
    self._activation = activation
    self._number_of_groups = number_of_groups

    if self._num_units % self._number_of_groups != 0:
      raise ValueError("num_units must be divisible by number_of_groups")
    if self._num_proj:
      if self._num_proj % self._number_of_groups != 0:
        raise ValueError("num_proj must be divisible by number_of_groups")
      self._group_shape = [int(self._num_proj / self._number_of_groups),
                           int(self._num_units / self._number_of_groups)]
    else:
      self._group_shape = [int(self._num_units / self._number_of_groups),
                           int(self._num_units / self._number_of_groups)]

    if num_proj:
      self._state_size = rnn_cell_impl.LSTMStateTuple(num_units, num_proj)
      self._output_size = num_proj
    else:
      self._state_size = rnn_cell_impl.LSTMStateTuple(num_units, num_units)
      self._output_size = num_units
    self._linear1 = None
    self._linear2 = None

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def _get_input_for_group(self, inputs, group_id, group_size):
    """Slices inputs into groups to prepare for processing by cell's groups

    Args:
      inputs: cell input or it's previous state,
              a Tensor, 2D, [batch x num_units]
      group_id: group id, a Scalar, for which to prepare input
      group_size: size of the group

    Returns:
      subset of inputs corresponding to group "group_id",
      a Tensor, 2D, [batch x num_units/number_of_groups]
    """
    return array_ops.slice(input_=inputs,
                           begin=[0, group_id * group_size],
                           size=[self._batch_size, group_size],
                           name=("GLSTM_group%d_input_generation" % group_id))

  def call(self, inputs, state):
    """Run one step of G-LSTM.

    Args:
      inputs: input Tensor, 2D, [batch x num_units].
      state: this must be a tuple of state Tensors, both `2-D`,
      with column sizes `c_state` and `m_state`.

    Returns:
      A tuple containing:

      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        G-LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - LSTMStateTuple representing the new state of G-LSTM cell
        after reading `inputs` when the previous state was `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    (c_prev, m_prev) = state

    self._batch_size = inputs.shape[0].value or array_ops.shape(inputs)[0]
    dtype = inputs.dtype
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope, initializer=self._initializer):
      i_parts = []
      j_parts = []
      f_parts = []
      o_parts = []

      for group_id in range(self._number_of_groups):
        with vs.variable_scope("group%d" % group_id):
          x_g_id = array_ops.concat(
            [self._get_input_for_group(inputs, group_id,
                                       self._group_shape[0]),
             self._get_input_for_group(m_prev, group_id,
                                       self._group_shape[0])], axis=1)
          if self._linear1 is None:
            self._linear1 = _Linear(x_g_id, 4 * self._group_shape[1], False)
          R_k = self._linear1(x_g_id)  # pylint: disable=invalid-name
          i_k, j_k, f_k, o_k = array_ops.split(R_k, 4, 1)

        i_parts.append(i_k)
        j_parts.append(j_k)
        f_parts.append(f_k)
        o_parts.append(o_k)

      bi = vs.get_variable(name="bias_i",
                           shape=[self._num_units],
                           dtype=dtype,
                           initializer=
                           init_ops.constant_initializer(0.0, dtype=dtype))
      bj = vs.get_variable(name="bias_j",
                           shape=[self._num_units],
                           dtype=dtype,
                           initializer=
                           init_ops.constant_initializer(0.0, dtype=dtype))
      bf = vs.get_variable(name="bias_f",
                           shape=[self._num_units],
                           dtype=dtype,
                           initializer=
                           init_ops.constant_initializer(0.0, dtype=dtype))
      bo = vs.get_variable(name="bias_o",
                           shape=[self._num_units],
                           dtype=dtype,
                           initializer=
                           init_ops.constant_initializer(0.0, dtype=dtype))

      i = nn_ops.bias_add(array_ops.concat(i_parts, axis=1), bi)
      j = nn_ops.bias_add(array_ops.concat(j_parts, axis=1), bj)
      f = nn_ops.bias_add(array_ops.concat(f_parts, axis=1), bf)
      o = nn_ops.bias_add(array_ops.concat(o_parts, axis=1), bo)

    c = (math_ops.sigmoid(f + self._forget_bias) * c_prev +
         math_ops.sigmoid(i) * math_ops.tanh(j))
    m = math_ops.sigmoid(o) * self._activation(c)

    if self._num_proj is not None:
      with vs.variable_scope("projection"):
        if self._linear2 is None:
          self._linear2 = _Linear(m, self._num_proj, False)
        m = self._linear2(m)

    new_state = rnn_cell_impl.LSTMStateTuple(c, m)
    return m, new_state


class LayerNormLSTMCell(rnn_cell_impl.RNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.

  The default non-peephole implementation is based on:

    http://www.bioinf.jku.at/publications/older/2604.pdf

  S. Hochreiter and J. Schmidhuber.
  "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

  The peephole implementation is based on:

    https://research.google.com/pubs/archive/43905.pdf

  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.

  The class uses optional peep-hole connections, optional cell clipping, and
  an optional projection layer.

  Layer normalization implementation is based on:

    https://arxiv.org/abs/1607.06450.

  "Layer Normalization"
  Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

  and is applied before the internal nonlinearities.

  """

  def __init__(self,
               num_units,
               use_peepholes=False,
               cell_clip=None,
               initializer=None,
               num_proj=None,
               proj_clip=None,
               forget_bias=1.0,
               activation=None,
               layer_norm=False,
               norm_gain=1.0,
               norm_shift=0.0,
               reuse=None):
    """Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell
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
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training. Must set it manually to `0.0` when restoring from
        CudnnLSTM trained checkpoints.
      activation: Activation function of the inner states.  Default: `tanh`.
      layer_norm: If `True`, layer normalization will be applied.
      norm_gain: float, The layer normalization gain initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      norm_shift: float, The layer normalization shift initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.

      When restoring from CudnnLSTM-trained checkpoints, must use
      CudnnCompatibleLSTMCell instead.
    """
    super(LayerNormLSTMCell, self).__init__(_reuse=reuse)

    self._num_units = num_units
    self._use_peepholes = use_peepholes
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._forget_bias = forget_bias
    self._activation = activation or math_ops.tanh
    self._layer_norm = layer_norm
    self._norm_gain = norm_gain
    self._norm_shift = norm_shift

    if num_proj:
      self._state_size = (rnn_cell_impl.LSTMStateTuple(num_units, num_proj))
      self._output_size = num_proj
    else:
      self._state_size = (rnn_cell_impl.LSTMStateTuple(num_units, num_units))
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def _linear(self,
              args,
              output_size,
              bias,
              bias_initializer=None,
              kernel_initializer=None,
              layer_norm=False):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a Variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_initializer: starting value to initialize the bias
        (default is all zeros).
      kernel_initializer: starting value to initialize the weight.
      layer_norm: boolean, whether to apply layer normalization.


    Returns:
      A 2D Tensor with shape [batch x output_size] taking value
      sum_i(args[i] * W[i]), where each W[i] is a newly created Variable.

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
          "kernel", [total_arg_size, output_size],
          dtype=dtype,
          initializer=kernel_initializer)
      if len(args) == 1:
        res = math_ops.matmul(args[0], weights)
      else:
        res = math_ops.matmul(array_ops.concat(args, 1), weights)
      if not bias:
        return res
      with vs.variable_scope(outer_scope) as inner_scope:
        inner_scope.set_partitioner(None)
        if bias_initializer is None:
          bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
        biases = vs.get_variable(
            "bias", [output_size], dtype=dtype, initializer=bias_initializer)

    if not layer_norm:
      res = nn_ops.bias_add(res, biases)

    return res

  def call(self, inputs, state):
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: this must be a tuple of state Tensors,
       both `2-D`, with column sizes `c_state` and
        `m_state`.

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

    (c_prev, m_prev) = state

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope, initializer=self._initializer) as unit_scope:

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      lstm_matrix = self._linear(
          [inputs, m_prev],
          4 * self._num_units,
          bias=True,
          bias_initializer=None,
          layer_norm=self._layer_norm)
      i, j, f, o = array_ops.split(
          value=lstm_matrix, num_or_size_splits=4, axis=1)

      if self._layer_norm:
        i = _norm(self._norm_gain, self._norm_shift, i, "input")
        j = _norm(self._norm_gain, self._norm_shift, j, "transform")
        f = _norm(self._norm_gain, self._norm_shift, f, "forget")
        o = _norm(self._norm_gain, self._norm_shift, o, "output")

      # Diagonal connections
      if self._use_peepholes:
        with vs.variable_scope(unit_scope):
          w_f_diag = vs.get_variable(
              "w_f_diag", shape=[self._num_units], dtype=dtype)
          w_i_diag = vs.get_variable(
              "w_i_diag", shape=[self._num_units], dtype=dtype)
          w_o_diag = vs.get_variable(
              "w_o_diag", shape=[self._num_units], dtype=dtype)

      if self._use_peepholes:
        c = (
            sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev +
            sigmoid(i + w_i_diag * c_prev) * self._activation(j))
      else:
        c = (
            sigmoid(f + self._forget_bias) * c_prev +
            sigmoid(i) * self._activation(j))

      if self._layer_norm:
        c = _norm(self._norm_gain, self._norm_shift, c, "state")

      if self._cell_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
        # pylint: enable=invalid-unary-operand-type
      if self._use_peepholes:
        m = sigmoid(o + w_o_diag * c) * self._activation(c)
      else:
        m = sigmoid(o) * self._activation(c)

      if self._num_proj is not None:
        with vs.variable_scope("projection"):
          m = self._linear(m, self._num_proj, bias=False)

        if self._proj_clip is not None:
          # pylint: disable=invalid-unary-operand-type
          m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
          # pylint: enable=invalid-unary-operand-type

    new_state = (rnn_cell_impl.LSTMStateTuple(c, m))
    return m, new_state
