# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""TfLite LSTMCell wrapper.

TODO(renjieliu): Find a better home for this one.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from tensorflow.lite.python import lite
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.platform import tf_logging as logging


class TFLiteLSTMCell(rnn_cell_impl.LayerRNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.

  This is used only for TfLite, it provides hints and it also makes the
  variables in the desired for the tflite ops  (transposed and seaparated).

  The default non-peephole implementation is based on:

    https://pdfs.semanticscholar.org/1154/0131eae85b2e11d53df7f1360eeb6476e7f4.pdf

  Felix Gers, Jurgen Schmidhuber, and Fred Cummins.
  "Learning to forget: Continual prediction with LSTM." IET, 850-855, 1999.

  The peephole implementation is based on:

    https://research.google.com/pubs/archive/43905.pdf

  Hasim Sak, Andrew Senior, and Francoise Beaufays.
  "Long short-term memory recurrent neural network architectures for
   large scale acoustic modeling." INTERSPEECH, 2014.

  The class uses optional peep-hole connections, optional cell clipping, and
  an optional projection layer.

  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnLSTM` for better performance on GPU, or
  `tf.contrib.rnn.LSTMBlockCell` and `tf.contrib.rnn.LSTMBlockFusedCell` for
  better performance on CPU.
  """

  def __init__(self,
               num_units,
               use_peepholes=False,
               cell_clip=None,
               initializer=None,
               num_proj=None,
               proj_clip=None,
               num_unit_shards=None,
               num_proj_shards=None,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               reuse=None,
               name=None,
               dtype=None):
    """Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
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
      num_unit_shards: Deprecated, will be removed by Jan. 2017. Use a
        variable_scope partitioner instead.
      num_proj_shards: Deprecated, will be removed by Jan. 2017. Use a
        variable_scope partitioner instead.
      forget_bias: Biases of the forget gate are initialized by default to 1 in
        order to reduce the scale of forgetting at the beginning of the
        training. Must set it manually to `0.0` when restoring from CudnnLSTM
        trained checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of the
        `c_state` and `m_state`.  If False, they are concatenated along the
        column axis.  This latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables in
        an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will share
        weights, but to avoid mistakes we require reuse=True in such cases.
      dtype: Default dtype of the layer (default of `None` means use the type of
        the first input). Required when `build` is called before `call`.  When
        restoring from CudnnLSTM-trained checkpoints, use
        `CudnnCompatibleLSTMCell` instead.
    """
    super(TFLiteLSTMCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)
    # TODO(raziel): decide if we want to just support tuples (yes please!).
    if not state_is_tuple:
      logging.warn(
          "%s: Using a concatenated state is slower and will soon be "
          "deprecated.  Use state_is_tuple=True.", self)
    if num_unit_shards is not None or num_proj_shards is not None:
      logging.warn(
          "%s: The num_unit_shards and proj_unit_shards parameters are "
          "deprecated and will be removed in Jan 2017.  "
          "Use a variable scope with a partitioner instead.", self)

    # Inputs must be 2-dimensional.
    # TODO(raziel): layers stuff -- chop if un-layerizing Op.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._tflite_wrapper = lite.OpHint("UnidirectionalSequenceLstm")

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
    self._activation = activation or math_ops.tanh

    self._output_size = num_proj if num_proj else num_units
    self._state_size = (
        tf.nn.rnn_cell.LSTMStateTuple(num_units, self._output_size)
        if state_is_tuple else num_units + self._output_size)

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def build(self, inputs_shape):
    """Build TfLite LSTM cell graph.

    Args:
      inputs_shape: The inputs_shape must be known, and is [batch_size,
        input_size] shape.

    Raises:
      ValueError: if the inputs_shape is invalid.
    """
    if len(inputs_shape) != 2 or inputs_shape[1].value is None:
      raise ValueError("Invalid inputs_shape, saw shape: %s" % inputs_shape)

    input_depth = inputs_shape[1].value
    maybe_partitioner = (
        partitioned_variables.fixed_size_partitioner(self._num_unit_shards)
        if self._num_unit_shards is not None else None)
    input_weight_shape = [self._num_units, input_depth]
    cell_weight_shape = [self._num_units, self._output_size]
    bias_shape = [self._num_units]

    def add_variable_wrapped(name, shape, initializer, index, partitioner):
      var = self.add_variable(
          name, shape=shape, initializer=initializer, partitioner=partitioner)
      return self._tflite_wrapper.add_input(
          var, name=name, index_override=index)

    weight_initializer = self._initializer
    if self.dtype is None:
      bias_initializer = init_ops.zeros_initializer
    else:
      bias_initializer = init_ops.zeros_initializer(dtype=self.dtype)

    self.input_to_input_w = add_variable_wrapped(
        "input_to_input_w", input_weight_shape, weight_initializer, 1,
        maybe_partitioner)
    self.input_to_forget_w = add_variable_wrapped(
        "input_to_forget_w", input_weight_shape, weight_initializer, 2,
        maybe_partitioner)
    self.input_to_cell_w = add_variable_wrapped(
        "input_to_cell_w", input_weight_shape, weight_initializer, 3,
        maybe_partitioner)
    self.input_to_output_w = add_variable_wrapped(
        "input_to_output_w", input_weight_shape, weight_initializer, 4,
        maybe_partitioner)
    self.cell_to_input_w = add_variable_wrapped(
        "cell_to_input_w", cell_weight_shape, weight_initializer, 5,
        maybe_partitioner)
    self.cell_to_forget_w = add_variable_wrapped(
        "cell_to_forget_w", cell_weight_shape, weight_initializer, 6,
        maybe_partitioner)
    self.cell_to_cell_w = add_variable_wrapped(
        "cell_to_cell_w", cell_weight_shape, weight_initializer, 7,
        maybe_partitioner)
    self.cell_to_output_w = add_variable_wrapped(
        "cell_to_output_w", cell_weight_shape, weight_initializer, 8,
        maybe_partitioner)

    self.input_bias = add_variable_wrapped(
        "input_bias", bias_shape, bias_initializer, 12, maybe_partitioner)
    self.forget_bias = add_variable_wrapped(
        "forget_bias", bias_shape, bias_initializer, 13, maybe_partitioner)
    self.cell_bias = add_variable_wrapped(
        "cell_bias", bias_shape, bias_initializer, 14, maybe_partitioner)
    self.output_bias = add_variable_wrapped(
        "output_bias", bias_shape, bias_initializer, 15, maybe_partitioner)

    # index 9, 10, 11.
    # f stands for forget, i stands for input and o stands for output.
    if self._use_peepholes:
      self._w_f_diag = add_variable_wrapped("w_f_diag", [self._num_units],
                                            self._initializer, 10,
                                            maybe_partitioner)
      self._w_i_diag = add_variable_wrapped("w_i_diag", [self._num_units],
                                            self._initializer, 9,
                                            maybe_partitioner)
      self._w_o_diag = add_variable_wrapped("w_o_diag", [self._num_units],
                                            self._initializer, 11,
                                            maybe_partitioner)

    # index 16 for proj kernel.
    if self._num_proj is not None:
      maybe_proj_partitioner = (
          partitioned_variables.fixed_size_partitioner(self._num_proj_shards)
          if self._num_proj_shards is not None else None)
      self._proj_kernel = add_variable_wrapped(
          "projection/kernel", [self._num_proj, self._num_units],
          self._initializer,
          16,
          partitioner=maybe_proj_partitioner)

    self.built = True

  def call(self, inputs, state):
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, 2D, `[batch, num_units]`.
      state: if `state_is_tuple` is False, this must be a state Tensor, `2-D,
        [batch, state_size]`.  If `state_is_tuple` is True, this must be a tuple
        of state Tensors, both `2-D`, with column sizes `c_state` and `m_state`.

    Returns:
      A tuple containing:

      - A `2-D, [batch, output_dim]`, Tensor representing the output of the
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
    inputs = self._tflite_wrapper.add_input(
        inputs, tag="input", name="input", aggregate="stack", index_override=0)

    # Make sure inputs and bias_initializer has the same type.
    assert inputs.dtype == self.input_to_input_w.dtype

    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = math_ops.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
      m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

    # Note: For TfLite, cell_state is at index 19 while activation state at
    # index 18.
    c_prev = self._tflite_wrapper.add_input(
        c_prev,
        tag="c_prev",
        name="c_prev",
        aggregate="first",
        index_override=19)
    m_prev = self._tflite_wrapper.add_input(
        m_prev,
        tag="m_prev",
        name="m_prev",
        aggregate="first",
        index_override=18)

    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    inputs_and_m_prev = array_ops.concat([inputs, m_prev], axis=1)

    # i stands for input gate.
    # f stands for forget gate activation.
    # o outputs.
    # j output of LSTM unit.
    # c is the final state.
    # m is the output.
    i = nn_ops.bias_add(
        tf.matmul(
            inputs_and_m_prev,
            tf.concat([self.input_to_input_w, self.cell_to_input_w], axis=1),
            transpose_b=True), self.input_bias)
    f = nn_ops.bias_add(
        tf.matmul(
            inputs_and_m_prev,
            tf.concat([self.input_to_forget_w, self.cell_to_forget_w], axis=1),
            transpose_b=True), self.forget_bias)
    o = nn_ops.bias_add(
        tf.matmul(
            inputs_and_m_prev,
            tf.concat([self.input_to_output_w, self.cell_to_output_w], axis=1),
            transpose_b=True), self.output_bias)
    j = nn_ops.bias_add(
        tf.matmul(
            inputs_and_m_prev,
            tf.concat([self.input_to_cell_w, self.cell_to_cell_w], axis=1),
            transpose_b=True), self.cell_bias)

    # Diagonal connections
    if self._use_peepholes:
      c = (
          sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
          sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
    else:
      c = (
          sigmoid(f + self._forget_bias) * c_prev +
          sigmoid(i) * self._activation(j))

    if self._cell_clip is not None:
      # pylint: disable=invalid-unary-operand-type
      c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
      # pylint: enable=invalid-unary-operand-type
    if self._use_peepholes:
      m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
    else:
      m = sigmoid(o) * self._activation(c)

    if self._num_proj is not None:
      transposed_proj_kernel = tf.transpose(self._proj_kernel)
      m = math_ops.matmul(m, transposed_proj_kernel)

      if self._proj_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
        # pylint: enable=invalid-unary-operand-type

    c = self._tflite_wrapper.add_output(
        c, tag="c", name="c", aggregate="last", index_override=1)
    m = self._tflite_wrapper.add_output(
        m, tag="m", name="m", index_override=2, aggregate="stack")

    new_state = (
        tf.nn.rnn_cell.LSTMStateTuple(c, m)
        if self._state_is_tuple else array_ops.concat([c, m], 1))
    return m, new_state

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "use_peepholes": self._use_peepholes,
        "cell_clip": self._cell_clip,
        "initializer": initializers.serialize(self._initializer),
        "num_proj": self._num_proj,
        "proj_clip": self._proj_clip,
        "num_unit_shards": self._num_unit_shards,
        "num_proj_shards": self._num_proj_shards,
        "forget_bias": self._forget_bias,
        "state_is_tuple": self._state_is_tuple,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(TFLiteLSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
