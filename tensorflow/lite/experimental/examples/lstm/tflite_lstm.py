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
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn import _best_effort_input_batch_size
from tensorflow.python.ops.rnn import _dynamic_rnn_loop
from tensorflow.python.ops.rnn import _should_cache
from tensorflow.python.ops.rnn import _transpose_batch_time
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


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


def dynamic_rnn(cell,
                inputs,
                sequence_length=None,
                initial_state=None,
                dtype=None,
                parallel_iterations=None,
                swap_memory=False,
                time_major=True,
                scope=None):
  """Creates a recurrent neural network specified by RNNCell `cell`.

  Performs fully dynamic unrolling of `inputs`.

  Example:

  ```python
  # create a BasicRNNCell
  rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

  # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]

  # defining initial state
  initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

  # 'state' is a tensor of shape [batch_size, cell_state_size]
  outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
                                     initial_state=initial_state,
                                     dtype=tf.float32)
  ```

  ```python
  # create 2 LSTMCells
  rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]

  # create a RNN cell composed sequentially of a number of RNNCells
  multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

  # 'outputs' is a tensor of shape [batch_size, max_time, 256]
  # 'state' is a N-tuple where N is the number of LSTMCells containing a
  # tf.contrib.rnn.LSTMStateTuple for each cell
  outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                     inputs=data,
                                     dtype=tf.float32)
  ```


  Args:
    cell: An instance of RNNCell.
    inputs: The RNN inputs.
      If `time_major == False` (default), this must be a `Tensor` of shape:
        `[batch_size, max_time, ...]`, or a nested tuple of such elements.
      If `time_major == True`, this must be a `Tensor` of shape: `[max_time,
        batch_size, ...]`, or a nested tuple of such elements. This may also be
        a (possibly nested) tuple of Tensors satisfying this property.  The
        first two dimensions must match across all the inputs, but otherwise the
        ranks and other shape components may differ. In this case, input to
        `cell` at each time-step will replicate the structure of these tuples,
        except for the time dimension (from which the time is taken). The input
        to `cell` at each time step will be a `Tensor` or (possibly nested)
        tuple of Tensors each with dimensions `[batch_size, ...]`.
    sequence_length: (optional) An int32/int64 vector sized `[batch_size]`. Used
      to copy-through state and zero-out outputs when past a batch element's
      sequence length.  So it's more for performance than correctness.
    initial_state: (optional) An initial state for the RNN. If `cell.state_size`
      is an integer, this must be a `Tensor` of appropriate type and shape
      `[batch_size, cell.state_size]`. If `cell.state_size` is a tuple, this
      should be a tuple of tensors having shapes `[batch_size, s] for s in
      cell.state_size`.
    dtype: (optional) The data type for the initial state and expected output.
      Required if initial_state is not provided or RNN state has a heterogeneous
      dtype.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency and
      can be run in parallel, will be.  This parameter trades off time for
      space.  Values >> 1 use more memory but take less time, while smaller
      values use less memory but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs which
      would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.
    time_major: The shape format of the `inputs` and `outputs` Tensors. If true,
      these `Tensors` must be shaped `[max_time, batch_size, depth]`. If false,
      these `Tensors` must be shaped `[batch_size, max_time, depth]`. Using
      `time_major = True` is a bit more efficient because it avoids transposes
      at the beginning and end of the RNN calculation.  However, most TensorFlow
      data is batch-major, so by default this function accepts input and emits
      output in batch-major form.
    scope: VariableScope for the created subgraph; defaults to "rnn".

  Returns:
    A pair (outputs, state) where:

    outputs: The RNN output `Tensor`.

      If time_major == False (default), this will be a `Tensor` shaped:
        `[batch_size, max_time, cell.output_size]`.

      If time_major == True, this will be a `Tensor` shaped:
        `[max_time, batch_size, cell.output_size]`.

      Note, if `cell.output_size` is a (possibly nested) tuple of integers
      or `TensorShape` objects, then `outputs` will be a tuple having the
      same structure as `cell.output_size`, containing Tensors having shapes
      corresponding to the shape data in `cell.output_size`.

    state: The final state.  If `cell.state_size` is an int, this
      will be shaped `[batch_size, cell.state_size]`.  If it is a
      `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
      If it is a (possibly nested) tuple of ints or `TensorShape`, this will
      be a tuple having the corresponding shapes. If cells are `LSTMCells`
      `state` will be a tuple containing a `LSTMStateTuple` for each cell.

  Raises:
    TypeError: If `cell` is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
    RuntimeError: If not using control flow v2.
  """

  # Currently only support time_major == True case.
  assert time_major

  # TODO(b/123051275): We need to check if the cells are TfLiteLSTMCells or
  # TfLiteRNNCells.
  rnn_cell_impl.assert_like_rnncell("cell", cell)

  if not control_flow_util.ENABLE_CONTROL_FLOW_V2:
    raise RuntimeError("OpHint dynamic rnn only supports control flow v2.")

  parent_first_child_input = [{
      "parent_ophint_input_index": 0,
      "first_child_ophint_input_index": 0
  }]
  parent_last_child_output = [{
      "parent_output_index": 0,
      # For LstmCell, the index is 2.
      # For RnnCell, the index is 1.
      # So we use -1 meaning it's the last one.
      "child_output_index": -1
  }]
  internal_children_input_output = [{
      "child_input_index": 0,
      # For LstmCell, the index is 2.
      # For RnnCell, the index is 1.
      # So we use -1 meaning it's the last one.
      "child_output_index": -1
  }]
  inputs_outputs_mappings = {
      "parent_first_child_input": parent_first_child_input,
      "parent_last_child_output": parent_last_child_output,
      "internal_children_input_output": internal_children_input_output
  }
  tflite_wrapper = lite.OpHint(
      "TfLiteDynamicRnn",
      level=2,
      children_inputs_mappings=inputs_outputs_mappings)
  with vs.variable_scope(scope or "rnn") as varscope:
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    if _should_cache():
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    inputs = tflite_wrapper.add_input(inputs, name="input", index_override=0)

    # By default, time_major==False and inputs are batch-major: shaped
    #   [batch, time, depth]
    # For internal calculations, we transpose to [time, batch, depth]
    flat_input = nest.flatten(inputs)

    if not time_major:
      # (batch, time, depth) => (time, batch, depth)
      flat_input = [ops.convert_to_tensor(input_) for input_ in flat_input]
      flat_input = tuple(_transpose_batch_time(input_) for input_ in flat_input)

    parallel_iterations = parallel_iterations or 32
    if sequence_length is not None:
      sequence_length = math_ops.to_int32(sequence_length)
      if sequence_length.get_shape().rank not in (None, 1):
        raise ValueError(
            "sequence_length must be a vector of length batch_size, "
            "but saw shape: %s" % sequence_length.get_shape())
      sequence_length = array_ops.identity(  # Just to find it in the graph.
          sequence_length,
          name="sequence_length")

    batch_size = _best_effort_input_batch_size(flat_input)

    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If there is no initial_state, you must give a dtype.")
      if getattr(cell, "get_initial_state", None) is not None:
        state = cell.get_initial_state(
            inputs=None, batch_size=batch_size, dtype=dtype)
      else:
        state = cell.zero_state(batch_size, dtype)

    def _assert_has_shape(x, shape):
      x_shape = array_ops.shape(x)
      packed_shape = array_ops.stack(shape)
      return control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)), [
              "Expected shape for Tensor %s is " % x.name, packed_shape,
              " but saw shape: ", x_shape
          ])

    if not context.executing_eagerly() and sequence_length is not None:
      # Perform some shape validation
      with ops.control_dependencies(
          [_assert_has_shape(sequence_length, [batch_size])]):
        sequence_length = array_ops.identity(
            sequence_length, name="CheckSeqLen")

    inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)

    outputs, final_state = _dynamic_rnn_loop(
        cell,
        inputs,
        state,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        sequence_length=sequence_length,
        dtype=dtype)

    # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
    # If we are performing batch-major calculations, transpose output back
    # to shape [batch, time, depth]
    if not time_major:
      # (time, batch, depth) => (batch, time, depth)
      outputs = nest.map_structure(_transpose_batch_time, outputs)
    outputs = tflite_wrapper.add_output(outputs, name="outputs")

    return outputs, final_state
