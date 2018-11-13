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
"""A tf.nn.dynamic_rnn variant, built on the Recurrent class.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from tensorflow.contrib.recurrent.python.ops import recurrent
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


def _GetDTypesFromStructure(struct):
  dtypes_list = []
  for x in nest.flatten(struct):
    x = ops.convert_to_tensor(x)
    dtypes_list.append(x.dtype)
  return dtypes_list


def _SetShapeFromTemplate(struct, struct_template):
  as_list = nest.flatten(struct)
  template_as_list = nest.flatten(struct_template)
  for element, template in zip(as_list, template_as_list):
    element.set_shape(template.shape)


class _FunctionalRnnCell(object):
  """Wrapper around RNNCell which separates state from computation.

  This class accomplishes the following:
  * Turn the cell's `__call__` function into a pure function. The global
    side effects are separated as `theta`. They are the variables created
    for the weights of the computation.
  * Unless the output is aliased as part of the state, extend the state to
    contain the output so that we store the history in `Recurrent`.
  * Set static shapes as required.
  """

  def __init__(self, rnn_cell, seq_inputs, initial_state):
    assert initial_state is not None

    # TODO(drpng): Dtype needs to be configurable.
    input_dtypes = [dtypes.float32] + _GetDTypesFromStructure(initial_state)
    # See _index.
    like_inputs_t = nest.map_structure(
        lambda x: array_ops.stop_gradient(array_ops.gather(x, 0)), seq_inputs)
    input_structure = (like_inputs_t, initial_state)

    @function.Defun(*input_dtypes)
    def FlatCellStep(*flat_inputs):
      """The flattened version of `rnn_cell`."""
      inputs_t, state0 = nest.pack_sequence_as(input_structure, flat_inputs)
      _SetShapeFromTemplate(state0, initial_state)
      _SetShapeFromTemplate(inputs_t, like_inputs_t)
      outputs_t, state1 = rnn_cell(inputs_t, state0)
      state_list = nest.flatten(state1)
      self._output_shape = outputs_t.shape

      if outputs_t in state_list:
        output_index_in_state = state_list.index(outputs_t)
      else:
        output_index_in_state = None

      if output_index_in_state is None:
        self._prepend_output = True
        self._output_state_idx = 0
        return [outputs_t] + state_list
      else:
        self._output_state_idx = output_index_in_state
        self._prepend_output = False
        # To save memory, we don't store return the output separately
        # from the state list, since we know it's the same.
        return state_list

    def _ToPureFunction(func):
      # NOTE: This forces the creating of the function.
      if func.captured_inputs:
        pure_func = copy.copy(func)
        # pylint: disable=protected-access
        pure_func._extra_inputs = []
        return pure_func
      return func

    pure_flat_cell_step = _ToPureFunction(FlatCellStep)

    def CellStep(theta, extended_state0, inputs_t):
      """Performs one time steps on structured inputs.

      The purpose of this function is to turn the parameters into flattened
      versions, and to resolve the parameter order difference between
      `Recurrent` and `RNNCell`.

      In the event the cell returns a transformed output that is not aliased
      within its state, the `extended_state0` also contains the output as its
      first element.

      Args:
        theta: Weights required for the computation. A structure of tensors.
        extended_state0: the state0, and possibly the output at the previous
          time step. A structure of tensors.
        inputs_t: the inputs at time t.

      Returns:
        A pair of the next state (inclusive of the output), and an empty list
        (unused `extras`).
        The next state is congruent to state0.
      """
      extended_state0_flat = nest.flatten(extended_state0)
      state0_flat = self.MaybeRemoveOutputFromState(extended_state0_flat)
      full_inputs = [inputs_t] + state0_flat + theta
      # Note that the thetas are additional inputs appeneded as extra
      # parameters.
      cell_out = pure_flat_cell_step(*full_inputs)
      return cell_out, []

    self._cell_step = CellStep
    self._theta = FlatCellStep.captured_inputs
    self._zero_state = rnn_cell.zero_state
    self._state_template = initial_state
    self._output_size = rnn_cell.output_size

  @property
  def extended_initial_state(self):
    if self._prepend_output:
      return [array_ops.zeros(self._output_shape), self._state_template]
    else:
      # The base case, where the output is just the hidden state.
      return self._state_template

  @property
  def cell_step(self):
    return self._cell_step

  @property
  def theta(self):
    return self._theta

  @property
  def state_template(self):
    return self._state_template

  @property
  def output_shape(self):
    return self._output_shape

  def GetOutputFromState(self, state):
    return nest.flatten(state)[self._output_state_idx]

  def MaybeRemoveOutputFromState(self, flat_state):
    if self._prepend_output:
      return flat_state[1:]
    return flat_state


def _ApplyLengthsToBatch(sequence_lengths, tf_output):
  # TODO(drpng): just use Update so that we don't carry over the gradients?
  """Sets the output to be zero at the end of the sequence."""
  # output is batch major.
  shape = array_ops.shape(tf_output)
  batch_size, max_time, vector_size = shape[0], shape[1], shape[2]
  output_time = array_ops.tile(math_ops.range(0, max_time), [batch_size])
  output_time = array_ops.reshape(output_time, [batch_size, max_time])
  lengths = array_ops.tile(
      array_ops.reshape(sequence_lengths, [-1, 1]), [1, max_time])
  is_less = math_ops.cast(
      math_ops.less(output_time, lengths), dtype=dtypes.float32)
  keep_mask = array_ops.tile(
      array_ops.expand_dims(is_less, -1),
      [1, 1, vector_size])
  final_output = keep_mask * tf_output
  return final_output


def _PickFinalStateFromHistory(acc_state, sequence_length):
  """Implements acc_state[sequence_length - 1]."""
  # This will work on all platforms, unlike the regular slice.
  last_value = []
  for state_var in nest.flatten(acc_state):
    # We compute the following with matrix operations:
    # last_var = state_var[sequence_length - 1]
    shape = array_ops.shape(state_var)
    max_time, batch_size = shape[0], shape[1]
    output_time = array_ops.tile(math_ops.range(0, max_time), [batch_size])
    output_time = array_ops.reshape(output_time, [batch_size, max_time])
    lengths = array_ops.tile(array_ops.reshape(sequence_length,
                                               [-1, 1]), [1, max_time])
    last_idx = math_ops.cast(math_ops.equal(output_time, lengths - 1),
                             dtype=state_var.dtype)
    last_idx = array_ops.transpose(last_idx)
    last_idx_for_bcast = array_ops.expand_dims(last_idx, -1)
    sliced = math_ops.multiply(last_idx_for_bcast, state_var)
    last_var = math_ops.reduce_sum(sliced, 0)
    last_value += [last_var]
  return nest.pack_sequence_as(acc_state, last_value)


def _PostProcessOutput(extended_acc_state, extended_final_state, func_cell,
                       total_time, inputs_lengths):
  """Post-process output of recurrent.

  This function takes the accumulated extended state and extracts the requested
  state and output.

  When `inputs_lengths` has been set, it extracts the output from the
  accumulated state. It also sets outputs past.

  It also sets the static shape information.

  Args:
    extended_acc_state: A structure containing the accumulated state at each
      time. It may contain the output at each time as well.
    extended_final_state: A structure containing the final state. It may
      contain the output at the final time.
    func_cell: The functional wrapper around the cell.
    total_time: A scalar integer tensor.
    inputs_lengths: An integer tensor with one entry per input.

  Returns:
    A tuple with the outputs at each time, and the final state.
  """
  if inputs_lengths is None:
    flat_final_state = func_cell.MaybeRemoveOutputFromState(
        nest.flatten(extended_final_state))
    tf_state = nest.pack_sequence_as(func_cell.state_template, flat_final_state)
  else:
    # The accumulated state is over the entire sequence, so we pick it
    # out from the acc_state sequence.
    flat_acc_state = func_cell.MaybeRemoveOutputFromState(
        nest.flatten(extended_acc_state))
    acc_state = nest.pack_sequence_as(
        func_cell.state_template, flat_acc_state)
    tf_state = _PickFinalStateFromHistory(acc_state, inputs_lengths)

  output_from_state = func_cell.GetOutputFromState(extended_acc_state)
  tf_output = array_ops.transpose(output_from_state, [1, 0, 2])
  tf_output.set_shape(
      [func_cell.output_shape[0], total_time, func_cell.output_shape[1]])
  if inputs_lengths is not None:
    # Need set the outputs to zero.
    tf_output = _ApplyLengthsToBatch(inputs_lengths, tf_output)
    # tf_output = array_ops.zeros([4, 3, 5])
  _SetShapeFromTemplate(tf_state, func_cell.state_template)
  return tf_output, tf_state


# pylint: disable=invalid-name
def functional_rnn(cell, inputs, sequence_length=None,
                   initial_state=None, dtype=None, time_major=False,
                   scope=None, use_tpu=False):
  """Same interface as `tf.nn.dynamic_rnn`."""
  with variable_scope.variable_scope(scope or 'rnn'):
    if not time_major:
      inputs = nest.map_structure(
          lambda t: array_ops.transpose(t, [1, 0, 2]), inputs)
    inputs_flat = nest.flatten(inputs)
    batch_size = array_ops.shape(inputs_flat[0])[1]
    if initial_state is None:
      initial_state = cell.zero_state(batch_size, dtype)
    func_cell = _FunctionalRnnCell(cell, inputs, initial_state)
  if sequence_length is not None:
    max_length = math_ops.reduce_max(sequence_length)
  else:
    max_length = None
  extended_acc_state, extended_final_state = recurrent.Recurrent(
      theta=func_cell.theta,
      state0=func_cell.extended_initial_state,
      inputs=inputs,
      cell_fn=func_cell.cell_step,
      max_input_length=max_length,
      use_tpu=use_tpu)
  tf_output, tf_state = _PostProcessOutput(
      extended_acc_state, extended_final_state, func_cell,
      inputs_flat[0].shape[0], sequence_length)

  if time_major:
    tf_output = array_ops.transpose(tf_output, [1, 0, 2])
  return tf_output, tf_state


def bidirectional_functional_rnn(
    cell_fw,
    cell_bw,
    inputs,
    initial_state_fw=None,
    initial_state_bw=None,
    dtype=None,
    sequence_length=None,
    time_major=False,
    use_tpu=False,
    scope=None):
  """Creates a bidirectional recurrent neural network.

  Performs fully dynamic unrolling of inputs in both directions. Built to be API
  compatible with `tf.nn.bidirectional_dynamic_rnn`, but implemented with
  functional control flow for TPU compatibility.

  Args:
    cell_fw: An instance of `tf.contrib.rnn.RNNCell`.
    cell_bw: An instance of `tf.contrib.rnn.RNNCell`.
    inputs: The RNN inputs. If time_major == False (default), this must be a
      Tensor (or hierarchical structure of Tensors) of shape
      [batch_size, max_time, ...]. If time_major == True, this must be a Tensor
      (or hierarchical structure of Tensors) of shape:
      [max_time, batch_size, ...]. The first two dimensions must match across
      all the inputs, but otherwise the ranks and other shape components may
      differ.
    initial_state_fw: An optional initial state for `cell_fw`. Should match
      `cell_fw.zero_state` in structure and type.
    initial_state_bw: An optional initial state for `cell_bw`. Should match
      `cell_bw.zero_state` in structure and type.
    dtype: (optional) The data type for the initial state and expected output.
      Required if initial_states are not provided or RNN state has a
      heterogeneous dtype.
    sequence_length: An optional int32/int64 vector sized [batch_size]. Used to
      copy-through state and zero-out outputs when past a batch element's
      sequence length. So it's more for correctness than performance.
    time_major: Whether the `inputs` tensor is in "time major" format.
    use_tpu: Whether to enable TPU-compatible operation. If True, does not truly
      reverse `inputs` in the backwards RNN. Once b/69305369 is fixed, we can
      remove this flag.
    scope: An optional scope name for the dynamic RNN.

  Returns:
    outputs: A tuple of `(output_fw, output_bw)`. The output of the forward and
      backward RNN. If time_major == False (default), these will
      be Tensors shaped: [batch_size, max_time, cell.output_size]. If
      time_major == True, these will be Tensors shaped:
      [max_time, batch_size, cell.output_size]. Note, if cell.output_size is a
      (possibly nested) tuple of integers or TensorShape objects, then the
      output for that direction will be a tuple having the same structure as
      cell.output_size, containing Tensors having shapes corresponding to the
      shape data in cell.output_size.
    final_states: A tuple of `(final_state_fw, final_state_bw)`. A Tensor or
      hierarchical structure of Tensors indicating the final cell state in each
      direction. Must have the same structure and shape as cell.zero_state.

  Raises:
    ValueError: If `initial_state_fw` is None or `initial_state_bw` is None and
      `dtype` is not provided.
  """
  # Keep this code in sync with tf.nn.dynamic_rnn for compatibility.
  with variable_scope.variable_scope(scope or 'bidirectional_rnn'):
    # Forward direction
    with variable_scope.variable_scope('fw') as fw_scope:
      output_fw, output_state_fw = functional_rnn(
          cell=cell_fw, inputs=inputs, sequence_length=sequence_length,
          initial_state=initial_state_fw, dtype=dtype,
          time_major=time_major, scope=fw_scope, use_tpu=use_tpu)
    # Backward direction
    if not time_major:
      time_dim = 1
      batch_dim = 0
    else:
      time_dim = 0
      batch_dim = 1

    def _reverse(input_, seq_lengths, seq_dim, batch_dim):
      if seq_lengths is not None:
        return array_ops.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_dim=seq_dim, batch_dim=batch_dim)
      else:
        # See b/69305369.
        assert not use_tpu, (
            'Bidirectional with variable sequence lengths unsupported on TPU')
        return array_ops.reverse(input_, axis=[seq_dim])

    with variable_scope.variable_scope('bw') as bw_scope:
      inputs_reverse = _reverse(
          inputs, seq_lengths=sequence_length,
          seq_dim=time_dim, batch_dim=batch_dim)
      tmp, output_state_bw = functional_rnn(
          cell=cell_bw, inputs=inputs_reverse, sequence_length=sequence_length,
          initial_state=initial_state_bw, dtype=dtype,
          time_major=time_major, scope=bw_scope, use_tpu=use_tpu)

  output_bw = _reverse(
      tmp, seq_lengths=sequence_length,
      seq_dim=time_dim, batch_dim=batch_dim)

  outputs = (output_fw, output_bw)
  output_states = (output_state_fw, output_state_bw)

  return (outputs, output_states)
# pylint: enable=invalid-name
