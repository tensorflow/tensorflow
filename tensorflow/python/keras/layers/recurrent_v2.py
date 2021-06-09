# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-classes-have-attributes
"""Recurrent layers for TF 2."""

import uuid

from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager.context import get_device_name
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers import recurrent
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import sysconfig
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export


# The following string constants are used by Defun approach for unified backend
# of LSTM and GRU.
_FUNCTION_API_NAME_ATTRIBUTE = 'api_implements'
_FUNCTION_DEVICE_ATTRIBUTE = 'api_preferred_device'
_CPU_DEVICE_NAME = 'CPU'
_GPU_DEVICE_NAME = 'GPU'

# The following number constants are used to represent the runtime of the defun
# backend function. Since the CPU/GPU implementation are mathematically same, we
# need some signal for the function to indicate which function is executed. This
# is for testing purpose to verify the correctness of swapping backend function.
_RUNTIME_UNKNOWN = 0
_RUNTIME_CPU = 1
_RUNTIME_GPU = 2

_CUDNN_AVAILABLE_MSG = 'Layer %s will use cuDNN kernels when running on GPU.'
_CUDNN_NOT_AVAILABLE_MSG = ('Layer %s will not use cuDNN kernels since it '
                            'doesn\'t meet the criteria. It will '
                            'use a generic GPU kernel as fallback when running '
                            'on GPU.')


def _use_new_code():
  return False


# TODO(b/169707691): The wrapper can be removed if TFLite doesn't need to rely
# on supportive attributes from LSTM/GRU.
class _DefunWrapper(object):
  """A wrapper with no deep copy of the Defun in LSTM/GRU layer."""

  def __init__(self, time_major, go_backwards, layer_name):
    self.time_major = time_major
    self.go_backwards = go_backwards
    self.layer_name = layer_name
    if self.layer_name not in ['lstm', 'gru']:
      raise ValueError('Defun wrapper only applies to LSTM and GRU layer, '
                       'but given {}'.format(self.layer_name))
    # The first two attributes are added to support TFLite use case.
    supportive_attributes = {
        'time_major': self.time_major,
        'go_backwards': self.go_backwards,
        _FUNCTION_API_NAME_ATTRIBUTE: self.layer_name + '_' + str(uuid.uuid4())
    }
    if self.layer_name == 'lstm':
      layer_func = lstm_with_backend_selection
    else:
      layer_func = gru_with_backend_selection

    self.defun_layer = function.defun_with_attributes(
        layer_func,
        attributes=supportive_attributes,
        autograph=False)

  def __deepcopy__(self, memo):
    new_wrapper = type(self)(
        self.time_major, self.go_backwards, self.layer_name)
    memo[id(self)] = new_wrapper
    return new_wrapper


@keras_export('keras.layers.GRUCell', v1=[])
class GRUCell(recurrent.GRUCell):
  """Cell class for the GRU layer.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  This class processes one step within the whole time sequence input, whereas
  `tf.keras.layer.GRU` processes the whole sequence.

  For example:

  >>> inputs = tf.random.normal([32, 10, 8])
  >>> rnn = tf.keras.layers.RNN(tf.keras.layers.GRUCell(4))
  >>> output = rnn(inputs)
  >>> print(output.shape)
  (32, 4)
  >>> rnn = tf.keras.layers.RNN(
  ...    tf.keras.layers.GRUCell(4),
  ...    return_sequences=True,
  ...    return_state=True)
  >>> whole_sequence_output, final_state = rnn(inputs)
  >>> print(whole_sequence_output.shape)
  (32, 10, 4)
  >>> print(final_state.shape)
  (32, 4)

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use. Default: hyperbolic tangent
      (`tanh`). If you pass None, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use for the recurrent step.
      Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
      applied (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs. Default:
      `glorot_uniform`.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix, used for the linear transformation of the recurrent state.
      Default: `orthogonal`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to the
      `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the
      linear transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state. Default: 0.
    reset_after: GRU convention (whether to apply reset gate after or
      before matrix multiplication). False = "before",
      True = "after" (default and CuDNN compatible).

  Call arguments:
    inputs: A 2D tensor, with shape of `[batch, feature]`.
    states: A 2D tensor with shape of `[batch, units]`, which is the state from
      the previous time step. For timestep 0, the initial state provided by user
      will be feed to cell.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               reset_after=True,
               **kwargs):
    super(GRUCell, self).__init__(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=kwargs.pop('implementation', 2),
        reset_after=reset_after,
        **kwargs)


@keras_export('keras.layers.GRU', v1=[])
class GRU(recurrent.DropoutRNNCellMixin, recurrent.GRU):
  """Gated Recurrent Unit - Cho et al. 2014.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  Based on available runtime hardware and constraints, this layer
  will choose different implementations (cuDNN-based or pure-TensorFlow)
  to maximize the performance. If a GPU is available and all
  the arguments to the layer meet the requirement of the CuDNN kernel
  (see below for details), the layer will use a fast cuDNN implementation.

  The requirements to use the cuDNN implementation are:

  1. `activation` == `tanh`
  2. `recurrent_activation` == `sigmoid`
  3. `recurrent_dropout` == 0
  4. `unroll` is `False`
  5. `use_bias` is `True`
  6. `reset_after` is `True`
  7. Inputs, if use masking, are strictly right-padded.
  8. Eager execution is enabled in the outermost context.

  There are two variants of the GRU implementation. The default one is based on
  [v3](https://arxiv.org/abs/1406.1078v3) and has reset gate applied to hidden
  state before matrix multiplication. The other one is based on
  [original](https://arxiv.org/abs/1406.1078v1) and has the order reversed.

  The second variant is compatible with CuDNNGRU (GPU-only) and allows
  inference on CPU. Thus it has separate biases for `kernel` and
  `recurrent_kernel`. To use this variant, set `'reset_after'=True` and
  `recurrent_activation='sigmoid'`.

  For example:

  >>> inputs = tf.random.normal([32, 10, 8])
  >>> gru = tf.keras.layers.GRU(4)
  >>> output = gru(inputs)
  >>> print(output.shape)
  (32, 4)
  >>> gru = tf.keras.layers.GRU(4, return_sequences=True, return_state=True)
  >>> whole_sequence_output, final_state = gru(inputs)
  >>> print(whole_sequence_output.shape)
  (32, 10, 4)
  >>> print(final_state.shape)
  (32, 4)

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: sigmoid (`sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs. Default:
      `glorot_uniform`.
    recurrent_initializer: Initializer for the `recurrent_kernel`
       weights matrix, used for the linear transformation of the recurrent
       state. Default: `orthogonal`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to the
      `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation"). Default: `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the linear
      transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state. Default: 0.
    return_sequences: Boolean. Whether to return the last output
      in the output sequence, or the full sequence. Default: `False`.
    return_state: Boolean. Whether to return the last state in addition to the
      output. Default: `False`.
    go_backwards: Boolean (default `False`).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default False). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll: Boolean (default False).
      If True, the network will be unrolled,
      else a symbolic loop will be used.
      Unrolling can speed-up a RNN,
      although it tends to be more memory-intensive.
      Unrolling is only suitable for short sequences.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `[timesteps, batch, feature]`, whereas in the False case, it will be
      `[batch, timesteps, feature]`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.
    reset_after: GRU convention (whether to apply reset gate after or
      before matrix multiplication). False = "before",
      True = "after" (default and CuDNN compatible).

  Call arguments:
    inputs: A 3D tensor, with shape `[batch, timesteps, feature]`.
    mask: Binary tensor of shape `[samples, timesteps]` indicating whether
      a given timestep should be masked  (optional, defaults to `None`).
      An individual `True` entry indicates that the corresponding timestep
      should be utilized, while a `False` entry indicates that the
      corresponding timestep should be ignored.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used  (optional, defaults to `None`).
    initial_state: List of initial state tensors to be passed to the first
      call of the cell  (optional, defaults to `None` which causes creation
      of zero-filled initial state tensors).
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               time_major=False,
               reset_after=True,
               **kwargs):
    # return_runtime is a flag for testing, which shows the real backend
    # implementation chosen by grappler in graph mode.
    self._return_runtime = kwargs.pop('return_runtime', False)

    super(GRU, self).__init__(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=kwargs.pop('implementation', 2),
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        time_major=time_major,
        reset_after=reset_after,
        **kwargs)
    # GPU kernel uses following setting by default and not configurable.
    self._could_use_gpu_kernel = (
        self.activation in (activations.tanh, nn.tanh) and
        self.recurrent_activation in (activations.sigmoid, nn.sigmoid) and
        recurrent_dropout == 0 and not unroll and use_bias and
        reset_after and ops.executing_eagerly_outside_functions())
    if config.list_logical_devices('GPU'):
      # Only show the message when there is GPU available, user will not care
      # about the cuDNN if there isn't any GPU.
      if self._could_use_gpu_kernel:
        logging.debug(_CUDNN_AVAILABLE_MSG % self.name)
      else:
        logging.warning(_CUDNN_NOT_AVAILABLE_MSG % self.name)

    if _use_new_code():
      self._defun_wrapper = _DefunWrapper(time_major, go_backwards, 'gru')

  def call(self, inputs, mask=None, training=None, initial_state=None):
    # The input should be dense, padded with zeros. If a ragged input is fed
    # into the layer, it is padded and the row lengths are used for masking.
    inputs, row_lengths = backend.convert_inputs_if_ragged(inputs)
    is_ragged_input = (row_lengths is not None)
    self._validate_args_if_ragged(is_ragged_input, mask)

    # GRU does not support constants. Ignore it during process.
    inputs, initial_state, _ = self._process_inputs(inputs, initial_state, None)

    if isinstance(mask, list):
      mask = mask[0]

    input_shape = backend.int_shape(inputs)
    timesteps = input_shape[0] if self.time_major else input_shape[1]

    # TODO(b/156447398) Investigate why the cuDNN kernel fails with ragged
    # inputs.
    if is_ragged_input or not self._could_use_gpu_kernel:
      kwargs = {'training': training}
      self._maybe_reset_cell_dropout_mask(self.cell)

      def step(cell_inputs, cell_states):
        return self.cell(cell_inputs, cell_states, **kwargs)

      last_output, outputs, states = backend.rnn(
          step,
          inputs,
          initial_state,
          constants=None,
          go_backwards=self.go_backwards,
          mask=mask,
          unroll=self.unroll,
          input_length=row_lengths if row_lengths is not None else timesteps,
          time_major=self.time_major,
          zero_output_for_mask=self.zero_output_for_mask)
      # This is a dummy tensor for testing purpose.
      runtime = _runtime(_RUNTIME_UNKNOWN)
    else:
      last_output, outputs, runtime, states = self._defun_gru_call(
          inputs, initial_state, training, mask, row_lengths)

    if self.stateful:
      updates = [state_ops.assign(self.states[0], states[0])]
      self.add_update(updates)

    if self.return_sequences:
      output = backend.maybe_convert_to_ragged(
          is_ragged_input, outputs, row_lengths, go_backwards=self.go_backwards)
    else:
      output = last_output

    if self.return_state:
      return [output] + list(states)
    elif self._return_runtime:
      return output, runtime
    else:
      return output

  def _defun_gru_call(self, inputs, initial_state, training, mask,
                      sequence_lengths):
    # Use the new defun approach for backend implementation swap.
    # Note that different implementations need to have same function
    # signature, eg, the tensor parameters need to have same shape and dtypes.

    self.reset_dropout_mask()
    dropout_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
    if dropout_mask is not None:
      inputs = inputs * dropout_mask[0]

    if _use_new_code():
      gru_kwargs = {
          'inputs': inputs,
          'init_h': _read_variable_value(initial_state[0]),
          'kernel': _read_variable_value(self.cell.kernel),
          'recurrent_kernel': _read_variable_value(self.cell.recurrent_kernel),
          'bias': _read_variable_value(self.cell.bias),
          'mask': mask,
          'time_major': self.time_major,
          'go_backwards': self.go_backwards,
          'sequence_lengths': sequence_lengths,
          'zero_output_for_mask': self.zero_output_for_mask
      }
      (last_output, outputs, new_h,
       runtime) = self._defun_wrapper.defun_layer(**gru_kwargs)
    else:
      gpu_gru_kwargs = {
          'inputs': inputs,
          'init_h': _read_variable_value(initial_state[0]),
          'kernel': _read_variable_value(self.cell.kernel),
          'recurrent_kernel': _read_variable_value(self.cell.recurrent_kernel),
          'bias': _read_variable_value(self.cell.bias),
          'mask': mask,
          'time_major': self.time_major,
          'go_backwards': self.go_backwards,
          'sequence_lengths': sequence_lengths
      }
      normal_gru_kwargs = gpu_gru_kwargs.copy()
      normal_gru_kwargs.update({
          'zero_output_for_mask': self.zero_output_for_mask,
      })

      if context.executing_eagerly():
        device_type = _get_context_device_type()
        can_use_gpu = (
            # Either user specified GPU or unspecified but GPU is available.
            (device_type == _GPU_DEVICE_NAME or
             (device_type is None and config.list_logical_devices('GPU'))) and
            (mask is None or is_cudnn_supported_inputs(mask, self.time_major)))
        # Under eager context, check the device placement and prefer the
        if can_use_gpu:
          last_output, outputs, new_h, runtime = gpu_gru(**gpu_gru_kwargs)
        else:
          last_output, outputs, new_h, runtime = standard_gru(
              **normal_gru_kwargs)
      else:
        last_output, outputs, new_h, runtime = gru_with_backend_selection(
            **normal_gru_kwargs)

    states = [new_h]
    return last_output, outputs, runtime, states


def standard_gru(inputs, init_h, kernel, recurrent_kernel, bias, mask,
                 time_major, go_backwards, sequence_lengths,
                 zero_output_for_mask):
  """GRU with standard kernel implementation.

  This implementation can be run on all types of hardware.

  This implementation lifts out all the layer weights and make them function
  parameters. It has same number of tensor input params as the CuDNN
  counterpart. The RNN step logic has been simplified, eg dropout and mask is
  removed since CuDNN implementation does not support that.

  Args:
    inputs: Input tensor of GRU layer.
    init_h: Initial state tensor for the cell output.
    kernel: Weights for cell kernel.
    recurrent_kernel: Weights for cell recurrent kernel.
    bias: Weights for cell kernel bias and recurrent bias. The bias contains the
      combined input_bias and recurrent_bias.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked. An individual `True` entry indicates
      that the corresponding timestep should be utilized, while a `False` entry
      indicates that the corresponding timestep should be ignored.
    time_major: Boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.
    zero_output_for_mask: Boolean, whether to output zero for masked timestep.

  Returns:
    last_output: output tensor for the last timestep, which has shape
      [batch, units].
    outputs: output tensor for all timesteps, which has shape
      [batch, time, units].
    state_0: the cell output, which has same shape as init_h.
    runtime: constant string tensor which indicate real runtime hardware. This
      value is for testing purpose and should be used by user.
  """
  input_shape = backend.int_shape(inputs)
  timesteps = input_shape[0] if time_major else input_shape[1]

  input_bias, recurrent_bias = array_ops.unstack(bias)

  def step(cell_inputs, cell_states):
    """Step function that will be used by Keras RNN backend."""
    h_tm1 = cell_states[0]

    # inputs projected by all gate matrices at once
    matrix_x = backend.dot(cell_inputs, kernel)
    matrix_x = backend.bias_add(matrix_x, input_bias)

    x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=1)

    # hidden state projected by all gate matrices at once
    matrix_inner = backend.dot(h_tm1, recurrent_kernel)
    matrix_inner = backend.bias_add(matrix_inner, recurrent_bias)

    recurrent_z, recurrent_r, recurrent_h = array_ops.split(matrix_inner, 3,
                                                            axis=1)
    z = nn.sigmoid(x_z + recurrent_z)
    r = nn.sigmoid(x_r + recurrent_r)
    hh = nn.tanh(x_h + r * recurrent_h)

    # previous and candidate state mixed by update gate
    h = z * h_tm1 + (1 - z) * hh
    return h, [h]

  last_output, outputs, new_states = backend.rnn(
      step,
      inputs, [init_h],
      constants=None,
      unroll=False,
      time_major=time_major,
      mask=mask,
      go_backwards=go_backwards,
      input_length=sequence_lengths
      if sequence_lengths is not None else timesteps,
      zero_output_for_mask=zero_output_for_mask)
  return last_output, outputs, new_states[0], _runtime(_RUNTIME_CPU)


def gpu_gru(inputs, init_h, kernel, recurrent_kernel, bias, mask, time_major,
            go_backwards, sequence_lengths):
  """GRU with CuDNN implementation which is only available for GPU."""
  if not time_major and mask is None:
    inputs = array_ops.transpose(inputs, perm=(1, 0, 2))
    seq_axis, batch_axis = (0, 1)
  else:
    seq_axis, batch_axis = (0, 1) if time_major else (1, 0)
  # For init_h, cuDNN expects one more dim of num_layers before or after batch
  # dim for time major or batch major inputs respectively
  init_h = array_ops.expand_dims(init_h, axis=seq_axis)

  weights = array_ops.split(kernel, 3, axis=1)
  weights += array_ops.split(recurrent_kernel, 3, axis=1)
  # Note that the bias was initialized as shape (2, 3 * units), flat it into
  # (6 * units)
  bias = array_ops.split(backend.flatten(bias), 6)

  if sysconfig.get_build_info()['is_cuda_build']:
    # Note that the gate order for CuDNN is different from the canonical format.
    # canonical format is [z, r, h], whereas CuDNN is [r, z, h]. The swap need
    # to be done for kernel, recurrent_kernel, input_bias, recurrent_bias.
    # z is update gate weights.
    # r is reset gate weights.
    # h is output gate weights.
    weights[0], weights[1] = weights[1], weights[0]
    weights[3], weights[4] = weights[4], weights[3]
    bias[0], bias[1] = bias[1], bias[0]
    bias[3], bias[4] = bias[4], bias[3]

  params = _canonical_to_params(
      weights=weights,
      biases=bias,
      shape=constant_op.constant([-1]),
      transpose_weights=True)

  if mask is not None:
    sequence_lengths = calculate_sequence_by_mask(mask, time_major)

  if sequence_lengths is not None:
    if go_backwards:
      # Three reversals are required. E.g.,
      # normal input = [1, 2, 3, 0, 0]  # where 0 need to be masked
      # reversed_input_to_cudnn = [3, 2, 1, 0, 0]
      # output_from_cudnn = [6, 5, 4, 0, 0]
      # expected_output = [0, 0, 6, 5 ,4]
      inputs = array_ops.reverse_sequence_v2(
          inputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
    outputs, h, _, _, _ = gen_cudnn_rnn_ops.CudnnRNNV3(
        input=inputs,
        input_h=init_h,
        input_c=0,
        params=params,
        is_training=True,
        rnn_mode='gru',
        sequence_lengths=sequence_lengths,
        time_major=time_major)
    if go_backwards:
      outputs = array_ops.reverse_sequence_v2(
          outputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
      outputs = array_ops.reverse(outputs, axis=[seq_axis])
  else:
    if go_backwards:
      # Reverse axis 0 since the input is already convert to time major.
      inputs = array_ops.reverse(inputs, axis=[0])
    outputs, h, _, _ = gen_cudnn_rnn_ops.CudnnRNN(
        input=inputs, input_h=init_h, input_c=0, params=params,
        is_training=True, rnn_mode='gru')

  last_output = outputs[-1]
  if not time_major and mask is None:
    outputs = array_ops.transpose(outputs, perm=[1, 0, 2])
  h = array_ops.squeeze(h, axis=seq_axis)

  # In the case of variable length input, the cudnn kernel will fill zeros for
  # the output, whereas the default keras behavior is to bring over the previous
  # output for t-1, so that in the return_sequence=False case, user can quickly
  # get the final effect output instead just 0s at the last timestep.
  # In order to mimic the default keras behavior, we copy the final h state as
  # the last_output, since it is numerically same as the output.
  if mask is not None:
    last_output = h

  return last_output, outputs, h, _runtime(_RUNTIME_GPU)


def gru_with_backend_selection(inputs, init_h, kernel, recurrent_kernel, bias,
                               mask, time_major, go_backwards, sequence_lengths,
                               zero_output_for_mask):
  """Call the GRU with optimized backend kernel selection.

  Under the hood, this function will create two TF function, one with the most
  generic kernel and can run on all device condition, and the second one with
  CuDNN specific kernel, which can only run on GPU.

  The first function will be called with normal_lstm_params, while the second
  function is not called, but only registered in the graph. The Grappler will
  do the proper graph rewrite and swap the optimized TF function based on the
  device placement.

  Args:
    inputs: Input tensor of GRU layer.
    init_h: Initial state tensor for the cell output.
    kernel: Weights for cell kernel.
    recurrent_kernel: Weights for cell recurrent kernel.
    bias: Weights for cell kernel bias and recurrent bias. Only recurrent bias
      is used in this case.
    mask: Boolean tensor for mask out the steps within sequence.
      An individual `True` entry indicates that the corresponding timestep
      should be utilized, while a `False` entry indicates that the corresponding
      timestep should be ignored.
    time_major: Boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.
    zero_output_for_mask: Boolean, whether to output zero for masked timestep.

  Returns:
    List of output tensors, same as standard_gru.
  """
  params = {
      'inputs': inputs,
      'init_h': init_h,
      'kernel': kernel,
      'recurrent_kernel': recurrent_kernel,
      'bias': bias,
      'mask': mask,
      'time_major': time_major,
      'go_backwards': go_backwards,
      'sequence_lengths': sequence_lengths,
      'zero_output_for_mask': zero_output_for_mask,
  }

  def gpu_gru_with_fallback(inputs, init_h, kernel, recurrent_kernel, bias,
                            mask, time_major, go_backwards, sequence_lengths,
                            zero_output_for_mask):
    """Use CuDNN kernel when mask is none or strictly right padded."""
    if mask is None:
      return gpu_gru(
          inputs=inputs,
          init_h=init_h,
          kernel=kernel,
          recurrent_kernel=recurrent_kernel,
          bias=bias,
          mask=mask,
          time_major=time_major,
          go_backwards=go_backwards,
          sequence_lengths=sequence_lengths)

    def cudnn_gru_fn():
      return gpu_gru(
          inputs=inputs,
          init_h=init_h,
          kernel=kernel,
          recurrent_kernel=recurrent_kernel,
          bias=bias,
          mask=mask,
          time_major=time_major,
          go_backwards=go_backwards,
          sequence_lengths=sequence_lengths)

    def standard_gru_fn():
      return standard_gru(
          inputs=inputs,
          init_h=init_h,
          kernel=kernel,
          recurrent_kernel=recurrent_kernel,
          bias=bias,
          mask=mask,
          time_major=time_major,
          go_backwards=go_backwards,
          sequence_lengths=sequence_lengths,
          zero_output_for_mask=zero_output_for_mask)

    return control_flow_ops.cond(
        is_cudnn_supported_inputs(mask, time_major),
        true_fn=cudnn_gru_fn,
        false_fn=standard_gru_fn)

  if _use_new_code():
    # Chooses the implementation dynamically based on the running device.
    (last_output, outputs, new_h,
     runtime) = control_flow_ops.execute_fn_for_device(
         {
             _CPU_DEVICE_NAME: lambda: standard_gru(**params),
             _GPU_DEVICE_NAME: lambda: gpu_gru_with_fallback(**params)
         }, lambda: standard_gru(**params))
  else:
    # Each time a `tf.function` is called, we will give it a unique
    # identifiable API name, so that Grappler won't get confused when it
    # sees multiple GRU layers added into same graph, and it will be able
    # to pair up the different implementations across them.
    api_name = 'gru_' + str(uuid.uuid4())
    supportive_attribute = {
        'time_major': time_major,
        'go_backwards': go_backwards,
    }
    defun_standard_gru = _generate_defun_backend(api_name, _CPU_DEVICE_NAME,
                                                 standard_gru,
                                                 supportive_attribute)
    defun_gpu_gru = _generate_defun_backend(api_name, _GPU_DEVICE_NAME,
                                            gpu_gru_with_fallback,
                                            supportive_attribute)

    # Call the normal GRU impl and register the CuDNN impl function. The
    # grappler will kick in during session execution to optimize the graph.
    last_output, outputs, new_h, runtime = defun_standard_gru(**params)
    _function_register(defun_gpu_gru, **params)

  return last_output, outputs, new_h, runtime


@keras_export('keras.layers.LSTMCell', v1=[])
class LSTMCell(recurrent.LSTMCell):
  """Cell class for the LSTM layer.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  This class processes one step within the whole time sequence input, whereas
  `tf.keras.layer.LSTM` processes the whole sequence.

  For example:

  >>> inputs = tf.random.normal([32, 10, 8])
  >>> rnn = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(4))
  >>> output = rnn(inputs)
  >>> print(output.shape)
  (32, 4)
  >>> rnn = tf.keras.layers.RNN(
  ...    tf.keras.layers.LSTMCell(4),
  ...    return_sequences=True,
  ...    return_state=True)
  >>> whole_seq_output, final_memory_state, final_carry_state = rnn(inputs)
  >>> print(whole_seq_output.shape)
  (32, 10, 4)
  >>> print(final_memory_state.shape)
  (32, 4)
  >>> print(final_carry_state.shape)
  (32, 4)

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use. Default: hyperbolic tangent
      (`tanh`). If you pass `None`, no activation is applied (ie. "linear"
      activation: `a(x) = x`).
    recurrent_activation: Activation function to use for the recurrent step.
      Default: sigmoid (`sigmoid`). If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix, used for
      the linear transformation of the inputs. Default: `glorot_uniform`.
    recurrent_initializer: Initializer for the `recurrent_kernel` weights
      matrix, used for the linear transformation of the recurrent state.
      Default: `orthogonal`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    unit_forget_bias: Boolean (default `True`). If True, add 1 to the bias of
      the forget gate at initialization. Setting it to true will also force
      `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
        al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the linear
      transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state. Default: 0.

  Call arguments:
    inputs: A 2D tensor, with shape of `[batch, feature]`.
    states: List of 2 tensors that corresponding to the cell's units. Both of
      them have shape `[batch, units]`, the first tensor is the memory state
      from previous time step, the second tensor is the carry state from
      previous time step. For timestep 0, the initial state provided by user
      will be feed to cell.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
    super(LSTMCell, self).__init__(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=kwargs.pop('implementation', 2),
        **kwargs)


@keras_export('keras.layers.LSTM', v1=[])
class LSTM(recurrent.DropoutRNNCellMixin, recurrent.LSTM):
  """Long Short-Term Memory layer - Hochreiter 1997.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  Based on available runtime hardware and constraints, this layer
  will choose different implementations (cuDNN-based or pure-TensorFlow)
  to maximize the performance. If a GPU is available and all
  the arguments to the layer meet the requirement of the CuDNN kernel
  (see below for details), the layer will use a fast cuDNN implementation.

  The requirements to use the cuDNN implementation are:

  1. `activation` == `tanh`
  2. `recurrent_activation` == `sigmoid`
  3. `recurrent_dropout` == 0
  4. `unroll` is `False`
  5. `use_bias` is `True`
  6. Inputs, if use masking, are strictly right-padded.
  7. Eager execution is enabled in the outermost context.

  For example:

  >>> inputs = tf.random.normal([32, 10, 8])
  >>> lstm = tf.keras.layers.LSTM(4)
  >>> output = lstm(inputs)
  >>> print(output.shape)
  (32, 4)
  >>> lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
  >>> whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
  >>> print(whole_seq_output.shape)
  (32, 10, 4)
  >>> print(final_memory_state.shape)
  (32, 4)
  >>> print(final_carry_state.shape)
  (32, 4)

  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`). If you pass `None`, no activation
      is applied (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use for the recurrent step.
      Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
      applied (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean (default `True`), whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix, used for
      the linear transformation of the inputs. Default: `glorot_uniform`.
    recurrent_initializer: Initializer for the `recurrent_kernel` weights
      matrix, used for the linear transformation of the recurrent state.
      Default: `orthogonal`.
    bias_initializer: Initializer for the bias vector. Default: `zeros`.
    unit_forget_bias: Boolean (default `True`). If True, add 1 to the bias of
      the forget gate at initialization. Setting it to true will also force
      `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
          al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_regularizer: Regularizer function applied to the
      `recurrent_kernel` weights matrix. Default: `None`.
    bias_regularizer: Regularizer function applied to the bias vector. Default:
      `None`.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation"). Default: `None`.
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix. Default: `None`.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix. Default: `None`.
    bias_constraint: Constraint function applied to the bias vector. Default:
      `None`.
    dropout: Float between 0 and 1. Fraction of the units to drop for the linear
      transformation of the inputs. Default: 0.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state. Default: 0.
    return_sequences: Boolean. Whether to return the last output. in the output
      sequence, or the full sequence. Default: `False`.
    return_state: Boolean. Whether to return the last state in addition to the
      output. Default: `False`.
    go_backwards: Boolean (default `False`). If True, process the input sequence
      backwards and return the reversed sequence.
    stateful: Boolean (default `False`). If True, the last state for each sample
      at index i in a batch will be used as initial state for the sample of
      index i in the following batch.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `[timesteps, batch, feature]`, whereas in the False case, it will be
      `[batch, timesteps, feature]`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.
    unroll: Boolean (default `False`). If True, the network will be unrolled,
      else a symbolic loop will be used. Unrolling can speed-up a RNN, although
      it tends to be more memory-intensive. Unrolling is only suitable for short
      sequences.

  Call arguments:
    inputs: A 3D tensor with shape `[batch, timesteps, feature]`.
    mask: Binary tensor of shape `[batch, timesteps]` indicating whether
      a given timestep should be masked (optional, defaults to `None`).
      An individual `True` entry indicates that the corresponding timestep
      should be utilized, while a `False` entry indicates that the corresponding
      timestep should be ignored.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used (optional, defaults to `None`).
    initial_state: List of initial state tensors to be passed to the first
      call of the cell (optional, defaults to `None` which causes creation
      of zero-filled initial state tensors).
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               time_major=False,
               unroll=False,
               **kwargs):
    # return_runtime is a flag for testing, which shows the real backend
    # implementation chosen by grappler in graph mode.
    self.return_runtime = kwargs.pop('return_runtime', False)

    super(LSTM, self).__init__(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=kwargs.pop('implementation', 2),
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        time_major=time_major,
        unroll=unroll,
        **kwargs)

    self.state_spec = [
        InputSpec(shape=(None, dim)) for dim in (self.units, self.units)
    ]
    self._could_use_gpu_kernel = (
        self.activation in (activations.tanh, nn.tanh) and
        self.recurrent_activation in (activations.sigmoid, nn.sigmoid) and
        recurrent_dropout == 0 and not unroll and use_bias and
        ops.executing_eagerly_outside_functions())
    if config.list_logical_devices('GPU'):
      # Only show the message when there is GPU available, user will not care
      # about the cuDNN if there isn't any GPU.
      if self._could_use_gpu_kernel:
        logging.debug(_CUDNN_AVAILABLE_MSG % self.name)
      else:
        logging.warning(_CUDNN_NOT_AVAILABLE_MSG % self.name)

    if _use_new_code():
      self._defun_wrapper = _DefunWrapper(time_major, go_backwards, 'lstm')

  def call(self, inputs, mask=None, training=None, initial_state=None):
    # The input should be dense, padded with zeros. If a ragged input is fed
    # into the layer, it is padded and the row lengths are used for masking.
    inputs, row_lengths = backend.convert_inputs_if_ragged(inputs)
    is_ragged_input = (row_lengths is not None)
    self._validate_args_if_ragged(is_ragged_input, mask)

    # LSTM does not support constants. Ignore it during process.
    inputs, initial_state, _ = self._process_inputs(inputs, initial_state, None)

    if isinstance(mask, list):
      mask = mask[0]

    input_shape = backend.int_shape(inputs)
    timesteps = input_shape[0] if self.time_major else input_shape[1]

    # TODO(b/156447398) Investigate why the cuDNN kernel fails with ragged
    # inputs.
    if is_ragged_input or not self._could_use_gpu_kernel:
      # Fall back to use the normal LSTM.
      kwargs = {'training': training}
      self._maybe_reset_cell_dropout_mask(self.cell)

      def step(inputs, states):
        return self.cell(inputs, states, **kwargs)

      last_output, outputs, states = backend.rnn(
          step,
          inputs,
          initial_state,
          constants=None,
          go_backwards=self.go_backwards,
          mask=mask,
          unroll=self.unroll,
          input_length=row_lengths if row_lengths is not None else timesteps,
          time_major=self.time_major,
          zero_output_for_mask=self.zero_output_for_mask)
      runtime = _runtime(_RUNTIME_UNKNOWN)
    else:
      # Use the new defun approach for backend implementation swap.
      # Note that different implementations need to have same function
      # signature, eg, the tensor parameters need to have same shape and dtypes.
      # Since the CuDNN has an extra set of bias, those bias will be passed to
      # both normal and CuDNN implementations.
      self.reset_dropout_mask()
      dropout_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
      if dropout_mask is not None:
        inputs = inputs * dropout_mask[0]
      if _use_new_code():
        lstm_kwargs = {
            'inputs':
                inputs,
            'init_h':
                _read_variable_value(initial_state[0]),
            'init_c':
                _read_variable_value(initial_state[1]),
            'kernel':
                _read_variable_value(self.cell.kernel),
            'recurrent_kernel':
                _read_variable_value(self.cell.recurrent_kernel),
            'bias':
                _read_variable_value(self.cell.bias),
            'mask':
                mask,
            'time_major':
                self.time_major,
            'go_backwards':
                self.go_backwards,
            'sequence_lengths':
                row_lengths,
            'zero_output_for_mask':
                self.zero_output_for_mask,
        }
        (last_output, outputs, new_h, new_c,
         runtime) = self._defun_wrapper.defun_layer(**lstm_kwargs)
      else:
        gpu_lstm_kwargs = {
            'inputs':
                inputs,
            'init_h':
                _read_variable_value(initial_state[0]),
            'init_c':
                _read_variable_value(initial_state[1]),
            'kernel':
                _read_variable_value(self.cell.kernel),
            'recurrent_kernel':
                _read_variable_value(self.cell.recurrent_kernel),
            'bias':
                _read_variable_value(self.cell.bias),
            'mask':
                mask,
            'time_major':
                self.time_major,
            'go_backwards':
                self.go_backwards,
            'sequence_lengths':
                row_lengths
        }
        normal_lstm_kwargs = gpu_lstm_kwargs.copy()
        normal_lstm_kwargs.update({
            'zero_output_for_mask': self.zero_output_for_mask,
        })

        if context.executing_eagerly():
          device_type = _get_context_device_type()
          can_use_gpu = (
              # Either user specified GPU or unspecified but GPU is available.
              (device_type == _GPU_DEVICE_NAME or
               (device_type is None and config.list_logical_devices('GPU'))) and
              (mask is None or
               is_cudnn_supported_inputs(mask, self.time_major)))
          # Under eager context, check the device placement and prefer the
          # GPU implementation when GPU is available.
          if can_use_gpu:
            last_output, outputs, new_h, new_c, runtime = gpu_lstm(
                **gpu_lstm_kwargs)
          else:
            last_output, outputs, new_h, new_c, runtime = standard_lstm(
                **normal_lstm_kwargs)
        else:
          (last_output, outputs, new_h, new_c,
           runtime) = lstm_with_backend_selection(**normal_lstm_kwargs)

      states = [new_h, new_c]

    if self.stateful:
      updates = [
          state_ops.assign(self_state, state)
          for self_state, state in zip(self.states, states)
      ]
      self.add_update(updates)

    if self.return_sequences:
      output = backend.maybe_convert_to_ragged(
          is_ragged_input, outputs, row_lengths, go_backwards=self.go_backwards)
    else:
      output = last_output

    if self.return_state:
      return [output] + list(states)
    elif self.return_runtime:
      return output, runtime
    else:
      return output


def _canonical_to_params(weights, biases, shape, transpose_weights=False):
  """Utility function convert variable to CuDNN compatible parameter.

  Note that Keras weights for kernels are different from the CuDNN format. Eg.:

  ```
    Keras                 CuDNN
    [[0, 1, 2],  <--->  [[0, 2, 4],
     [3, 4, 5]]          [1, 3, 5]]
  ```

  If the input weights need to be in a unified format, then set
  `transpose_weights=True` to convert the weights.

  Args:
    weights: list of weights for the individual kernels and recurrent kernels.
    biases: list of biases for individual gate.
    shape: the shape for the converted variables that will be feed to CuDNN.
    transpose_weights: boolean, whether to transpose the weights.

  Returns:
    The converted weights that can be feed to CuDNN ops as param.
  """
  def convert(w):
    return array_ops.transpose(w) if transpose_weights else w

  weights = [array_ops.reshape(convert(x), shape) for x in weights]
  biases = [array_ops.reshape(x, shape) for x in biases]
  return array_ops.concat(weights + biases, axis=0)


def standard_lstm(inputs, init_h, init_c, kernel, recurrent_kernel, bias,
                  mask, time_major, go_backwards, sequence_lengths,
                  zero_output_for_mask):
  """LSTM with standard kernel implementation.

  This implementation can be run on all types for hardware.

  This implementation lifts out all the layer weights and make them function
  parameters. It has same number of tensor input params as the CuDNN
  counterpart. The RNN step logic has been simplified, eg dropout and mask is
  removed since CuDNN implementation does not support that.

  Note that the first half of the bias tensor should be ignored by this impl.
  The CuDNN impl need an extra set of input gate bias. In order to make the both
  function take same shape of parameter, that extra set of bias is also feed
  here.

  Args:
    inputs: input tensor of LSTM layer.
    init_h: initial state tensor for the cell output.
    init_c: initial state tensor for the cell hidden state.
    kernel: weights for cell kernel.
    recurrent_kernel: weights for cell recurrent kernel.
    bias: weights for cell kernel bias and recurrent bias. Only recurrent bias
      is used in this case.
    mask: Boolean tensor for mask out the steps within sequence.
      An individual `True` entry indicates that the corresponding timestep
      should be utilized, while a `False` entry indicates that the corresponding
      timestep should be ignored.
    time_major: boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.
    zero_output_for_mask: Boolean, whether to output zero for masked timestep.

  Returns:
    last_output: output tensor for the last timestep, which has shape
      [batch, units].
    outputs: output tensor for all timesteps, which has shape
      [batch, time, units].
    state_0: the cell output, which has same shape as init_h.
    state_1: the cell hidden state, which has same shape as init_c.
    runtime: constant string tensor which indicate real runtime hardware. This
      value is for testing purpose and should be used by user.
  """
  input_shape = backend.int_shape(inputs)
  timesteps = input_shape[0] if time_major else input_shape[1]

  def step(cell_inputs, cell_states):
    """Step function that will be used by Keras RNN backend."""
    h_tm1 = cell_states[0]  # previous memory state
    c_tm1 = cell_states[1]  # previous carry state

    z = backend.dot(cell_inputs, kernel)
    z += backend.dot(h_tm1, recurrent_kernel)
    z = backend.bias_add(z, bias)

    z0, z1, z2, z3 = array_ops.split(z, 4, axis=1)

    i = nn.sigmoid(z0)
    f = nn.sigmoid(z1)
    c = f * c_tm1 + i * nn.tanh(z2)
    o = nn.sigmoid(z3)

    h = o * nn.tanh(c)
    return h, [h, c]

  last_output, outputs, new_states = backend.rnn(
      step,
      inputs, [init_h, init_c],
      constants=None,
      unroll=False,
      time_major=time_major,
      mask=mask,
      go_backwards=go_backwards,
      input_length=(sequence_lengths
                    if sequence_lengths is not None else timesteps),
      zero_output_for_mask=zero_output_for_mask)
  return (last_output, outputs, new_states[0], new_states[1],
          _runtime(_RUNTIME_CPU))


def gpu_lstm(inputs, init_h, init_c, kernel, recurrent_kernel, bias, mask,
             time_major, go_backwards, sequence_lengths):
  """LSTM with either CuDNN or ROCm implementation which is only available for GPU.

  Note that currently only right padded data is supported, or the result will be
  polluted by the unmasked data which should be filtered.

  Args:
    inputs: Input tensor of LSTM layer.
    init_h: Initial state tensor for the cell output.
    init_c: Initial state tensor for the cell hidden state.
    kernel: Weights for cell kernel.
    recurrent_kernel: Weights for cell recurrent kernel.
    bias: Weights for cell kernel bias and recurrent bias. Only recurrent bias
      is used in this case.
    mask: Boolean tensor for mask out the steps within sequence.
      An individual `True` entry indicates that the corresponding timestep
      should be utilized, while a `False` entry indicates that the corresponding
      timestep should be ignored.
    time_major: Boolean, whether the inputs are in the format of [time, batch,
      feature] or [batch, time, feature].
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.

  Returns:
    last_output: Output tensor for the last timestep, which has shape
      [batch, units].
    outputs: Output tensor for all timesteps, which has shape
      [batch, time, units].
    state_0: The cell output, which has same shape as init_h.
    state_1: The cell hidden state, which has same shape as init_c.
    runtime: Constant string tensor which indicate real runtime hardware. This
      value is for testing purpose and should not be used by user.
  """
  if not time_major and mask is None:
    inputs = array_ops.transpose(inputs, perm=(1, 0, 2))
    seq_axis, batch_axis = (0, 1)
  else:
    seq_axis, batch_axis = (0, 1) if time_major else (1, 0)
  # For init_h and init_c, cuDNN expects one more dim of num_layers before or
  # after batch dim for time major or batch major inputs respectively
  init_h = array_ops.expand_dims(init_h, axis=seq_axis)
  init_c = array_ops.expand_dims(init_c, axis=seq_axis)

  weights = array_ops.split(kernel, 4, axis=1)
  weights += array_ops.split(recurrent_kernel, 4, axis=1)
  # CuDNN has an extra set of bias for inputs, we disable them (setting to 0),
  # so that mathematically it is same as the canonical LSTM implementation.
  full_bias = array_ops.concat((array_ops.zeros_like(bias), bias), 0)

  if sysconfig.get_build_info()['is_rocm_build']:
    # ROCm MIOpen's weight sequence for LSTM is different from both canonical
    # and Cudnn format
    # MIOpen: [i, f, o, c] Cudnn/Canonical: [i, f, c, o]
    # i is input gate weights.
    # f is forget gate weights.
    # o is output gate weights.
    # c is cell gate weights.
    weights = [weights[x] for x in (0, 1, 3, 2, 4, 5, 7, 6)]
    # full_bias is a tensor of shape (8*n,)
    full_bias = array_ops.split(full_bias, 8, axis=0)
    full_bias = [full_bias[x] for x in (0, 1, 3, 2, 4, 5, 7, 6)]

  params = _canonical_to_params(
      weights=weights,
      biases=array_ops.split(full_bias, 8),
      shape=constant_op.constant([-1]),
      transpose_weights=True)

  if mask is not None:
    sequence_lengths = calculate_sequence_by_mask(mask, time_major)

  if sequence_lengths is not None:
    if go_backwards:
      # Three reversals are required. E.g.,
      # normal input = [1, 2, 3, 0, 0]  # where 0 need to be masked
      # reversed_input_to_cudnn = [3, 2, 1, 0, 0]
      # output_from_cudnn = [6, 5, 4, 0, 0]
      # expected_output = [0, 0, 6, 5 ,4]
      inputs = array_ops.reverse_sequence_v2(
          inputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
    outputs, h, c, _, _ = gen_cudnn_rnn_ops.CudnnRNNV3(
        input=inputs,
        input_h=init_h,
        input_c=init_c,
        params=params,
        is_training=True,
        rnn_mode='lstm',
        sequence_lengths=sequence_lengths,
        time_major=time_major)
    if go_backwards:
      outputs = array_ops.reverse_sequence_v2(
          outputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
      outputs = array_ops.reverse(outputs, axis=[seq_axis])
  else:
    # # Fill the array with shape [batch] with value of max timesteps.
    # sequence_length = array_ops.fill([array_ops.shape(inputs)[1]],
    #                                  array_ops.shape(inputs)[0])
    if go_backwards:
      # Reverse axis 0 since the input is already convert to time major.
      inputs = array_ops.reverse(inputs, axis=[0])
    outputs, h, c, _ = gen_cudnn_rnn_ops.CudnnRNN(
        input=inputs, input_h=init_h, input_c=init_c, params=params,
        is_training=True, rnn_mode='lstm')

  last_output = outputs[-1]
  if not time_major and mask is None:
    outputs = array_ops.transpose(outputs, perm=[1, 0, 2])
  h = array_ops.squeeze(h, axis=seq_axis)
  c = array_ops.squeeze(c, axis=seq_axis)

  # In the case of variable length input, the cudnn kernel will fill zeros for
  # the output, whereas the default keras behavior is to bring over the previous
  # output for t-1, so that in the return_sequence=False case, user can quickly
  # get the final effect output instead just 0s at the last timestep.
  # In order to mimic the default keras behavior, we copy the final h state as
  # the last_output, since it is numerically same as the output.
  if mask is not None:
    last_output = h
  return last_output, outputs, h, c, _runtime(_RUNTIME_GPU)


def lstm_with_backend_selection(inputs, init_h, init_c, kernel,
                                recurrent_kernel, bias, mask, time_major,
                                go_backwards, sequence_lengths,
                                zero_output_for_mask):
  """Call the LSTM with optimized backend kernel selection.

  Under the hood, this function will create two TF function, one with the most
  generic kernel and can run on all device condition, and the second one with
  CuDNN specific kernel, which can only run on GPU.

  The first function will be called with normal_lstm_params, while the second
  function is not called, but only registered in the graph. The Grappler will
  do the proper graph rewrite and swap the optimized TF function based on the
  device placement.

  Args:
    inputs: Input tensor of LSTM layer.
    init_h: Initial state tensor for the cell output.
    init_c: Initial state tensor for the cell hidden state.
    kernel: Weights for cell kernel.
    recurrent_kernel: Weights for cell recurrent kernel.
    bias: Weights for cell kernel bias and recurrent bias. Only recurrent bias
      is used in this case.
    mask: Boolean tensor for mask out the steps within sequence.
      An individual `True` entry indicates that the corresponding timestep
      should be utilized, while a `False` entry indicates that the corresponding
      timestep should be ignored.
    time_major: Boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.
    zero_output_for_mask: Boolean, whether to output zero for masked timestep.

  Returns:
    List of output tensors, same as standard_lstm.
  """
  params = {
      'inputs': inputs,
      'init_h': init_h,
      'init_c': init_c,
      'kernel': kernel,
      'recurrent_kernel': recurrent_kernel,
      'bias': bias,
      'mask': mask,
      'time_major': time_major,
      'go_backwards': go_backwards,
      'sequence_lengths': sequence_lengths,
      'zero_output_for_mask': zero_output_for_mask,
  }

  def gpu_lstm_with_fallback(inputs, init_h, init_c, kernel, recurrent_kernel,
                             bias, mask, time_major, go_backwards,
                             sequence_lengths, zero_output_for_mask):
    """Use CuDNN kernel when mask is none or strictly right padded."""
    if mask is None:
      return gpu_lstm(
          inputs=inputs,
          init_h=init_h,
          init_c=init_c,
          kernel=kernel,
          recurrent_kernel=recurrent_kernel,
          bias=bias,
          mask=mask,
          time_major=time_major,
          go_backwards=go_backwards,
          sequence_lengths=sequence_lengths)

    def cudnn_lstm_fn():
      return gpu_lstm(
          inputs=inputs,
          init_h=init_h,
          init_c=init_c,
          kernel=kernel,
          recurrent_kernel=recurrent_kernel,
          bias=bias,
          mask=mask,
          time_major=time_major,
          go_backwards=go_backwards,
          sequence_lengths=sequence_lengths)

    def stardard_lstm_fn():
      return standard_lstm(
          inputs=inputs,
          init_h=init_h,
          init_c=init_c,
          kernel=kernel,
          recurrent_kernel=recurrent_kernel,
          bias=bias,
          mask=mask,
          time_major=time_major,
          go_backwards=go_backwards,
          sequence_lengths=sequence_lengths,
          zero_output_for_mask=zero_output_for_mask)

    return control_flow_ops.cond(
        is_cudnn_supported_inputs(mask, time_major),
        true_fn=cudnn_lstm_fn,
        false_fn=stardard_lstm_fn)

  if _use_new_code():
    # Chooses the implementation dynamically based on the running device.
    (last_output, outputs, new_h, new_c,
     runtime) = control_flow_ops.execute_fn_for_device(
         {
             _CPU_DEVICE_NAME: lambda: standard_lstm(**params),
             _GPU_DEVICE_NAME: lambda: gpu_lstm_with_fallback(**params)
         }, lambda: standard_lstm(**params))
  else:
    # Each time a `tf.function` is called, we will give it a unique
    # identifiable API name, so that Grappler won't get confused when it
    # sees multiple LSTM layers added into same graph, and it will be able
    # to pair up the different implementations across them.
    api_name = 'lstm_' + str(uuid.uuid4())
    supportive_attribute = {
        'time_major': time_major,
        'go_backwards': go_backwards,
    }
    defun_standard_lstm = _generate_defun_backend(api_name, _CPU_DEVICE_NAME,
                                                  standard_lstm,
                                                  supportive_attribute)
    defun_gpu_lstm = _generate_defun_backend(api_name, _GPU_DEVICE_NAME,
                                             gpu_lstm_with_fallback,
                                             supportive_attribute)

    # Call the normal LSTM impl and register the CuDNN impl function. The
    # grappler will kick in during session execution to optimize the graph.
    last_output, outputs, new_h, new_c, runtime = defun_standard_lstm(**params)
    _function_register(defun_gpu_lstm, **params)

  return last_output, outputs, new_h, new_c, runtime


def is_sequence_right_padded(mask):
  """Check the mask tensor and see if it right padded.

  For CuDNN kernel, it uses the sequence length param to skip the tailing
  timestep. If the data is left padded, or not a strict right padding (has
  masked value in the middle of the sequence), then CuDNN kernel won't be work
  properly in those cases.

  Left padded data: [[False, False, True, True, True]].
  Right padded data: [[True, True, True, False, False]].
  Mixture of mask/unmasked data: [[True, False, True, False, False]].

  Note that for the mixed data example above, the actually data RNN should see
  are those 2 Trues (index 0 and 2), the index 1 False should be ignored and not
  pollute the internal states.

  Args:
    mask: the Boolean tensor with shape [batch, timestep]

  Returns:
    boolean scalar tensor, whether the mask is strictly right padded.
  """
  max_seq_length = array_ops.shape(mask)[1]
  count_of_true = math_ops.reduce_sum(math_ops.cast(mask, dtypes.int32), axis=1)
  right_padded_mask = array_ops.sequence_mask(
      count_of_true, maxlen=max_seq_length)
  return math_ops.reduce_all(math_ops.equal(mask, right_padded_mask))


def has_fully_masked_sequence(mask):
  # See https://github.com/tensorflow/tensorflow/issues/33148 for more details.
  # Cudnn kernel will error out if the input sequence contains any fully masked
  # data. We walk around this issue by rerouting the computation to standard
  # kernel, until the issue on cudnn side has been fixed.
  # For a fully masked sequence, it will contain all Falses. To make it easy to
  # check, we inverse the boolean, check if any of the sequence has all True.
  return math_ops.reduce_any(
      math_ops.reduce_all(
          math_ops.logical_not(mask),
          axis=1))


def is_cudnn_supported_inputs(mask, time_major):
  if time_major:
    mask = array_ops.transpose(mask)

  return math_ops.logical_and(
      is_sequence_right_padded(mask),
      math_ops.logical_not(has_fully_masked_sequence(mask)))


def calculate_sequence_by_mask(mask, time_major):
  """Calculate the sequence length tensor (1-D) based on the masking tensor.

  The masking tensor is a 2D boolean tensor with shape [batch, timestep]. For
  any timestep that should be masked, the corresponding field will be False.
  Consider the following example:
    a = [[True, True, False, False],
         [True, True, True, False]]
  It is a (2, 4) tensor, and the corresponding sequence length result should be
  1D tensor with value [2, 3]. Note that the masking tensor must be right
  padded that could be checked by, e.g., `is_sequence_right_padded()`.

  Args:
    mask: Boolean tensor with shape [batch, timestep] or [timestep, batch] if
      time_major=True.
    time_major: Boolean, which indicates whether the mask is time major or batch
      major.
  Returns:
    sequence_length: 1D int32 tensor.
  """
  timestep_index = 0 if time_major else 1
  return math_ops.reduce_sum(math_ops.cast(mask, dtypes.int32),
                             axis=timestep_index)


def _generate_defun_backend(unique_api_name, preferred_device, func,
                            supportive_attributes):
  function_attributes = {
      _FUNCTION_API_NAME_ATTRIBUTE: unique_api_name,
      _FUNCTION_DEVICE_ATTRIBUTE: preferred_device,
  }
  function_attributes.update(supportive_attributes)
  return function.defun_with_attributes(func=func,
                                        attributes=function_attributes,
                                        autograph=False)


def _get_context_device_type():
  """Parse the current context and return the device type, eg CPU/GPU."""
  current_device = get_device_name()
  if current_device is None:
    return None
  return device.DeviceSpec.from_string(current_device).device_type


def _runtime(runtime_name):
  with ops.device('/cpu:0'):
    return constant_op.constant(
        runtime_name, dtype=dtypes.float32, name='runtime')


def _read_variable_value(v):
  """Read the value of a variable if it is variable."""
  if isinstance(v, variables.Variable):
    return v.read_value()
  return v


def _function_register(func, *args, **kwargs):
  """Register a specialization of a `Function` into the graph.

  This won't actually call the function with the inputs, and only put the
  function definition into graph. Register function with different input param
  will result into multiple version of functions registered in graph.

  Args:
    func: the `Function` instance that generated by a @defun
    *args: input arguments for the Python function.
    **kwargs: input keyword arguments for the Python function.

  Returns:
    a `ConcreteFunction` object specialized to inputs and execution context.

  Raises:
    ValueError: When the input function is not a defun wrapped python function.
  """
  concrete_func = func.get_concrete_function(*args, **kwargs)
  concrete_func.add_to_graph()
  concrete_func.add_gradient_functions_to_graph()
  return concrete_func
