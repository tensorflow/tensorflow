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
"""Recurrent layers for TF 2.0.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers import recurrent
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.util.tf_export import keras_export


# The following string constants are used by Defun approach for unified backend
# of LSTM and GRU.
_DEFUN_API_NAME_ATTRIBUTE = 'api_implements'
_DEFUN_DEVICE_ATTRIBUTE = 'api_preferred_device'
_CPU_DEVICE_NAME = 'CPU'
_GPU_DEVICE_NAME = 'GPU'


@keras_export('keras.layers.GRU', v1=[])
class GRU(recurrent.DropoutRNNCellMixin, recurrent.GRU):
  """Gated Recurrent Unit - Cho et al. 2014.

  Based on available runtime hardware and constraints, this layer
  will choose different implementations (cuDNN-based or pure-TensorFlow)
  to maximize the performance. If a GPU is available and all
  the arguments to the layer meet the requirement of the CuDNN kernel
  (see below for details), the layer will use a fast cuDNN implementation.

  The requirements to use the cuDNN implementation are:

  1. `activation` == 'tanh'
  2. `recurrent_activation` == 'sigmoid'
  3. `recurrent_dropout` == 0
  4. `unroll` is False
  5. `use_bias` is True
  6. `reset_after` is True
  7. No use of masking.

  There are two variants of the GRU implementation. The default one is based on
  [v3](https://arxiv.org/abs/1406.1078v3) and has reset gate applied to hidden
  state before matrix multiplication. The other one is based on
  [original](https://arxiv.org/abs/1406.1078v1) and has the order reversed.

  The second variant is compatible with CuDNNGRU (GPU-only) and allows
  inference on CPU. Thus it has separate biases for `kernel` and
  `recurrent_kernel`. To use this variant, set `'reset_after'=True` and
  `recurrent_activation='sigmoid'`.

  Arguments:
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
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs.
    recurrent_initializer: Initializer for the `recurrent_kernel`
       weights matrix,
       used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")..
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    recurrent_constraint: Constraint function applied to
      the `recurrent_kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.
    implementation: Implementation mode, either 1 or 2.
      Mode 1 will structure its operations as a larger number of
      smaller dot products and additions, whereas mode 2 will
      batch them into fewer, larger operations. These modes will
      have different performance profiles on different hardware and
      for different applications.
    return_sequences: Boolean. Whether to return the last output
      in the output sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state
      in addition to the output.
    go_backwards: Boolean (default False).
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
    reset_after: GRU convention (whether to apply reset gate after or
      before matrix multiplication). False = "before",
      True = "after" (default and CuDNN compatible).

  Call arguments:
    inputs: A 3D tensor.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
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
               implementation=1,
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
        implementation=implementation,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        time_major=time_major,
        reset_after=reset_after,
        **kwargs)
    # CuDNN uses following setting by default and not configurable.
    self.could_use_cudnn = (
        activation == 'tanh' and recurrent_activation == 'sigmoid' and
        recurrent_dropout == 0 and not unroll and use_bias and
        reset_after)

  def call(self, inputs, mask=None, training=None, initial_state=None):
    # GRU does not support constants. Ignore it during process.
    inputs, initial_state, _ = self._process_inputs(inputs, initial_state, None)

    if isinstance(mask, list):
      mask = mask[0]

    input_shape = K.int_shape(inputs)
    timesteps = input_shape[0] if self.time_major else input_shape[1]

    if mask is not None or not self.could_use_cudnn:
      # CuDNN does not support masking, fall back to use the normal GRU.
      kwargs = {'training': training}

      def step(cell_inputs, cell_states):
        return self.cell.call(cell_inputs, cell_states, **kwargs)

      last_output, outputs, states = K.rnn(
          step,
          inputs,
          initial_state,
          constants=None,
          go_backwards=self.go_backwards,
          mask=mask,
          unroll=self.unroll,
          input_length=timesteps,
          time_major=self.time_major,
          zero_output_for_mask=self.zero_output_for_mask)
      # This is a dummy tensor for testing purpose.
      runtime = _runtime('unknown')
    else:
      last_output, outputs, runtime, states = self._defun_gru_call(
          inputs, initial_state, training)

    if self.stateful:
      updates = [state_ops.assign(self.states[0], states[0])]
      self.add_update(updates, inputs)

    if self.return_sequences:
      output = outputs
    else:
      output = last_output

    if self.return_state:
      return [output] + list(states)
    elif self._return_runtime:
      return output, runtime
    else:
      return output

  def _defun_gru_call(self, inputs, initial_state, training):
    # Use the new defun approach for backend implementation swap.
    # Note that different implementations need to have same function
    # signature, eg, the tensor parameters need to have same shape and dtypes.
    if self.go_backwards:
      # Reverse time axis.
      inputs = K.reverse(inputs, 0 if self.time_major else 1)

    self.reset_dropout_mask()
    dropout_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
    if dropout_mask is not None:
      inputs *= dropout_mask[0]
    if context.executing_eagerly():
      device_type = _get_context_device_type()
      if device_type == _GPU_DEVICE_NAME or (
          device_type is None and context.num_gpus() > 0):
        # Under eager context, check the device placement and prefer the
        # GPU implementation when GPU is available.
        last_output, outputs, new_h, runtime = cudnn_gru(
            inputs=inputs,
            init_h=initial_state[0],
            kernel=self.cell.kernel,
            recurrent_kernel=self.cell.recurrent_kernel,
            bias=self.cell.bias,
            time_major=self.time_major)
      else:
        last_output, outputs, new_h, runtime = standard_gru(
            inputs=inputs,
            init_h=initial_state[0],
            kernel=self.cell.kernel,
            recurrent_kernel=self.cell.recurrent_kernel,
            bias=self.cell.bias,
            activation=self.activation,
            recurrent_activation=self.recurrent_activation,
            time_major=self.time_major)
    else:
      api_name = 'gru_' + str(uuid.uuid4())
      defun_standard_gru = _generate_defun_backend(
          api_name, _CPU_DEVICE_NAME, standard_gru)
      defun_cudnn_gru = _generate_defun_backend(
          api_name, _GPU_DEVICE_NAME, cudnn_gru)
      # Call the normal GRU impl and register the CuDNN impl function. The
      # grappler will kick in during session execution to optimize the graph.
      last_output, outputs, new_h, runtime = defun_standard_gru(
          inputs=inputs,
          init_h=initial_state[0],
          kernel=self.cell.kernel,
          recurrent_kernel=self.cell.recurrent_kernel,
          bias=self.cell.bias,
          activation=self.activation,
          recurrent_activation=self.recurrent_activation,
          time_major=self.time_major)

      function.register(defun_cudnn_gru, inputs, initial_state[0],
                        self.cell.kernel, self.cell.recurrent_kernel,
                        self.cell.bias, self.time_major)
    states = [new_h]
    return last_output, outputs, runtime, states


def standard_gru(inputs, init_h, kernel, recurrent_kernel, bias, activation,
                 recurrent_activation, time_major):
  """GRU with standard kernel implementation.

  This implementation can be run on all types of hardware.

  This implementation lifts out all the layer weights and make them function
  parameters. It has same number of tensor input params as the CuDNN
  counterpart. The RNN step logic has been simplified, eg dropout and mask is
  removed since CuDNN implementation does not support that.

  Arguments:
    inputs: input tensor of GRU layer.
    init_h: initial state tensor for the cell output.
    kernel: weights for cell kernel.
    recurrent_kernel: weights for cell recurrent kernel.
    bias: weights for cell kernel bias and recurrent bias. The bias contains the
      combined input_bias and recurrent_bias.
    activation: Activation function to use for output.
    recurrent_activation: Activation function to use for hidden recurrent state.
    time_major: boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].

  Returns:
    last_output: output tensor for the last timestep, which has shape
      [batch, units].
    outputs: output tensor for all timesteps, which has shape
      [batch, time, units].
    state_0: the cell output, which has same shape as init_h.
    runtime: constant string tensor which indicate real runtime hardware. This
      value is for testing purpose and should be used by user.
  """
  input_shape = K.int_shape(inputs)
  timesteps = input_shape[0] if time_major else input_shape[1]

  input_bias, recurrent_bias = array_ops.unstack(bias)

  def step(cell_inputs, cell_states):
    """Step function that will be used by Keras RNN backend."""
    h_tm1 = cell_states[0]

    # inputs projected by all gate matrices at once
    matrix_x = K.dot(cell_inputs, kernel)
    matrix_x = K.bias_add(matrix_x, input_bias)

    x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=1)

    # hidden state projected by all gate matrices at once
    matrix_inner = K.dot(h_tm1, recurrent_kernel)
    matrix_inner = K.bias_add(matrix_inner, recurrent_bias)

    recurrent_z, recurrent_r, recurrent_h = array_ops.split(matrix_inner, 3,
                                                            axis=1)
    z = recurrent_activation(x_z + recurrent_z)
    r = recurrent_activation(x_r + recurrent_r)
    hh = activation(x_h + r * recurrent_h)

    # previous and candidate state mixed by update gate
    h = z * h_tm1 + (1 - z) * hh
    return h, [h]

  last_output, outputs, new_states = K.rnn(
      step,
      inputs, [init_h],
      constants=None,
      unroll=False,
      time_major=time_major,
      input_length=timesteps)
  return last_output, outputs, new_states[0], _runtime('cpu')


def cudnn_gru(inputs, init_h, kernel, recurrent_kernel, bias, time_major):
  """GRU with CuDNN implementation which is only available for GPU."""
  if not time_major:
    inputs = array_ops.transpose(inputs, perm=(1, 0, 2))
  init_h = array_ops.expand_dims(init_h, axis=0)

  weights = array_ops.split(kernel, 3, axis=1)
  weights += array_ops.split(recurrent_kernel, 3, axis=1)
  # Note that the bias was initialized as shape (2, 3 * units), flat it into
  # (6 * units)
  bias = array_ops.split(K.flatten(bias), 6)
  # Note that the gate order for CuDNN is different from the canonical format.
  # canonical format is [z, r, h], whereas CuDNN is [r, z, h]. The swap need to
  # be done for kernel, recurrent_kernel, input_bias, recurrent_bias.
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

  outputs, h, _, _ = gen_cudnn_rnn_ops.cudnn_rnn(
      inputs,
      input_h=init_h,
      input_c=0,
      params=params,
      is_training=True,
      rnn_mode='gru')
  last_output = outputs[-1]
  if not time_major:
    outputs = array_ops.transpose(outputs, perm=[1, 0, 2])
  h = h[0]
  return last_output, outputs, h, _runtime('cudnn')


@keras_export('keras.layers.LSTM', v1=[])
class LSTM(recurrent.DropoutRNNCellMixin, recurrent.LSTM):
  """Long Short-Term Memory layer - Hochreiter 1997.

  Based on available runtime hardware and constraints, this layer
  will choose different implementations (cuDNN-based or pure-TensorFlow)
  to maximize the performance. If a GPU is available and all
  the arguments to the layer meet the requirement of the CuDNN kernel
  (see below for details), the layer will use a fast cuDNN implementation.

  The requirements to use the cuDNN implementation are:

  1. `activation` == 'tanh'
  2. `recurrent_activation` == 'sigmoid'
  3. `recurrent_dropout` == 0
  4. `unroll` is False
  5. `use_bias` is True
  7. No use of masking.

  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`). If you pass `None`, no activation
      is applied (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use for the recurrent step.
      Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
      applied (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix, used for
      the linear transformation of the inputs..
    recurrent_initializer: Initializer for the `recurrent_kernel` weights
      matrix, used for the linear transformation of the recurrent state..
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate at
      initialization. Setting it to true will also force
      `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
          al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix.
    recurrent_regularizer: Regularizer function applied to the
      `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation")..
    kernel_constraint: Constraint function applied to the `kernel` weights
      matrix.
    recurrent_constraint: Constraint function applied to the `recurrent_kernel`
      weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1. Fraction of the units to drop for the linear
      transformation of the inputs.
    recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
      the linear transformation of the recurrent state.
    implementation: Implementation mode, either 1 or 2. Mode 1 will structure
      its operations as a larger number of smaller dot products and additions,
      whereas mode 2 will batch them into fewer, larger operations. These modes
      will have different performance profiles on different hardware and for
      different applications.
    return_sequences: Boolean. Whether to return the last output. in the output
      sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state in addition to the
      output.
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    stateful: Boolean (default False). If True, the last state for each sample
      at index i in a batch will be used as initial state for the sample of
      index i in the following batch.
    unroll: Boolean (default False). If True, the network will be unrolled, else
      a symbolic loop will be used. Unrolling can speed-up a RNN, although it
      tends to be more memory-intensive. Unrolling is only suitable for short
      sequences.

  Call arguments:
    inputs: A 3D tensor.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
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
               implementation=1,
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
        implementation=implementation,
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
    self.could_use_cudnn = (
        activation == 'tanh' and recurrent_activation == 'sigmoid' and
        recurrent_dropout == 0 and not unroll and use_bias)

  def call(self, inputs, mask=None, training=None, initial_state=None):
    # LSTM does not support constants. Ignore it during process.
    inputs, initial_state, _ = self._process_inputs(inputs, initial_state, None)

    if isinstance(mask, list):
      mask = mask[0]

    input_shape = K.int_shape(inputs)
    timesteps = input_shape[0] if self.time_major else input_shape[1]

    if mask is not None or not self.could_use_cudnn:
      # CuDNN does not support masking, fall back to use the normal LSTM.
      kwargs = {'training': training}

      def step(inputs, states):
        return self.cell.call(inputs, states, **kwargs)

      last_output, outputs, states = K.rnn(
          step,
          inputs,
          initial_state,
          constants=None,
          go_backwards=self.go_backwards,
          mask=mask,
          unroll=self.unroll,
          input_length=timesteps,
          time_major=self.time_major,
          zero_output_for_mask=self.zero_output_for_mask)
      runtime = _runtime('unknown')
    else:
      # Use the new defun approach for backend implementation swap.
      # Note that different implementations need to have same function
      # signature, eg, the tensor parameters need to have same shape and dtypes.
      # Since the CuDNN has an extra set of bias, those bias will be passed to
      # both normal and CuDNN implementations.
      if self.go_backwards:
        # Reverse time axis.
        inputs = K.reverse(inputs, 0 if self.time_major else 1)

      self.reset_dropout_mask()
      dropout_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
      if dropout_mask is not None:
        inputs *= dropout_mask[0]

      if context.executing_eagerly():
        device_type = _get_context_device_type()
        if device_type == _GPU_DEVICE_NAME or (
            device_type is None and context.num_gpus() > 0):
          # Under eager context, check the device placement and prefer the
          # GPU implementation when GPU is available.
          last_output, outputs, new_h, new_c, runtime = cudnn_lstm(
              inputs, initial_state[0], initial_state[1], self.cell.kernel,
              self.cell.recurrent_kernel, self.cell.bias, self.time_major)
        else:
          last_output, outputs, new_h, new_c, runtime = standard_lstm(
              inputs, initial_state[0], initial_state[1], self.cell.kernel,
              self.cell.recurrent_kernel, self.cell.bias, self.activation,
              self.recurrent_activation, self.time_major)
      else:
        # Each time a `tf.function` is called, we will give it a unique
        # identifiable API name, so that Grappler won't get confused when it
        # sees multiple LSTM layers added into same graph, and it will be able
        # to pair up the different implementations across them.
        api_name = 'lstm_' + str(uuid.uuid4())
        defun_standard_lstm = _generate_defun_backend(
            api_name, _CPU_DEVICE_NAME, standard_lstm)
        defun_cudnn_lstm = _generate_defun_backend(
            api_name, _GPU_DEVICE_NAME, cudnn_lstm)

        # Call the normal LSTM impl and register the CuDNN impl function. The
        # grappler will kick in during session execution to optimize the graph.
        last_output, outputs, new_h, new_c, runtime = defun_standard_lstm(
            inputs, initial_state[0], initial_state[1], self.cell.kernel,
            self.cell.recurrent_kernel, self.cell.bias, self.activation,
            self.recurrent_activation, self.time_major)

        function.register(defun_cudnn_lstm, inputs, initial_state[0],
                          initial_state[1], self.cell.kernel,
                          self.cell.recurrent_kernel, self.cell.bias,
                          self.time_major)
      states = [new_h, new_c]

    if self.stateful:
      updates = []
      for i in range(len(states)):
        updates.append(state_ops.assign(self.states[i], states[i]))
      self.add_update(updates, inputs)

    if self.return_sequences:
      output = outputs
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
                  activation, recurrent_activation, time_major):
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
    activation: Activation function to use for output.
    recurrent_activation: Activation function to use for hidden recurrent state.
    time_major: boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].

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
  input_shape = K.int_shape(inputs)
  timesteps = input_shape[0] if time_major else input_shape[1]

  def step(cell_inputs, cell_states):
    """Step function that will be used by Keras RNN backend."""
    h_tm1 = cell_states[0]  # previous memory state
    c_tm1 = cell_states[1]  # previous carry state

    z = K.dot(cell_inputs, kernel)
    z += K.dot(h_tm1, recurrent_kernel)
    z = K.bias_add(z, bias)

    z0, z1, z2, z3 = array_ops.split(z, 4, axis=1)

    i = recurrent_activation(z0)
    f = recurrent_activation(z1)
    c = f * c_tm1 + i * activation(z2)
    o = recurrent_activation(z3)

    h = o * activation(c)
    return h, [h, c]

  last_output, outputs, new_states = K.rnn(
      step,
      inputs, [init_h, init_c],
      constants=None,
      unroll=False,
      time_major=time_major,
      input_length=timesteps)
  return last_output, outputs, new_states[0], new_states[1], _runtime('cpu')


def cudnn_lstm(inputs, input_h, input_c, kernel, recurrent_kernel, bias,
               time_major):
  """LSTM with CuDNN implementation which is only available for GPU."""
  if not time_major:
    inputs = array_ops.transpose(inputs, perm=(1, 0, 2))
  input_h = array_ops.expand_dims(input_h, axis=0)
  input_c = array_ops.expand_dims(input_c, axis=0)

  weights = array_ops.split(kernel, 4, axis=1)
  weights += array_ops.split(recurrent_kernel, 4, axis=1)
  # CuDNN has an extra set of bias for inputs, we disable them (setting to 0),
  # so that mathematically it is same as the canonical LSTM implementation.
  full_bias = array_ops.concat((array_ops.zeros_like(bias), bias), 0)

  params = _canonical_to_params(
      weights=weights,
      biases=array_ops.split(full_bias, 8),
      shape=constant_op.constant([-1]),
      transpose_weights=True)

  outputs, h, c, _ = gen_cudnn_rnn_ops.cudnn_rnn(
      inputs, input_h=input_h, input_c=input_c, params=params, is_training=True)
  last_output = outputs[-1]
  if not time_major:
    outputs = array_ops.transpose(outputs, perm=[1, 0, 2])
  h = h[0]
  c = c[0]

  return last_output, outputs, h, c, _runtime('cudnn')


def _generate_defun_backend(unique_api_name, preferred_device, func):
  function_attributes = {
      _DEFUN_API_NAME_ATTRIBUTE: unique_api_name,
      _DEFUN_DEVICE_ATTRIBUTE: preferred_device,
  }
  return function.defun_with_attributes(func=func,
                                        attributes=function_attributes)


def _get_context_device_type():
  """Parse the current context and return the device type, eg CPU/GPU."""
  current_device = context.context().device_name
  if current_device is None:
    return None
  return device.DeviceSpec.from_string(current_device).device_type


def _runtime(runtime_name):
  with ops.device('/cpu:0'):
    return constant_op.constant(
        runtime_name, dtype=dtypes.string, name='runtime')
