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
# pylint: disable=protected-access
"""Recurrent layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.keras.python.keras import activations
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras import constraints
from tensorflow.contrib.keras.python.keras import initializers
from tensorflow.contrib.keras.python.keras import regularizers
from tensorflow.contrib.keras.python.keras.engine import InputSpec
from tensorflow.contrib.keras.python.keras.engine import Layer
from tensorflow.python.framework import tensor_shape


# pylint: disable=access-member-before-definition


def _time_distributed_dense(x,
                            w,
                            b=None,
                            dropout=None,
                            input_dim=None,
                            output_dim=None,
                            timesteps=None,
                            training=None):
  """Apply `y . w + b` for every temporal slice y of x.

  Arguments:
      x: input tensor.
      w: weight matrix.
      b: optional bias vector.
      dropout: wether to apply dropout (same dropout mask
          for every temporal slice of the input).
      input_dim: integer; optional dimensionality of the input.
      output_dim: integer; optional dimensionality of the output.
      timesteps: integer; optional number of timesteps.
      training: training phase tensor or boolean.

  Returns:
      Output tensor.
  """
  if not input_dim:
    input_dim = K.shape(x)[2]
  if not timesteps:
    timesteps = K.shape(x)[1]
  if not output_dim:
    output_dim = K.shape(w)[1]

  if dropout is not None and 0. < dropout < 1.:
    # apply the same dropout pattern at every timestep
    ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
    dropout_matrix = K.dropout(ones, dropout)
    expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
    x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

  # collapse time dimension and batch dimension together
  x = K.reshape(x, (-1, input_dim))
  x = K.dot(x, w)
  if b is not None:
    x = K.bias_add(x, b)
  # reshape to 3D tensor
  if K.backend() == 'tensorflow':
    x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
    x.set_shape([None, None, output_dim])
  else:
    x = K.reshape(x, (-1, timesteps, output_dim))
  return x


class Recurrent(Layer):
  """Abstract base class for recurrent layers.

  Do not use in a model -- it's not a valid layer!
  Use its children classes `LSTM`, `GRU` and `SimpleRNN` instead.

  All recurrent layers (`LSTM`, `GRU`, `SimpleRNN`) also
  follow the specifications of this class and accept
  the keyword arguments listed below.

  Example:

  ```python
      # as the first layer in a Sequential model
      model = Sequential()
      model.add(LSTM(32, input_shape=(10, 64)))
      # now model.output_shape == (None, 32)
      # note: `None` is the batch dimension.

      # for subsequent layers, no need to specify the input size:
      model.add(LSTM(16))

      # to stack recurrent layers, you must use return_sequences=True
      # on any recurrent layer that feeds into another recurrent layer.
      # note that you only need to specify the input size on the first layer.
      model = Sequential()
      model.add(LSTM(64, input_dim=64, input_length=10, return_sequences=True))
      model.add(LSTM(32, return_sequences=True))
      model.add(LSTM(10))
  ```

  Arguments:
      weights: list of Numpy arrays to set as initial weights.
          The list should have 3 elements, of shapes:
          `[(input_dim, output_dim), (output_dim, output_dim), (output_dim,)]`.
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
      implementation: one of {0, 1, or 2}.
          If set to 0, the RNN will use
          an implementation that uses fewer, larger matrix products,
          thus running faster on CPU but consuming more memory.
          If set to 1, the RNN will use more matrix products,
          but smaller ones, thus running slower
          (may actually be faster on GPU) while consuming less memory.
          If set to 2 (LSTM/GRU only),
          the RNN will combine the input gate,
          the forget gate and the output gate into a single matrix,
          enabling more time-efficient parallelization on the GPU.
          Note: RNN dropout must be shared for all gates,
          resulting in a slightly reduced regularization.
      input_dim: dimensionality of the input (integer).
          This argument (or alternatively, the keyword argument `input_shape`)
          is required when using this layer as the first layer in a model.
      input_length: Length of input sequences, to be specified
          when it is constant.
          This argument is required if you are going to connect
          `Flatten` then `Dense` layers upstream
          (without it, the shape of the dense outputs cannot be computed).
          Note that if the recurrent layer is not the first layer
          in your model, you would need to specify the input length
          at the level of the first layer
          (e.g. via the `input_shape` argument)

  Input shape:s
      3D tensor with shape `(batch_size, timesteps, input_dim)`,
      (Optional) 2D tensors with shape `(batch_size, output_dim)`.

  Output shape:
      - if `return_state`: a list of tensors. The first tensor is
          the output. The remaining tensors are the last states,
          each with shape `(batch_size, units)`.
      - if `return_sequences`: 3D tensor with shape
          `(batch_size, timesteps, units)`.
      - else, 2D tensor with shape `(batch_size, units)`.

  # Masking
      This layer supports masking for input data with a variable number
      of timesteps. To introduce masks to your data,
      use an `Embedding` layer with the `mask_zero` parameter
      set to `True`.

  # Note on using statefulness in RNNs
      You can set RNN layers to be 'stateful', which means that the states
      computed for the samples in one batch will be reused as initial states
      for the samples in the next batch. This assumes a one-to-one mapping
      between samples in different successive batches.

      To enable statefulness:
          - specify `stateful=True` in the layer constructor.
          - specify a fixed batch size for your model, by passing
              if sequential model:
                `batch_input_shape=(...)` to the first layer in your model.
              else for functional model with 1 or more Input layers:
                `batch_shape=(...)` to all the first layers in your model.
              This is the expected shape of your inputs
              *including the batch size*.
              It should be a tuple of integers, e.g. `(32, 10, 100)`.
          - specify `shuffle=False` when calling fit().

      To reset the states of your model, call `.reset_states()` on either
      a specific layer, or on your entire model.

  # Note on specifying the initial state of RNNs
      You can specify the initial state of RNN layers symbolically by
      calling them with the keyword argument `initial_state`. The value of
      `initial_state` should be a tensor or list of tensors representing
      the initial state of the RNN layer.

      You can specify the initial state of RNN layers numerically by
      calling `reset_states` with the keyword argument `states`. The value of
      `states` should be a numpy array or list of numpy arrays representing
      the initial state of the RNN layer.
  """

  def __init__(self,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               implementation=0,
               **kwargs):
    super(Recurrent, self).__init__(**kwargs)
    self.return_sequences = return_sequences
    self.return_state = return_state
    self.go_backwards = go_backwards
    self.stateful = stateful
    self.unroll = unroll
    self.implementation = implementation
    self.supports_masking = True
    self.input_spec = [InputSpec(ndim=3)]
    self.state_spec = None
    self.dropout = 0
    self.recurrent_dropout = 0

  def _compute_output_shape(self, input_shape):
    if isinstance(input_shape, list):
      input_shape = input_shape[0]
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.return_sequences:
      output_shape = (input_shape[0], input_shape[1], self.units)
    else:
      output_shape = (input_shape[0], self.units)

    if self.return_state:
      state_shape = [tensor_shape.TensorShape(
          (input_shape[0], self.units)) for _ in self.states]
      return [tensor_shape.TensorShape(output_shape)] + state_shape
    return tensor_shape.TensorShape(output_shape)

  def compute_mask(self, inputs, mask):
    if isinstance(mask, list):
      mask = mask[0]
    output_mask = mask if self.return_sequences else None
    if self.return_state:
      state_mask = [None for _ in self.states]
      return [output_mask] + state_mask
    return output_mask

  def step(self, inputs, states):
    raise NotImplementedError

  def get_constants(self, inputs, training=None):
    return []

  def get_initial_state(self, inputs):
    # build an all-zero tensor of shape (samples, output_dim)
    initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
    initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
    initial_state = K.expand_dims(initial_state)  # (samples, 1)
    initial_state = K.tile(initial_state, [1,
                                           self.units])  # (samples, output_dim)
    initial_state = [initial_state for _ in range(len(self.states))]
    return initial_state

  def preprocess_input(self, inputs, training=None):
    return inputs

  def __call__(self, inputs, initial_state=None, **kwargs):
    if (isinstance(inputs, (list, tuple)) and
        len(inputs) > 1
        and initial_state is None):
      initial_state = inputs[1:]
      inputs = inputs[0]

    # If `initial_state` is specified,
    # and if it a Keras tensor,
    # then add it to the inputs and temporarily
    # modify the input spec to include the state.
    if initial_state is None:
      return super(Recurrent, self).__call__(inputs, **kwargs)

    if not isinstance(initial_state, (list, tuple)):
      initial_state = [initial_state]

    is_keras_tensor = hasattr(initial_state[0], '_keras_history')
    for tensor in initial_state:
      if hasattr(tensor, '_keras_history') != is_keras_tensor:
        raise ValueError('The initial state of an RNN layer cannot be'
                         ' specified with a mix of Keras tensors and'
                         ' non-Keras tensors')

    if is_keras_tensor:
      # Compute the full input spec, including state
      input_spec = self.input_spec
      state_spec = self.state_spec
      if not isinstance(input_spec, list):
        input_spec = [input_spec]
      if not isinstance(state_spec, list):
        state_spec = [state_spec]
      self.input_spec = input_spec + state_spec

      # Compute the full inputs, including state
      inputs = [inputs] + list(initial_state)

      # Perform the call
      output = super(Recurrent, self).__call__(inputs, **kwargs)

      # Restore original input spec
      self.input_spec = input_spec
      return output
    else:
      kwargs['initial_state'] = initial_state
      return super(Recurrent, self).__call__(inputs, **kwargs)

  def call(self, inputs, mask=None, training=None, initial_state=None):
    # input shape: `(samples, time (padded with zeros), input_dim)`
    # note that the .build() method of subclasses MUST define
    # self.input_spec and self.state_spec with complete input shapes.
    if isinstance(inputs, list):
      initial_state = inputs[1:]
      inputs = inputs[0]
    elif initial_state is not None:
      pass
    elif self.stateful:
      initial_state = self.states
    else:
      initial_state = self.get_initial_state(inputs)

    if isinstance(mask, list):
      mask = mask[0]

    if len(initial_state) != len(self.states):
      raise ValueError('Layer has ' + str(len(self.states)) +
                       ' states but was passed ' + str(len(initial_state)) +
                       ' initial states.')
    input_shape = K.int_shape(inputs)
    if self.unroll and input_shape[1] is None:
      raise ValueError('Cannot unroll a RNN if the '
                       'time dimension is undefined. \n'
                       '- If using a Sequential model, '
                       'specify the time dimension by passing '
                       'an `input_shape` or `batch_input_shape` '
                       'argument to your first layer. If your '
                       'first layer is an Embedding, you can '
                       'also use the `input_length` argument.\n'
                       '- If using the functional API, specify '
                       'the time dimension by passing a `shape` '
                       'or `batch_shape` argument to your Input layer.')
    constants = self.get_constants(inputs, training=None)
    preprocessed_input = self.preprocess_input(inputs, training=None)
    last_output, outputs, states = K.rnn(
        self.step,
        preprocessed_input,
        initial_state,
        go_backwards=self.go_backwards,
        mask=mask,
        constants=constants,
        unroll=self.unroll)
    if self.stateful:
      updates = []
      for i in range(len(states)):
        updates.append((self.states[i], states[i]))
      self.add_update(updates, inputs)

    # Properly set learning phase
    if 0 < self.dropout + self.recurrent_dropout:
      last_output._uses_learning_phase = True
      outputs._uses_learning_phase = True

    if not self.return_sequences:
      outputs = last_output

    if self.return_state:
      if not isinstance(states, (list, tuple)):
        states = [states]
      else:
        states = list(states)
      return [outputs] + states
    return outputs

  def reset_states(self, states=None):
    if not self.stateful:
      raise AttributeError('Layer must be stateful.')
    batch_size = self.input_spec[0].shape[0]
    if not batch_size:
      raise ValueError('If a RNN is stateful, it needs to know '
                       'its batch size. Specify the batch size '
                       'of your input tensors: \n'
                       '- If using a Sequential model, '
                       'specify the batch size by passing '
                       'a `batch_input_shape` '
                       'argument to your first layer.\n'
                       '- If using the functional API, specify '
                       'the time dimension by passing a '
                       '`batch_shape` argument to your Input layer.')
    # initialize state if None
    if self.states[0] is None:
      self.states = [K.zeros((batch_size, self.units)) for _ in self.states]
    elif states is None:
      for state in self.states:
        K.set_value(state, np.zeros((batch_size, self.units)))
    else:
      if not isinstance(states, (list, tuple)):
        states = [states]
      if len(states) != len(self.states):
        raise ValueError('Layer ' + self.name + ' expects ' +
                         str(len(self.states)) + ' states, '
                         'but it received ' + str(len(states)) +
                         ' state values. Input received: ' + str(states))
      for index, (value, state) in enumerate(zip(states, self.states)):
        if value.shape != (batch_size, self.units):
          raise ValueError('State ' + str(index) +
                           ' is incompatible with layer ' + self.name +
                           ': expected shape=' + str((batch_size, self.units)) +
                           ', found shape=' + str(value.shape))
        K.set_value(state, value)

  def get_config(self):
    config = {
        'return_sequences': self.return_sequences,
        'return_state': self.return_state,
        'go_backwards': self.go_backwards,
        'stateful': self.stateful,
        'unroll': self.unroll,
        'implementation': self.implementation
    }
    base_config = super(Recurrent, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class SimpleRNN(Recurrent):
  """Fully-connected RNN where the output is to be fed back to input.

  Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
          If you pass None, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
          used for the linear transformation of the inputs..
      recurrent_initializer: Initializer for the `recurrent_kernel`
          weights matrix,
          used for the linear transformation of the recurrent state..
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
          Fraction of the units to drop for
          the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the recurrent state.

  References:
      - [A Theoretically Grounded Application of Dropout in Recurrent Neural
        Networks](http://arxiv.org/abs/1512.05287)
  """

  def __init__(self,
               units,
               activation='tanh',
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
               **kwargs):
    super(SimpleRNN, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    self.state_spec = InputSpec(shape=(None, self.units))

  def build(self, input_shape):
    if isinstance(input_shape, list):
      input_shape = input_shape[0]
    input_shape = tensor_shape.TensorShape(input_shape).as_list()

    batch_size = input_shape[0] if self.stateful else None
    self.input_dim = input_shape[2]
    self.input_spec[0] = InputSpec(shape=(batch_size, None, self.input_dim))

    self.states = [None]
    if self.stateful:
      self.reset_states()

    self.kernel = self.add_weight(
        shape=(self.input_dim, self.units),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)
    if self.use_bias:
      self.bias = self.add_weight(
          shape=(self.units,),
          name='bias',
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    self.built = True

  def preprocess_input(self, inputs, training=None):
    if self.implementation > 0:
      return inputs
    else:
      input_shape = inputs.get_shape().as_list()
      input_dim = input_shape[2]
      timesteps = input_shape[1]
      return _time_distributed_dense(
          inputs,
          self.kernel,
          self.bias,
          self.dropout,
          input_dim,
          self.units,
          timesteps,
          training=training)

  def step(self, inputs, states):
    if self.implementation == 0:
      h = inputs
    else:
      if 0 < self.dropout < 1:
        h = K.dot(inputs * states[1], self.kernel)
      else:
        h = K.dot(inputs, self.kernel)
      if self.bias is not None:
        h = K.bias_add(h, self.bias)

    prev_output = states[0]
    if 0 < self.recurrent_dropout < 1:
      prev_output *= states[2]
    output = h + K.dot(prev_output, self.recurrent_kernel)
    if self.activation is not None:
      output = self.activation(output)

    # Properly set learning phase on output tensor.
    if 0 < self.dropout + self.recurrent_dropout:
      output._uses_learning_phase = True
    return output, [output]

  def get_constants(self, inputs, training=None):
    constants = []
    if self.implementation != 0 and 0 < self.dropout < 1:
      input_shape = K.int_shape(inputs)
      input_dim = input_shape[-1]
      ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
      ones = K.tile(ones, (1, int(input_dim)))

      def dropped_inputs():
        return K.dropout(ones, self.dropout)

      dp_mask = K.in_train_phase(dropped_inputs, ones, training=training)
      constants.append(dp_mask)
    else:
      constants.append(K.cast_to_floatx(1.))

    if 0 < self.recurrent_dropout < 1:
      ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
      ones = K.tile(ones, (1, self.units))

      def dropped_inputs():  # pylint: disable=function-redefined
        return K.dropout(ones, self.recurrent_dropout)

      rec_dp_mask = K.in_train_phase(dropped_inputs, ones, training=training)
      constants.append(rec_dp_mask)
    else:
      constants.append(K.cast_to_floatx(1.))
    return constants

  def get_config(self):
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint),
        'dropout': self.dropout,
        'recurrent_dropout': self.recurrent_dropout
    }
    base_config = super(SimpleRNN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class GRU(Recurrent):
  """Gated Recurrent Unit - Cho et al.

  2014.

  Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
          If you pass None, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
          for the recurrent step.
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
          used for the linear transformation of the inputs..
      recurrent_initializer: Initializer for the `recurrent_kernel`
          weights matrix,
          used for the linear transformation of the recurrent state..
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
          Fraction of the units to drop for
          the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the recurrent state.

  References:
      - [On the Properties of Neural Machine Translation: Encoder-Decoder
        Approaches](https://arxiv.org/abs/1409.1259)
      - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
        Modeling](http://arxiv.org/abs/1412.3555v1)
      - [A Theoretically Grounded Application of Dropout in Recurrent Neural
        Networks](http://arxiv.org/abs/1512.05287)
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
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
               **kwargs):
    super(GRU, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    self.state_spec = InputSpec(shape=(None, self.units))

  def build(self, input_shape):
    if isinstance(input_shape, list):
      input_shape = input_shape[0]
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    batch_size = input_shape[0] if self.stateful else None
    self.input_dim = input_shape[2]
    self.input_spec[0] = InputSpec(shape=(batch_size, None, self.input_dim))

    self.states = [None]
    if self.stateful:
      self.reset_states()

    self.kernel = self.add_weight(
        shape=(self.input_dim, self.units * 3),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 3),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

    if self.use_bias:
      self.bias = self.add_weight(
          shape=(self.units * 3,),
          name='bias',
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None

    self.kernel_z = self.kernel[:, :self.units]
    self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
    self.kernel_r = self.kernel[:, self.units:self.units * 2]
    self.recurrent_kernel_r = self.recurrent_kernel[:, self.units:
                                                    self.units * 2]
    self.kernel_h = self.kernel[:, self.units * 2:]
    self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

    if self.use_bias:
      self.bias_z = self.bias[:self.units]
      self.bias_r = self.bias[self.units:self.units * 2]
      self.bias_h = self.bias[self.units * 2:]
    else:
      self.bias_z = None
      self.bias_r = None
      self.bias_h = None
    self.built = True

  def preprocess_input(self, inputs, training=None):
    if self.implementation == 0:
      input_shape = inputs.get_shape().as_list()
      input_dim = input_shape[2]
      timesteps = input_shape[1]

      x_z = _time_distributed_dense(
          inputs,
          self.kernel_z,
          self.bias_z,
          self.dropout,
          input_dim,
          self.units,
          timesteps,
          training=training)
      x_r = _time_distributed_dense(
          inputs,
          self.kernel_r,
          self.bias_r,
          self.dropout,
          input_dim,
          self.units,
          timesteps,
          training=training)
      x_h = _time_distributed_dense(
          inputs,
          self.kernel_h,
          self.bias_h,
          self.dropout,
          input_dim,
          self.units,
          timesteps,
          training=training)
      return K.concatenate([x_z, x_r, x_h], axis=2)
    else:
      return inputs

  def get_constants(self, inputs, training=None):
    constants = []
    if self.implementation != 0 and 0 < self.dropout < 1:
      input_shape = K.int_shape(inputs)
      input_dim = input_shape[-1]
      ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
      ones = K.tile(ones, (1, int(input_dim)))

      def dropped_inputs():
        return K.dropout(ones, self.dropout)

      dp_mask = [
          K.in_train_phase(dropped_inputs, ones, training=training)
          for _ in range(3)
      ]
      constants.append(dp_mask)
    else:
      constants.append([K.cast_to_floatx(1.) for _ in range(3)])

    if 0 < self.recurrent_dropout < 1:
      ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
      ones = K.tile(ones, (1, self.units))

      def dropped_inputs():  # pylint: disable=function-redefined
        return K.dropout(ones, self.recurrent_dropout)

      rec_dp_mask = [
          K.in_train_phase(dropped_inputs, ones, training=training)
          for _ in range(3)
      ]
      constants.append(rec_dp_mask)
    else:
      constants.append([K.cast_to_floatx(1.) for _ in range(3)])
    return constants

  def step(self, inputs, states):
    h_tm1 = states[0]  # previous memory
    dp_mask = states[1]  # dropout matrices for recurrent units
    rec_dp_mask = states[2]

    if self.implementation == 2:
      matrix_x = K.dot(inputs * dp_mask[0], self.kernel)
      if self.use_bias:
        matrix_x = K.bias_add(matrix_x, self.bias)
      matrix_inner = K.dot(h_tm1 * rec_dp_mask[0],
                           self.recurrent_kernel[:, :2 * self.units])

      x_z = matrix_x[:, :self.units]
      x_r = matrix_x[:, self.units:2 * self.units]
      recurrent_z = matrix_inner[:, :self.units]
      recurrent_r = matrix_inner[:, self.units:2 * self.units]

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      x_h = matrix_x[:, 2 * self.units:]
      recurrent_h = K.dot(r * h_tm1 * rec_dp_mask[0],
                          self.recurrent_kernel[:, 2 * self.units:])
      hh = self.activation(x_h + recurrent_h)
    else:
      if self.implementation == 0:
        x_z = inputs[:, :self.units]
        x_r = inputs[:, self.units:2 * self.units]
        x_h = inputs[:, 2 * self.units:]
      elif self.implementation == 1:
        x_z = K.dot(inputs * dp_mask[0], self.kernel_z)
        x_r = K.dot(inputs * dp_mask[1], self.kernel_r)
        x_h = K.dot(inputs * dp_mask[2], self.kernel_h)
        if self.use_bias:
          x_z = K.bias_add(x_z, self.bias_z)
          x_r = K.bias_add(x_r, self.bias_r)
          x_h = K.bias_add(x_h, self.bias_h)
      else:
        raise ValueError('Unknown `implementation` mode.')
      z = self.recurrent_activation(x_z + K.dot(h_tm1 * rec_dp_mask[0],
                                                self.recurrent_kernel_z))
      r = self.recurrent_activation(x_r + K.dot(h_tm1 * rec_dp_mask[1],
                                                self.recurrent_kernel_r))

      hh = self.activation(x_h + K.dot(r * h_tm1 * rec_dp_mask[2],
                                       self.recurrent_kernel_h))
    h = z * h_tm1 + (1 - z) * hh
    if 0 < self.dropout + self.recurrent_dropout:
      h._uses_learning_phase = True
    return h, [h]

  def get_config(self):
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint),
        'dropout': self.dropout,
        'recurrent_dropout': self.recurrent_dropout
    }
    base_config = super(GRU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class LSTM(Recurrent):
  """Long-Short Term Memory unit - Hochreiter 1997.

  For a step-by-step description of the algorithm, see
  [this tutorial](http://deeplearning.net/tutorial/lstm.html).

  Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
          If you pass None, no activation is applied
          (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
          for the recurrent step.
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
          used for the linear transformation of the inputs..
      recurrent_initializer: Initializer for the `recurrent_kernel`
          weights matrix,
          used for the linear transformation of the recurrent state..
      bias_initializer: Initializer for the bias vector.
      unit_forget_bias: Boolean.
          If True, add 1 to the bias of the forget gate at initialization.
          Setting it to true will also force `bias_initializer="zeros"`.
          This is recommended in [Jozefowicz et
            al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
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
          Fraction of the units to drop for
          the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the recurrent state.

  References:
      - [Long short-term
        memory]((http://www.bioinf.jku.at/publications/older/2604.pdf)
        (original 1997 paper)
      - [Supervised sequence labeling with recurrent neural
        networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
      - [A Theoretically Grounded Application of Dropout in Recurrent Neural
        Networks](http://arxiv.org/abs/1512.05287)
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
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
               **kwargs):
    super(LSTM, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.unit_forget_bias = unit_forget_bias

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    self.state_spec = [
        InputSpec(shape=(None, self.units)),
        InputSpec(shape=(None, self.units))
    ]

  def build(self, input_shape):
    if isinstance(input_shape, list):
      input_shape = input_shape[0]
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    batch_size = input_shape[0] if self.stateful else None
    self.input_dim = input_shape[2]
    self.input_spec[0] = InputSpec(shape=(batch_size, None, self.input_dim))

    self.states = [None, None]
    if self.stateful:
      self.reset_states()

    self.kernel = self.add_weight(
        shape=(self.input_dim, self.units * 4),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

    if self.use_bias:
      if self.unit_forget_bias:

        def bias_initializer(_, *args, **kwargs):
          return K.concatenate([
              self.bias_initializer((self.units,), *args, **kwargs),
              initializers.Ones()((self.units,), *args, **kwargs),
              self.bias_initializer((self.units * 2,), *args, **kwargs),
          ])
      else:
        bias_initializer = self.bias_initializer
      self.bias = self.add_weight(
          shape=(self.units * 4,),
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None

    self.kernel_i = self.kernel[:, :self.units]
    self.kernel_f = self.kernel[:, self.units:self.units * 2]
    self.kernel_c = self.kernel[:, self.units * 2:self.units * 3]
    self.kernel_o = self.kernel[:, self.units * 3:]

    self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
    self.recurrent_kernel_f = self.recurrent_kernel[:, self.units:
                                                    self.units * 2]
    self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2:
                                                    self.units * 3]
    self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

    if self.use_bias:
      self.bias_i = self.bias[:self.units]
      self.bias_f = self.bias[self.units:self.units * 2]
      self.bias_c = self.bias[self.units * 2:self.units * 3]
      self.bias_o = self.bias[self.units * 3:]
    else:
      self.bias_i = None
      self.bias_f = None
      self.bias_c = None
      self.bias_o = None
    self.built = True

  def preprocess_input(self, inputs, training=None):
    if self.implementation == 0:
      input_shape = inputs.get_shape().as_list()
      input_dim = input_shape[2]
      timesteps = input_shape[1]

      x_i = _time_distributed_dense(
          inputs,
          self.kernel_i,
          self.bias_i,
          self.dropout,
          input_dim,
          self.units,
          timesteps,
          training=training)
      x_f = _time_distributed_dense(
          inputs,
          self.kernel_f,
          self.bias_f,
          self.dropout,
          input_dim,
          self.units,
          timesteps,
          training=training)
      x_c = _time_distributed_dense(
          inputs,
          self.kernel_c,
          self.bias_c,
          self.dropout,
          input_dim,
          self.units,
          timesteps,
          training=training)
      x_o = _time_distributed_dense(
          inputs,
          self.kernel_o,
          self.bias_o,
          self.dropout,
          input_dim,
          self.units,
          timesteps,
          training=training)
      return K.concatenate([x_i, x_f, x_c, x_o], axis=2)
    else:
      return inputs

  def get_constants(self, inputs, training=None):
    constants = []
    if self.implementation != 0 and 0 < self.dropout < 1:
      input_shape = K.int_shape(inputs)
      input_dim = input_shape[-1]
      ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
      ones = K.tile(ones, (1, int(input_dim)))

      def dropped_inputs():
        return K.dropout(ones, self.dropout)

      dp_mask = [
          K.in_train_phase(dropped_inputs, ones, training=training)
          for _ in range(4)
      ]
      constants.append(dp_mask)
    else:
      constants.append([K.cast_to_floatx(1.) for _ in range(4)])

    if 0 < self.recurrent_dropout < 1:
      ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
      ones = K.tile(ones, (1, self.units))

      def dropped_inputs():  # pylint: disable=function-redefined
        return K.dropout(ones, self.recurrent_dropout)

      rec_dp_mask = [
          K.in_train_phase(dropped_inputs, ones, training=training)
          for _ in range(4)
      ]
      constants.append(rec_dp_mask)
    else:
      constants.append([K.cast_to_floatx(1.) for _ in range(4)])
    return constants

  def step(self, inputs, states):
    h_tm1 = states[0]
    c_tm1 = states[1]
    dp_mask = states[2]
    rec_dp_mask = states[3]

    if self.implementation == 2:
      z = K.dot(inputs * dp_mask[0], self.kernel)
      z += K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel)
      if self.use_bias:
        z = K.bias_add(z, self.bias)

      z0 = z[:, :self.units]
      z1 = z[:, self.units:2 * self.units]
      z2 = z[:, 2 * self.units:3 * self.units]
      z3 = z[:, 3 * self.units:]

      i = self.recurrent_activation(z0)
      f = self.recurrent_activation(z1)
      c = f * c_tm1 + i * self.activation(z2)
      o = self.recurrent_activation(z3)
    else:
      if self.implementation == 0:
        x_i = inputs[:, :self.units]
        x_f = inputs[:, self.units:2 * self.units]
        x_c = inputs[:, 2 * self.units:3 * self.units]
        x_o = inputs[:, 3 * self.units:]
      elif self.implementation == 1:
        x_i = K.dot(inputs * dp_mask[0], self.kernel_i) + self.bias_i
        x_f = K.dot(inputs * dp_mask[1], self.kernel_f) + self.bias_f
        x_c = K.dot(inputs * dp_mask[2], self.kernel_c) + self.bias_c
        x_o = K.dot(inputs * dp_mask[3], self.kernel_o) + self.bias_o
      else:
        raise ValueError('Unknown `implementation` mode.')

      i = self.recurrent_activation(x_i + K.dot(h_tm1 * rec_dp_mask[0],
                                                self.recurrent_kernel_i))
      f = self.recurrent_activation(x_f + K.dot(h_tm1 * rec_dp_mask[1],
                                                self.recurrent_kernel_f))
      c = f * c_tm1 + i * self.activation(
          x_c + K.dot(h_tm1 * rec_dp_mask[2], self.recurrent_kernel_c))
      o = self.recurrent_activation(x_o + K.dot(h_tm1 * rec_dp_mask[3],
                                                self.recurrent_kernel_o))
    h = o * self.activation(c)
    if 0 < self.dropout + self.recurrent_dropout:
      h._uses_learning_phase = True
    return h, [h, c]

  def get_config(self):
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'unit_forget_bias': self.unit_forget_bias,
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint),
        'dropout': self.dropout,
        'recurrent_dropout': self.recurrent_dropout
    }
    base_config = super(LSTM, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
