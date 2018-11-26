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
"""Recurrent layers backed by cuDNN.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import constant_op
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.util.tf_export import tf_export


class _CuDNNRNN(RNN):
  """Private base class for CuDNNGRU and CuDNNLSTM layers.

  Arguments:
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
    time_major: Boolean (default False). If true, the inputs and outputs will be
        in shape `(timesteps, batch, ...)`, whereas in the False case, it will
        be `(batch, timesteps, ...)`.
  """

  def __init__(self,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               time_major=False,
               **kwargs):
    # We invoke the base layer's initializer directly here because we do not
    # want to create RNN cell instance.
    super(RNN, self).__init__(**kwargs)  # pylint: disable=bad-super-call
    self.return_sequences = return_sequences
    self.return_state = return_state
    self.go_backwards = go_backwards
    self.stateful = stateful
    self.time_major = time_major
    self.supports_masking = False
    self.input_spec = [InputSpec(ndim=3)]
    if hasattr(self.cell.state_size, '__len__'):
      state_size = self.cell.state_size
    else:
      state_size = [self.cell.state_size]
    self.state_spec = [InputSpec(shape=(None, dim)) for dim in state_size]
    self.constants_spec = None
    self._states = None
    self._num_constants = None
    self._num_inputs = None
    self._vector_shape = constant_op.constant([-1])

  def _canonical_to_params(self, weights, biases):
    weights = [array_ops.reshape(x, self._vector_shape) for x in weights]
    biases = [array_ops.reshape(x, self._vector_shape) for x in biases]
    return array_ops.concat(weights + biases, axis=0)

  def call(self, inputs, mask=None, training=None, initial_state=None):
    if isinstance(mask, list):
      mask = mask[0]
    if mask is not None:
      raise ValueError('Masking is not supported for CuDNN RNNs.')

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

    if len(initial_state) != len(self.states):
      raise ValueError('Layer has ' + str(len(self.states)) +
                       ' states but was passed ' + str(len(initial_state)) +
                       ' initial states.')

    if self.go_backwards:
      # Reverse time axis.
      inputs = K.reverse(inputs, 1)
    output, states = self._process_batch(inputs, initial_state)

    if self.stateful:
      updates = []
      for i in range(len(states)):
        updates.append(state_ops.assign(self.states[i], states[i]))
      self.add_update(updates, inputs)

    if self.return_state:
      return [output] + states
    else:
      return output

  def get_config(self):
    config = {
        'return_sequences': self.return_sequences,
        'return_state': self.return_state,
        'go_backwards': self.go_backwards,
        'stateful': self.stateful,
        'time_major': self.time_major,
    }
    base_config = super(  # pylint: disable=bad-super-call
        RNN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  @property
  def trainable_weights(self):
    if self.trainable and self.built:
      return [self.kernel, self.recurrent_kernel, self.bias]
    return []

  @property
  def non_trainable_weights(self):
    if not self.trainable and self.built:
      return [self.kernel, self.recurrent_kernel, self.bias]
    return []

  @property
  def losses(self):
    return super(RNN, self).losses

  def get_losses_for(self, inputs=None):
    return super(  # pylint: disable=bad-super-call
        RNN, self).get_losses_for(inputs=inputs)


@tf_export('keras.layers.CuDNNGRU')
class CuDNNGRU(_CuDNNRNN):
  """Fast GRU implementation backed by cuDNN.

  More information about cuDNN can be found on the [NVIDIA
  developer website](https://developer.nvidia.com/cudnn).
  Can only be run on GPU.

  Arguments:
      units: Positive integer, dimensionality of the output space.
      kernel_initializer: Initializer for the `kernel` weights matrix, used for
        the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel` weights
        matrix, used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation").
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix.
      recurrent_constraint: Constraint function applied to the
        `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      return_sequences: Boolean. Whether to return the last output in the output
        sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state in addition to the
        output.
      go_backwards: Boolean (default False). If True, process the input sequence
        backwards and return the reversed sequence.
      stateful: Boolean (default False). If True, the last state for each sample
        at index i in a batch will be used as initial state for the sample of
        index i in the following batch.
  """

  def __init__(self,
               units,
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
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               **kwargs):
    self.units = units
    cell_spec = collections.namedtuple('cell', 'state_size')
    self._cell = cell_spec(state_size=self.units)
    super(CuDNNGRU, self).__init__(
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        **kwargs)

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

  @property
  def cell(self):
    return self._cell

  def build(self, input_shape):
    super(CuDNNGRU, self).build(input_shape)
    if isinstance(input_shape, list):
      input_shape = input_shape[0]
    input_dim = int(input_shape[-1])

    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 3),
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

    self.bias = self.add_weight(
        shape=(self.units * 6,),
        name='bias',
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint)

    self.built = True

  def _process_batch(self, inputs, initial_state):
    if not self.time_major:
      inputs = array_ops.transpose(inputs, perm=(1, 0, 2))
    input_h = initial_state[0]
    input_h = array_ops.expand_dims(input_h, axis=0)

    params = self._canonical_to_params(
        weights=[
            self.kernel[:, self.units:self.units * 2],
            self.kernel[:, :self.units],
            self.kernel[:, self.units * 2:],
            self.recurrent_kernel[:, self.units:self.units * 2],
            self.recurrent_kernel[:, :self.units],
            self.recurrent_kernel[:, self.units * 2:],
        ],
        biases=[
            self.bias[self.units:self.units * 2],
            self.bias[:self.units],
            self.bias[self.units * 2:self.units * 3],
            self.bias[self.units * 4:self.units * 5],
            self.bias[self.units * 3:self.units * 4],
            self.bias[self.units * 5:],
        ],
    )

    outputs, h, _, _ = gen_cudnn_rnn_ops.cudnn_rnn(
        inputs,
        input_h=input_h,
        input_c=0,
        params=params,
        is_training=True,
        rnn_mode='gru')

    if self.stateful or self.return_state:
      h = h[0]
    if self.return_sequences:
      if self.time_major:
        output = outputs
      else:
        output = array_ops.transpose(outputs, perm=(1, 0, 2))
    else:
      output = outputs[-1]
    return output, [h]

  def get_config(self):
    config = {
        'units': self.units,
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
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(CuDNNGRU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf_export('keras.layers.CuDNNLSTM')
class CuDNNLSTM(_CuDNNRNN):
  """Fast LSTM implementation backed by cuDNN.

  More information about cuDNN can be found on the [NVIDIA
  developer website](https://developer.nvidia.com/cudnn).
  Can only be run on GPU.

  Arguments:
      units: Positive integer, dimensionality of the output space.
      kernel_initializer: Initializer for the `kernel` weights matrix, used for
        the linear transformation of the inputs.
      unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate
        at initialization. Setting it to true will also force
        `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
        al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
      recurrent_initializer: Initializer for the `recurrent_kernel` weights
        matrix, used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation").
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix.
      recurrent_constraint: Constraint function applied to the
        `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      return_sequences: Boolean. Whether to return the last output. in the
        output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state in addition to the
        output.
      go_backwards: Boolean (default False). If True, process the input sequence
        backwards and return the reversed sequence.
      stateful: Boolean (default False). If True, the last state for each sample
        at index i in a batch will be used as initial state for the sample of
        index i in the following batch.
  """

  def __init__(self,
               units,
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
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               **kwargs):
    self.units = units
    cell_spec = collections.namedtuple('cell', 'state_size')
    self._cell = cell_spec(state_size=(self.units, self.units))
    super(CuDNNLSTM, self).__init__(
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        **kwargs)

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

  @property
  def cell(self):
    return self._cell

  def build(self, input_shape):
    super(CuDNNLSTM, self).build(input_shape)
    if isinstance(input_shape, list):
      input_shape = input_shape[0]
    input_dim = int(input_shape[-1])

    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 4),
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

    if self.unit_forget_bias:

      def bias_initializer(_, *args, **kwargs):
        return array_ops.concat([
            self.bias_initializer((self.units * 5,), *args, **kwargs),
            initializers.Ones()((self.units,), *args, **kwargs),
            self.bias_initializer((self.units * 2,), *args, **kwargs),
        ], axis=0)
    else:
      bias_initializer = self.bias_initializer
    self.bias = self.add_weight(
        shape=(self.units * 8,),
        name='bias',
        initializer=bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint)

    self.built = True

  def _process_batch(self, inputs, initial_state):
    if not self.time_major:
      inputs = array_ops.transpose(inputs, perm=(1, 0, 2))
    input_h = initial_state[0]
    input_c = initial_state[1]
    input_h = array_ops.expand_dims(input_h, axis=0)
    input_c = array_ops.expand_dims(input_c, axis=0)

    params = self._canonical_to_params(
        weights=[
            self.kernel[:, :self.units],
            self.kernel[:, self.units:self.units * 2],
            self.kernel[:, self.units * 2:self.units * 3],
            self.kernel[:, self.units * 3:],
            self.recurrent_kernel[:, :self.units],
            self.recurrent_kernel[:, self.units:self.units * 2],
            self.recurrent_kernel[:, self.units * 2:self.units * 3],
            self.recurrent_kernel[:, self.units * 3:],
        ],
        biases=[
            self.bias[:self.units],
            self.bias[self.units:self.units * 2],
            self.bias[self.units * 2:self.units * 3],
            self.bias[self.units * 3:self.units * 4],
            self.bias[self.units * 4:self.units * 5],
            self.bias[self.units * 5:self.units * 6],
            self.bias[self.units * 6:self.units * 7],
            self.bias[self.units * 7:],
        ],
    )

    outputs, h, c, _ = gen_cudnn_rnn_ops.cudnn_rnn(
        inputs,
        input_h=input_h,
        input_c=input_c,
        params=params,
        is_training=True)

    if self.stateful or self.return_state:
      h = h[0]
      c = c[0]
    if self.return_sequences:
      if self.time_major:
        output = outputs
      else:
        output = array_ops.transpose(outputs, perm=(1, 0, 2))
    else:
      output = outputs[-1]
    return output, [h, c]

  def get_config(self):
    config = {
        'units': self.units,
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
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(CuDNNLSTM, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
