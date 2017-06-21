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
"""Convolutional-recurrent layers.
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
from tensorflow.contrib.keras.python.keras.layers.recurrent import Recurrent
from tensorflow.contrib.keras.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape


class ConvRecurrent2D(Recurrent):
  """Abstract base class for convolutional recurrent layers.

  Do not use in a model -- it's not a functional layer!

  Arguments:
      filters: Integer, the dimensionality of the output space
          (i.e. the number output of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
          dimensions of the convolution window.
      strides: An integer or tuple/list of n integers,
          specifying the strides of the convolution.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, time, ..., channels)`
          while `channels_first` corresponds to
          inputs with shape `(batch, time, channels, ...)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      dilation_rate: An integer or tuple/list of n integers, specifying
          the dilation rate to use for dilated convolution.
          Currently, specifying any `dilation_rate` value != 1 is
          incompatible with specifying any `strides` value != 1.
      return_sequences: Boolean. Whether to return the last output
          in the output sequence, or the full sequence.
      go_backwards: Boolean (default False).
          If True, rocess the input sequence backwards.
      stateful: Boolean (default False). If True, the last state
          for each sample at index i in a batch will be used as initial
          state for the sample of index i in the following batch.

  Input shape:
      5D tensor with shape `(num_samples, timesteps, channels, rows, cols)`.

  Output shape:
      - if `return_sequences`: 5D tensor with shape
          `(num_samples, timesteps, channels, rows, cols)`.
      - else, 4D tensor with shape `(num_samples, channels, rows, cols)`.

  # Masking
      This layer supports masking for input data with a variable number
      of timesteps. To introduce masks to your data,
      use an `Embedding` layer with the `mask_zero` parameter
      set to `True`.
      **Note:** for the time being, masking is only supported with Theano.

  # Note on using statefulness in RNNs
      You can set RNN layers to be 'stateful', which means that the states
      computed for the samples in one batch will be reused as initial states
      for the samples in the next batch.
      This assumes a one-to-one mapping between
      samples in different successive batches.

      To enable statefulness:
          - specify `stateful=True` in the layer constructor.
          - specify a fixed batch size for your model, by passing
              a `batch_input_size=(...)` to the first layer in your model.
              This is the expected shape of your inputs *including the batch
              size*.
              It should be a tuple of integers, e.g. `(32, 10, 100)`.

      To reset the states of your model, call `.reset_states()` on either
      a specific layer, or on your entire model.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               return_sequences=False,
               go_backwards=False,
               stateful=False,
               **kwargs):
    super(ConvRecurrent2D, self).__init__(**kwargs)
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                    'dilation_rate')
    self.return_sequences = return_sequences
    self.go_backwards = go_backwards
    self.stateful = stateful
    self.input_spec = [InputSpec(ndim=5)]
    self.state_spec = None

  def _compute_output_shape(self, input_shape):
    if isinstance(input_shape, list):
      input_shape = input_shape[0]
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_first':
      rows = input_shape[3]
      cols = input_shape[4]
    elif self.data_format == 'channels_last':
      rows = input_shape[2]
      cols = input_shape[3]
    rows = conv_utils.conv_output_length(
        rows,
        self.kernel_size[0],
        padding=self.padding,
        stride=self.strides[0],
        dilation=self.dilation_rate[0])
    cols = conv_utils.conv_output_length(
        cols,
        self.kernel_size[1],
        padding=self.padding,
        stride=self.strides[1],
        dilation=self.dilation_rate[1])
    if self.return_sequences:
      if self.data_format == 'channels_first':
        return tensor_shape.TensorShape(
            [input_shape[0], input_shape[1], self.filters, rows, cols])
      elif self.data_format == 'channels_last':
        return tensor_shape.TensorShape(
            [input_shape[0], input_shape[1], rows, cols, self.filters])
    else:
      if self.data_format == 'channels_first':
        return tensor_shape.TensorShape(
            [input_shape[0], self.filters, rows, cols])
      elif self.data_format == 'channels_last':
        return tensor_shape.TensorShape(
            [input_shape[0], rows, cols, self.filters])

  def get_config(self):
    config = {
        'filters': self.filters,
        'kernel_size': self.kernel_size,
        'strides': self.strides,
        'padding': self.padding,
        'data_format': self.data_format,
        'dilation_rate': self.dilation_rate,
        'return_sequences': self.return_sequences,
        'go_backwards': self.go_backwards,
        'stateful': self.stateful
    }
    base_config = super(ConvRecurrent2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class ConvLSTM2D(ConvRecurrent2D):
  """Convolutional LSTM.

  It is similar to an LSTM layer, but the input transformations
  and recurrent transformations are both convolutional.

  Arguments:
      filters: Integer, the dimensionality of the output space
          (i.e. the number output of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
          dimensions of the convolution window.
      strides: An integer or tuple/list of n integers,
          specifying the strides of the convolution.
          Specifying any stride value != 1 is incompatible with specifying
          any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, time, ..., channels)`
          while `channels_first` corresponds to
          inputs with shape `(batch, time, channels, ...)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".
      dilation_rate: An integer or tuple/list of n integers, specifying
          the dilation rate to use for dilated convolution.
          Currently, specifying any `dilation_rate` value != 1 is
          incompatible with specifying any `strides` value != 1.
      activation: Activation function to use.
          If you don't specify anything, no activation is applied
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
          Use in combination with `bias_initializer="zeros"`.
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
      return_sequences: Boolean. Whether to return the last output
          in the output sequence, or the full sequence.
      go_backwards: Boolean (default False).
          If True, rocess the input sequence backwards.
      stateful: Boolean (default False). If True, the last state
          for each sample at index i in a batch will be used as initial
          state for the sample of index i in the following batch.
      dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
          Fraction of the units to drop for
          the linear transformation of the recurrent state.

  Input shape:
      - if data_format='channels_first'
          5D tensor with shape:
          `(samples,time, channels, rows, cols)`
      - if data_format='channels_last'
          5D tensor with shape:
          `(samples,time, rows, cols, channels)`

   Output shape:
      - if `return_sequences`
           - if data_format='channels_first'
              5D tensor with shape:
              `(samples, time, filters, output_row, output_col)`
           - if data_format='channels_last'
              5D tensor with shape:
              `(samples, time, output_row, output_col, filters)`
      - else
          - if data_format ='channels_first'
              4D tensor with shape:
              `(samples, filters, output_row, output_col)`
          - if data_format='channels_last'
              4D tensor with shape:
              `(samples, output_row, output_col, filters)`
          where o_row and o_col depend on the shape of the filter and
          the padding

  Raises:
      ValueError: in case of invalid constructor arguments.

  References:
      - [Convolutional LSTM Network: A Machine Learning Approach for
      Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)
      The current implementation does not include the feedback loop on the
      cells output
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
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
               return_sequences=False,
               go_backwards=False,
               stateful=False,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
    super(ConvLSTM2D, self).__init__(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        return_sequences=return_sequences,
        go_backwards=go_backwards,
        stateful=stateful,
        **kwargs)
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
    self.state_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

  def build(self, input_shape):
    if isinstance(input_shape, list):
      input_shape = input_shape[0]
    input_shape = tuple(tensor_shape.TensorShape(input_shape).as_list())
    batch_size = input_shape[0] if self.stateful else None
    self.input_spec[0] = InputSpec(shape=(batch_size, None) + input_shape[2:])

    if self.stateful:
      self.reset_states()
    else:
      # initial states: 2 all-zero tensor of shape (filters)
      self.states = [None, None]

    if self.data_format == 'channels_first':
      channel_axis = 2
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis]
    state_shape = [None] * 4
    state_shape[channel_axis] = input_dim
    state_shape = tuple(state_shape)
    self.state_spec = [
        InputSpec(shape=state_shape),
        InputSpec(shape=state_shape)
    ]
    kernel_shape = self.kernel_size + (input_dim, self.filters * 4)
    self.kernel_shape = kernel_shape
    recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 4)

    self.kernel = self.add_weight(
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        name='kernel',
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_kernel = self.add_weight(
        shape=recurrent_kernel_shape,
        initializer=self.recurrent_initializer,
        name='recurrent_kernel',
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)
    if self.use_bias:
      self.bias = self.add_weight(
          shape=(self.filters * 4,),
          initializer=self.bias_initializer,
          name='bias',
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
      if self.unit_forget_bias:
        bias_value = np.zeros((self.filters * 4,))
        bias_value[self.filters:self.filters * 2] = 1.
        K.set_value(self.bias, bias_value)
    else:
      self.bias = None

    self.kernel_i = self.kernel[:, :, :, :self.filters]
    self.recurrent_kernel_i = self.recurrent_kernel[:, :, :, :self.filters]
    self.kernel_f = self.kernel[:, :, :, self.filters:self.filters * 2]
    self.recurrent_kernel_f = self.recurrent_kernel[:, :, :, self.filters:
                                                    self.filters * 2]
    self.kernel_c = self.kernel[:, :, :, self.filters * 2:self.filters * 3]
    self.recurrent_kernel_c = self.recurrent_kernel[:, :, :, self.filters * 2:
                                                    self.filters * 3]
    self.kernel_o = self.kernel[:, :, :, self.filters * 3:]
    self.recurrent_kernel_o = self.recurrent_kernel[:, :, :, self.filters * 3:]

    if self.use_bias:
      self.bias_i = self.bias[:self.filters]
      self.bias_f = self.bias[self.filters:self.filters * 2]
      self.bias_c = self.bias[self.filters * 2:self.filters * 3]
      self.bias_o = self.bias[self.filters * 3:]
    else:
      self.bias_i = None
      self.bias_f = None
      self.bias_c = None
      self.bias_o = None
    self.built = True

  def get_initial_state(self, inputs):
    # (samples, timesteps, rows, cols, filters)
    initial_state = K.zeros_like(inputs)
    # (samples, rows, cols, filters)
    initial_state = K.sum(initial_state, axis=1)
    shape = list(self.kernel_shape)
    shape[-1] = self.filters
    initial_state = self.input_conv(
        initial_state, K.zeros(tuple(shape)), padding=self.padding)

    initial_states = [initial_state for _ in range(2)]
    return initial_states

  def reset_states(self):
    if not self.stateful:
      raise RuntimeError('Layer must be stateful.')
    input_shape = self.input_spec[0].shape
    output_shape = self._compute_output_shape(input_shape)

    if not input_shape[0]:
      raise ValueError('If a RNN is stateful, a complete '
                       'input_shape must be provided '
                       '(including batch size). '
                       'Got input shape: ' + str(input_shape))

    if self.return_sequences:
      out_row, out_col, out_filter = output_shape[2:]
    else:
      out_row, out_col, out_filter = output_shape[1:]

    if hasattr(self, 'states'):
      K.set_value(self.states[0],
                  np.zeros((input_shape[0], out_row, out_col, out_filter)))
      K.set_value(self.states[1],
                  np.zeros((input_shape[0], out_row, out_col, out_filter)))
    else:
      self.states = [
          K.zeros((input_shape[0], out_row, out_col, out_filter)),
          K.zeros((input_shape[0], out_row, out_col, out_filter))
      ]

  def get_constants(self, inputs, training=None):
    constants = []
    if self.implementation == 0 and 0 < self.dropout < 1:
      ones = K.zeros_like(inputs)
      ones = K.sum(ones, axis=1)
      ones += 1

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
      shape = list(self.kernel_shape)
      shape[-1] = self.filters
      ones = K.zeros_like(inputs)
      ones = K.sum(ones, axis=1)
      ones = self.input_conv(ones, K.zeros(shape), padding=self.padding)
      ones += 1.

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

  def input_conv(self, x, w, b=None, padding='valid'):
    conv_out = K.conv2d(
        x,
        w,
        strides=self.strides,
        padding=padding,
        data_format=self.data_format,
        dilation_rate=self.dilation_rate)
    if b is not None:
      conv_out = K.bias_add(conv_out, b, data_format=self.data_format)
    return conv_out

  def reccurent_conv(self, x, w):
    conv_out = K.conv2d(
        x, w, strides=(1, 1), padding='same', data_format=self.data_format)
    return conv_out

  def step(self, inputs, states):
    assert len(states) == 4
    h_tm1 = states[0]
    c_tm1 = states[1]
    dp_mask = states[2]
    rec_dp_mask = states[3]

    x_i = self.input_conv(
        inputs * dp_mask[0], self.kernel_i, self.bias_i, padding=self.padding)
    x_f = self.input_conv(
        inputs * dp_mask[1], self.kernel_f, self.bias_f, padding=self.padding)
    x_c = self.input_conv(
        inputs * dp_mask[2], self.kernel_c, self.bias_c, padding=self.padding)
    x_o = self.input_conv(
        inputs * dp_mask[3], self.kernel_o, self.bias_o, padding=self.padding)
    h_i = self.reccurent_conv(h_tm1 * rec_dp_mask[0], self.recurrent_kernel_i)
    h_f = self.reccurent_conv(h_tm1 * rec_dp_mask[1], self.recurrent_kernel_f)
    h_c = self.reccurent_conv(h_tm1 * rec_dp_mask[2], self.recurrent_kernel_c)
    h_o = self.reccurent_conv(h_tm1 * rec_dp_mask[3], self.recurrent_kernel_o)

    i = self.recurrent_activation(x_i + h_i)
    f = self.recurrent_activation(x_f + h_f)
    c = f * c_tm1 + i * self.activation(x_c + h_c)
    o = self.recurrent_activation(x_o + h_o)
    h = o * self.activation(c)
    return h, [h, c]

  def get_config(self):
    config = {
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout
    }
    base_config = super(ConvLSTM2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
