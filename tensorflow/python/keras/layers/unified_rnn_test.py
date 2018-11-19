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
"""Tests for UnifiedLSTM layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import keras
from tensorflow.python.client import session
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine.base_layer import \
  InputSpec
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


class RNNTest(test.TestCase):

  def test_unifiedRNN(self):
    rewrites = rewriter_config_pb2.RewriterConfig()
    rewrites.function_optimization = rewriter_config_pb2.RewriterConfig.OFF
    customer_optimizer = rewrites.custom_optimizers.add()
    customer_optimizer.name = 'ExperimentalImplementationSelector'
    rewrites.min_graph_nodes = -1
    graph_options = config_pb2.GraphOptions(rewrite_options=rewrites)
    config = config_pb2.ConfigProto(graph_options=graph_options)

    input_shape = 10
    rnn_state_size = 8
    output_shape = 8
    timestep = 4
    batch = 100
    epoch = 1

    with ops.Graph().as_default(), session.Session(config=config) as sess:
      (x_train, y_train), _ = testing_utils.get_test_data(
          train_samples=batch,
          test_samples=0,
          input_shape=(timestep, input_shape),
          num_classes=output_shape)
      y_train = keras.utils.to_categorical(y_train)

      layer = UnifiedLSTM(rnn_state_size)

      inputs = array_ops.placeholder(
          dtypes.float32, shape=(None, timestep, input_shape), name='inputs')
      predict = array_ops.placeholder(
          dtypes.float32, shape=(None, output_shape), name='predict')

      outputs, runtime = layer(inputs)
      loss = losses.softmax_cross_entropy(predict, outputs)
      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      train_op = optimizer.minimize(loss)

      sess.run([variables.global_variables_initializer()])
      existing_loss = 0
      for _ in range(epoch):
        loss_value, _, runtime_value = sess.run([loss, train_op, runtime], {
            inputs: x_train,
            predict: y_train
        })
        if test.is_gpu_available():
          self.assertEquals(runtime_value, b'cudnn')
        else:
          self.assertEquals(runtime_value, b'cpu')
        # Make sure the loss is updated for every epoch
        # (layer weights properly updated).
        self.assertNotEqual(existing_loss, loss_value)
        existing_loss = loss_value

  def test_unifiedRNN_with_cond(self):
    # This test is to demonstrate the graph rewrite of grappler plugin under
    # the condition that the function returns different number of internal
    # states.
    rewrites = rewriter_config_pb2.RewriterConfig()
    rewrites.function_optimization = rewriter_config_pb2.RewriterConfig.OFF
    customer_optimizer = rewrites.custom_optimizers.add()
    customer_optimizer.name = 'ExperimentalImplementationSelector'
    rewrites.min_graph_nodes = -1
    graph_options = config_pb2.GraphOptions(rewrite_options=rewrites)
    config = config_pb2.ConfigProto(graph_options=graph_options)

    input_shape = 10
    rnn_state_size = 8
    output_shape = 8
    timestep = 4
    batch = 100
    epoch = 1

    with ops.Graph().as_default(), session.Session(config=config) as sess:
      (x_train, y_train), _ = testing_utils.get_test_data(
          train_samples=batch,
          test_samples=0,
          input_shape=(timestep, input_shape),
          num_classes=output_shape)
      y_train = keras.utils.to_categorical(y_train)

      layer = UnifiedLSTM(rnn_state_size)

      inputs = array_ops.placeholder(
          dtypes.float32, shape=(None, timestep, input_shape), name='inputs')
      predict = array_ops.placeholder(
          dtypes.float32, shape=(None, output_shape), name='predict')

      zeros = array_ops.zeros([batch, output_shape])
      dummy_runtime = constant_op.constant(
          'unknown', dtype=dtypes.string, name='runtime')
      a = constant_op.constant(0)
      b = constant_op.constant(1)
      # Will always run the lstm layer.
      outputs, runtime = control_flow_ops.cond(
          gen_math_ops.less(a, b),
          lambda: layer(inputs),
          lambda: (zeros, dummy_runtime))
      loss = losses.softmax_cross_entropy(predict, outputs)
      optimizer = gradient_descent.GradientDescentOptimizer(0.001)
      train_op = optimizer.minimize(loss)

      sess.run([variables.global_variables_initializer()])
      existing_loss = 0

      for _ in range(epoch):
        loss_value, _, runtime_value = sess.run([loss, train_op, runtime], {
            inputs: x_train,
            predict: y_train
        })
        if test.is_gpu_available():
          self.assertEquals(runtime_value, b'cudnn')
        else:
          self.assertEquals(runtime_value, b'cpu')
        # Make sure the loss is updated for every epoch
        # (layer weights properly updated).
        self.assertNotEqual(existing_loss, loss_value)
        existing_loss = loss_value


class UnifiedLSTM(RNN):

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
               time_major=False,
               **kwargs):
    super(RNN, self).__init__(**kwargs)  # pylint: disable=bad-super-call
    self.units = units
    cell_spec = collections.namedtuple('cell', ['state_size', 'output_size'])
    self.cell = cell_spec(
        state_size=(self.units, self.units), output_size=self.units)

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

    self.return_sequences = return_sequences
    self.return_state = return_state
    self.go_backwards = go_backwards
    self.stateful = stateful
    self.time_major = time_major
    self._num_constants = None
    self._num_inputs = None
    self._states = None
    self.input_spec = [InputSpec(ndim=3)]
    self.state_spec = [
        InputSpec(shape=(None, dim)) for dim in (self.units, self.units)
    ]

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    super(UnifiedLSTM, self).build(input_shape)
    if isinstance(input_shape, list):
      input_shape = input_shape[0]
    input_dim = int(input_shape[-1])

    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 4),
        name='kernel',
        dtype=dtypes.float32,
        use_resource=True,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        dtype=dtypes.float32,
        use_resource=True,
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

    # Normal LSTM has 4 bias instead of 8.
    if self.unit_forget_bias:

      def bias_initializer(_, *args, **kwargs):
        return array_ops.concat([
            self.bias_initializer((self.units * 5,), *args, **kwargs),
            initializers.Ones()((self.units,), *args, **kwargs),
            self.bias_initializer((self.units * 2,), *args, **kwargs),
        ],
                                axis=0)
    else:
      bias_initializer = self.bias_initializer
    self.bias = self.add_weight(
        shape=(self.units * 8,),
        name='bias',
        dtype=dtypes.float32,
        use_resource=True,
        initializer=bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint)
    self.built = True

  def call(self, inputs, mask=None, training=None, initial_state=None):
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

    outputs, [new_h, new_c], runtime = normal_lstm(
        inputs, initial_state[0], initial_state[1], self.kernel,
        self.recurrent_kernel, self.bias, self.units)

    function.register(cudnn_lstm, inputs, initial_state[0], initial_state[1],
                      self.kernel, self.recurrent_kernel, self.bias, self.units)

    states = [new_h, new_c]

    if self.stateful:
      updates = []
      for i in range(len(states)):
        updates.append(state_ops.assign(self.states[i], states[i]))
      self.add_update(updates, inputs)

    if self.return_sequences:
      output = outputs
    else:
      output = outputs[:, -1, :]

    if self.return_state:
      return [output] + states
    else:
      return output, runtime

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if isinstance(input_shape, list):
      input_shape = input_shape[0]

    if _is_multiple_state(self.cell.state_size):
      state_size = self.cell.state_size
    else:
      state_size = [self.cell.state_size]

    if getattr(self.cell, 'output_size', None) is not None:
      output_dim = tensor_shape.as_shape(self.cell.output_size).as_list()
    else:
      # Note that state_size[0] could be a tensor_shape or int.
      output_dim = tensor_shape.as_shape(state_size[0]).as_list()

    if self.return_sequences:
      output_shape = tuple([input_shape[0], input_shape[1]] + output_dim)
    else:
      output_shape = tuple([input_shape[0]] + output_dim)

    if self.return_state:
      state_shape = [
          tuple([input_shape[0]] + tensor_shape.as_shape(dim).as_list())
          for dim in state_size
      ]
      return [output_shape] + state_shape
    else:
      return output_shape

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
    return super(RNN, self).get_losses_for(inputs=inputs)   # pylint: disable=bad-super-call

  def get_weights(self):
    return super(RNN, self).get_weights()  # pylint: disable=bad-super-call


def _canonical_to_params(weights, biases, shape):
  weights = [array_ops.reshape(x, shape) for x in weights]
  biases = [array_ops.reshape(x, shape) for x in biases]
  return array_ops.concat(weights + biases, axis=0)


def _is_multiple_state(state_size):
  """Check whether the state_size contains multiple states."""
  return (hasattr(state_size, '__len__') and
          not isinstance(state_size, tensor_shape.TensorShape))


@function.defun_with_attributes(
    attributes={
        'experimental_api_implements': 'lstm',
        'experimental_api_preferred_device': 'CPU'
    })
def normal_lstm(inputs, init_h, init_c, kernel, recurrent_kernel, bias, units):
  input_shape = K.int_shape(inputs)
  timesteps = input_shape[1]

  def step(cell_inputs, cell_states):
    h_tm1 = cell_states[0]  # previous memory state
    c_tm1 = cell_states[1]  # previous carry state

    # Only use the second half of the bias weights.
    _, real_bias = array_ops.split(bias, 2)

    z = K.dot(cell_inputs, kernel)
    z += K.dot(h_tm1, recurrent_kernel)
    z = K.bias_add(z, real_bias)

    z0 = z[:, :units]
    z1 = z[:, units:2 * units]
    z2 = z[:, 2 * units:3 * units]
    z3 = z[:, 3 * units:]

    i = activations.get('hard_sigmoid')(z0)
    f = activations.get('hard_sigmoid')(z1)
    c = f * c_tm1 + i * activations.get('tanh')(z2)
    o = activations.get('hard_sigmoid')(z3)

    h = o * activations.get('tanh')(c)
    return h, [h, c]

  _, outputs, new_states = K.rnn(
      step,
      inputs, [init_h, init_c],
      constants=None,
      unroll=False,
      input_length=timesteps)
  return outputs, new_states, constant_op.constant(
      'cpu', dtype=dtypes.string, name='runtime')


@function.defun_with_attributes(
    attributes={
        'experimental_api_implements': 'lstm',
        'experimental_api_preferred_device': 'GPU'
    })
def cudnn_lstm(inputs, input_h, input_c, kernel, recurrent_kernel, bias, units):
  inputs = array_ops.transpose(inputs, perm=(1, 0, 2))
  input_h = array_ops.expand_dims(input_h, axis=0)
  input_c = array_ops.expand_dims(input_c, axis=0)

  params = _canonical_to_params(
      weights=[
          kernel[:, :units],
          kernel[:, units:units * 2],
          kernel[:, units * 2:units * 3],
          kernel[:, units * 3:],
          recurrent_kernel[:, :units],
          recurrent_kernel[:, units:units * 2],
          recurrent_kernel[:, units * 2:units * 3],
          recurrent_kernel[:, units * 3:],
      ],
      biases=[
          bias[:units],
          bias[units:units * 2],
          bias[units * 2:units * 3],
          bias[units * 3:units * 4],
          bias[units * 4:units * 5],
          bias[units * 5:units * 6],
          bias[units * 6:units * 7],
          bias[units * 7:],
      ],
      shape=constant_op.constant([-1]))

  outputs, h, c, _ = gen_cudnn_rnn_ops.cudnn_rnn(
      inputs, input_h=input_h, input_c=input_c, params=params)
  outputs = array_ops.transpose(outputs, perm=[1, 0, 2])
  h = h[0]
  c = c[0]
  return outputs, [h, c], constant_op.constant(
      'cudnn', dtype=dtypes.string, name='runtime')


if __name__ == '__main__':
  test.main()
