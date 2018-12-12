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

import os
import shutil
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import keras
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import gradient_descent


# Global config for grappler setting that is used for graph mode test.
_rewrites = rewriter_config_pb2.RewriterConfig()
_rewrites.function_optimization = rewriter_config_pb2.RewriterConfig.OFF
_customer_optimizer = _rewrites.custom_optimizers.add()
_customer_optimizer.name = 'ExperimentalImplementationSelector'
_rewrites.min_graph_nodes = -1
_graph_options = config_pb2.GraphOptions(rewrite_options=_rewrites)
_config = config_pb2.ConfigProto(graph_options=_graph_options)


@test_util.run_v1_only('b/120545219')
class UnifiedLSTMTest(test.TestCase, parameterized.TestCase):

  def test_unifiedLSTM(self):
    input_shape = 10
    rnn_state_size = 8
    output_shape = 8
    timestep = 4
    batch = 100
    epoch = 1

    with self.cached_session(config=_config, use_gpu=True) as sess:
      (x_train, y_train), _ = testing_utils.get_test_data(
          train_samples=batch,
          test_samples=0,
          input_shape=(timestep, input_shape),
          num_classes=output_shape)
      y_train = keras.utils.to_categorical(y_train, output_shape)

      layer = keras.layers.UnifiedLSTM(rnn_state_size, return_runtime=True)

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
          self.assertEqual(runtime_value, b'cudnn')
        else:
          self.assertEqual(runtime_value, b'cpu')
        # Make sure the loss is updated for every epoch
        # (layer weights properly updated).
        self.assertNotEqual(existing_loss, loss_value)
        existing_loss = loss_value

  def test_unifiedLSTM_with_cond(self):
    # This test is to demonstrate the graph rewrite of grappler plugin under
    # the condition that the function returns different number of internal
    # states.
    input_shape = 10
    rnn_state_size = 8
    output_shape = 8
    timestep = 4
    batch = 100
    epoch = 1

    with self.cached_session(config=_config, use_gpu=True) as sess:
      (x_train, y_train), _ = testing_utils.get_test_data(
          train_samples=batch,
          test_samples=0,
          input_shape=(timestep, input_shape),
          num_classes=output_shape)
      y_train = keras.utils.to_categorical(y_train, output_shape)

      layer = keras.layers.UnifiedLSTM(rnn_state_size, return_runtime=True)

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
          self.assertEqual(runtime_value, b'cudnn')
        else:
          self.assertEqual(runtime_value, b'cpu')
        # Make sure the loss is updated for every epoch
        # (layer weights properly updated).
        self.assertNotEqual(existing_loss, loss_value)
        existing_loss = loss_value

  @parameterized.named_parameters(
      ('non_tan_activation', 'relu', 'sigmoid', 0, False, True),
      ('non_sigmoid_recur_activation', 'tanh', 'relu', 0, False, True),
      ('use_recurrent_dropout', 'tanh', 'sigmoid', 0.1, False, True),
      ('unroll', 'tanh', 'sigmoid', 0, True, True),
      ('not_use_bias', 'tanh', 'sigmoid', 0, False, False),
  )
  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_could_use_defun_backend(self, activation, recurrent_activation,
                                   recurrent_dropout, unroll, use_bias):
    layer = keras.layers.UnifiedLSTM(
        1,
        activation=activation,
        recurrent_activation=recurrent_activation,
        recurrent_dropout=recurrent_dropout,
        unroll=unroll,
        use_bias=use_bias)
    self.assertFalse(layer.could_use_cudnn)

  def test_unified_lstm_feature_parity_with_canonical_lstm(self):
    with context.eager_mode():
      # Run this test under eager only due to b/120160788 for model.set_weights.
      input_shape = 10
      rnn_state_size = 8
      timestep = 4
      batch = 20

      (x_train, y_train), _ = testing_utils.get_test_data(
          train_samples=batch,
          test_samples=0,
          input_shape=(timestep, input_shape),
          num_classes=rnn_state_size)
      y_train = keras.utils.to_categorical(y_train, rnn_state_size)

      inputs = keras.layers.Input(
          shape=[timestep, input_shape], dtype=dtypes.float32)
      lstm_layer = keras.layers.LSTM(rnn_state_size,
                                     recurrent_activation='sigmoid')
      output = lstm_layer(inputs)
      lstm_model = keras.models.Model(inputs, output)
      weights = lstm_model.get_weights()
      y_1 = lstm_model.predict(x_train)
      lstm_model.compile('rmsprop', 'mse')
      lstm_model.fit(x_train, y_train)
      y_2 = lstm_model.predict(x_train)

      with test_util.device(use_gpu=True):
        cudnn_layer = keras.layers.UnifiedLSTM(rnn_state_size)
        cudnn_model = keras.models.Model(inputs, cudnn_layer(inputs))
      cudnn_model.set_weights(weights)
      y_3 = cudnn_model.predict(x_train)
      cudnn_model.compile('rmsprop', 'mse')
      cudnn_model.fit(x_train, y_train)
      y_4 = cudnn_model.predict(x_train)

      self.assertAllClose(y_1, y_3)
      self.assertAllClose(y_2, y_4)

  @parameterized.named_parameters(
      # test_name, use_bias, bias_initializer, activation
      ('normal', True, 'zeros'),
      ('no_bias', False, 'zeros'),
      ('random_bias', True, 'random_uniform'),
  )
  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_unified_lstm_model_save_load(self, use_bias, bias_initializer):
    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)
    h5_path = os.path.join(temp_dir, 'test.h5')

    batch = 10
    timestep = 3
    input_dim = 5
    units = 2

    x = np.random.random((batch, timestep, input_dim))

    def build_model():
      inputs = keras.layers.Input(
          shape=[timestep, input_dim], dtype=dtypes.float32)
      layer = keras.layers.UnifiedLSTM(
          units,
          use_bias=use_bias,
          bias_initializer=bias_initializer)
      output = layer(inputs)
      return keras.models.Model(inputs, output), layer

    model, layer = build_model()
    y_ref = model.predict(x)
    model.save_weights(h5_path)

    cloned_model, new_layer = build_model()
    cloned_model.load_weights(h5_path)
    y = cloned_model.predict(x)

    self.assertAllClose(y, y_ref)
    self.assertAllClose(layer.get_weights(), new_layer.get_weights())

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_unified_lstm_output_on_multiple_kernel(self):
    input_shape = 10
    rnn_state_size = 8
    timestep = 4
    batch = 100

    x_train = np.random.random((batch, timestep, input_shape))

    inputs = keras.layers.Input(
        shape=[timestep, input_shape], dtype=dtypes.float32)
    with test_util.device(use_gpu=False):
      layer = keras.layers.UnifiedLSTM(rnn_state_size)
      output = layer(inputs)
      cpu_model = keras.models.Model(inputs, output)
      weights = cpu_model.get_weights()
      y_1 = cpu_model.predict(x_train)

    with test_util.device(use_gpu=True):
      layer = keras.layers.UnifiedLSTM(rnn_state_size)
      output = layer(inputs)
      gpu_model = keras.models.Model(inputs, output)
      gpu_model.set_weights(weights)
      y_2 = gpu_model.predict(x_train)

    # Note that CuDNN uses 'sigmoid' as activation, so the unified LSTM uses
    # 'sigmoid' as default. Construct the canonical LSTM with sigmoid to achieve
    # the same output.
    with test_util.device(use_gpu=True):
      layer = keras.layers.LSTM(rnn_state_size, recurrent_activation='sigmoid')
      output = layer(inputs)
      canonical_model = keras.models.Model(inputs, output)
      # Remove the extra cudnn bias since canonical lstm will not use it.
      canonical_model.set_weights(weights[:3])
      y_3 = canonical_model.predict(x_train)

    self.assertAllClose(y_1, y_2)
    self.assertAllClose(y_2, y_3)

  @parameterized.named_parameters(
      # test_name, time_major, go_backwards
      ('normal', False, False),
      ('time_major', True, False),
      ('go_backwards', False, True),
      ('both', True, True),
  )
  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_time_major_and_go_backward(self, time_major, go_backwards):
    input_shape = 10
    rnn_state_size = 8
    timestep = 4
    batch = 100

    x_train = np.random.random((batch, timestep, input_shape))

    def build_model(layer_cls):
      inputs = keras.layers.Input(
          shape=[timestep, input_shape], dtype=dtypes.float32)
      layer = layer_cls(rnn_state_size,
                        recurrent_activation='sigmoid',
                        time_major=time_major,
                        return_sequences=True,
                        go_backwards=go_backwards)
      if time_major:
        converted_input = keras.layers.Lambda(
            lambda t: array_ops.transpose(t, [1, 0, 2]))(inputs)
        outputs = layer(converted_input)
        outputs = keras.layers.Lambda(
            lambda t: array_ops.transpose(t, [1, 0, 2]))(outputs)
      else:
        outputs = layer(inputs)
      return keras.models.Model(inputs, outputs)

    lstm_model = build_model(keras.layers.LSTM)
    y_ref = lstm_model.predict(x_train)
    weights = lstm_model.get_weights()

    unified_lstm_model = build_model(keras.layers.UnifiedLSTM)
    unified_lstm_model.set_weights(weights)
    y = unified_lstm_model.predict(x_train)

    self.assertAllClose(y, y_ref)

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_keras_model_with_lstm(self):
    input_shape = 10
    rnn_state_size = 8
    output_shape = 8
    timestep = 4
    batch = 100
    epoch = 10

    (x_train, y_train), _ = testing_utils.get_test_data(
        train_samples=batch,
        test_samples=0,
        input_shape=(timestep, input_shape),
        num_classes=output_shape)
    y_train = keras.utils.to_categorical(y_train, output_shape)

    layer = keras.layers.UnifiedLSTM(rnn_state_size)

    inputs = keras.layers.Input(
        shape=[timestep, input_shape], dtype=dtypes.float32)

    outputs = layer(inputs)
    model = keras.models.Model(inputs, outputs)
    model.compile('rmsprop', loss='mse')
    model.fit(x_train, y_train, epochs=epoch)
    model.evaluate(x_train, y_train)
    model.predict(x_train)

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_return_sequences_LSTM(self):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    testing_utils.layer_test(
        keras.layers.UnifiedLSTM,
        kwargs={
            'units': units,
            'return_sequences': True
        },
        input_shape=(num_samples, timesteps, embedding_dim))

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_static_shape_inference_LSTM(self):
    # Github issue: 15165
    timesteps = 3
    embedding_dim = 4
    units = 2

    model = keras.models.Sequential()
    inputs = keras.layers.Dense(
        embedding_dim, input_shape=(timesteps, embedding_dim))
    model.add(inputs)
    layer = keras.layers.UnifiedLSTM(units, return_sequences=True)
    model.add(layer)
    outputs = model.layers[-1].output
    self.assertEqual(outputs.get_shape().as_list(), [None, timesteps, units])

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_dynamic_behavior_LSTM(self):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    layer = keras.layers.UnifiedLSTM(units, input_shape=(None, embedding_dim))
    model = keras.models.Sequential()
    model.add(layer)
    model.compile(gradient_descent.GradientDescentOptimizer(0.001), 'mse')
    x = np.random.random((num_samples, timesteps, embedding_dim))
    y = np.random.random((num_samples, units))
    model.train_on_batch(x, y)

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_dropout_LSTM(self):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    testing_utils.layer_test(
        keras.layers.UnifiedLSTM,
        kwargs={
            'units': units,
            'dropout': 0.1,
            'recurrent_dropout': 0.1
        },
        input_shape=(num_samples, timesteps, embedding_dim))

  @parameterized.parameters([0, 1, 2])
  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_implementation_mode_LSTM(self, implementation_mode):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    testing_utils.layer_test(
        keras.layers.UnifiedLSTM,
        kwargs={
            'units': units,
            'implementation': implementation_mode
        },
        input_shape=(num_samples, timesteps, embedding_dim))

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_constraints_LSTM(self):
    embedding_dim = 4
    layer_class = keras.layers.UnifiedLSTM
    k_constraint = keras.constraints.max_norm(0.01)
    r_constraint = keras.constraints.max_norm(0.01)
    b_constraint = keras.constraints.max_norm(0.01)
    layer = layer_class(
        5,
        return_sequences=False,
        weights=None,
        input_shape=(None, embedding_dim),
        kernel_constraint=k_constraint,
        recurrent_constraint=r_constraint,
        bias_constraint=b_constraint)
    layer.build((None, None, embedding_dim))
    self.assertEqual(layer.cell.kernel.constraint, k_constraint)
    self.assertEqual(layer.cell.recurrent_kernel.constraint, r_constraint)
    self.assertEqual(layer.cell.bias.constraint, b_constraint)

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_with_masking_layer_LSTM(self):
    layer_class = keras.layers.UnifiedLSTM
    inputs = np.random.random((2, 3, 4))
    targets = np.abs(np.random.random((2, 3, 5)))
    targets /= targets.sum(axis=-1, keepdims=True)
    model = keras.models.Sequential()
    model.add(keras.layers.Masking(input_shape=(3, 4)))
    model.add(layer_class(units=5, return_sequences=True, unroll=False))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=gradient_descent.GradientDescentOptimizer(0.01))
    model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_stacking_LSTM(self):
    inputs = np.random.random((2, 3, 4))
    targets = np.abs(np.random.random((2, 3, 5)))
    targets /= targets.sum(axis=-1, keepdims=True)
    model = keras.models.Sequential()
    model.add(keras.layers.UnifiedLSTM(10, return_sequences=True, unroll=False))
    model.add(keras.layers.UnifiedLSTM(5, return_sequences=True, unroll=False))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=gradient_descent.GradientDescentOptimizer(0.01))
    model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_masking_with_stacking_LSTM(self):
    inputs = np.random.random((2, 3, 4))
    targets = np.abs(np.random.random((2, 3, 5)))
    targets /= targets.sum(axis=-1, keepdims=True)
    model = keras.models.Sequential()
    model.add(keras.layers.Masking(input_shape=(3, 4)))
    model.add(keras.layers.UnifiedLSTM(10, return_sequences=True, unroll=False))
    model.add(keras.layers.UnifiedLSTM(5, return_sequences=True, unroll=False))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=gradient_descent.GradientDescentOptimizer(0.01))
    model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_from_config_LSTM(self):
    layer_class = keras.layers.UnifiedLSTM
    for stateful in (False, True):
      l1 = layer_class(units=1, stateful=stateful)
      l2 = layer_class.from_config(l1.get_config())
      assert l1.get_config() == l2.get_config()

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_specify_initial_state_keras_tensor(self):
    num_states = 2
    timesteps = 3
    embedding_dim = 4
    units = 3
    num_samples = 2

    # Test with Keras tensor
    inputs = keras.Input((timesteps, embedding_dim))
    initial_state = [keras.Input((units,)) for _ in range(num_states)]
    layer = keras.layers.UnifiedLSTM(units)
    if len(initial_state) == 1:
      output = layer(inputs, initial_state=initial_state[0])
    else:
      output = layer(inputs, initial_state=initial_state)
    assert initial_state[0] in layer._inbound_nodes[0].input_tensors

    model = keras.models.Model([inputs] + initial_state, output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=gradient_descent.GradientDescentOptimizer(0.01))

    inputs = np.random.random((num_samples, timesteps, embedding_dim))
    initial_state = [
        np.random.random((num_samples, units)) for _ in range(num_states)
    ]
    targets = np.random.random((num_samples, units))
    model.train_on_batch([inputs] + initial_state, targets)

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def DISABLED_test_specify_initial_state_non_keras_tensor(self):
    num_states = 2
    timesteps = 3
    embedding_dim = 4
    units = 3
    num_samples = 2

    # Test with non-Keras tensor
    inputs = keras.Input((timesteps, embedding_dim))
    initial_state = [
        keras.backend.random_normal_variable((num_samples, units), 0, 1)
        for _ in range(num_states)
    ]
    layer = keras.layers.UnifiedLSTM(units)
    output = layer(inputs, initial_state=initial_state)

    model = keras.models.Model(inputs, output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=gradient_descent.GradientDescentOptimizer(0.01))

    inputs = np.random.random((num_samples, timesteps, embedding_dim))
    targets = np.random.random((num_samples, units))
    model.train_on_batch(inputs, targets)

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_reset_states_with_values(self):
    num_states = 2
    timesteps = 3
    embedding_dim = 4
    units = 3
    num_samples = 2

    layer = keras.layers.UnifiedLSTM(units, stateful=True)
    layer.build((num_samples, timesteps, embedding_dim))
    layer.reset_states()
    assert len(layer.states) == num_states
    assert layer.states[0] is not None
    self.assertAllClose(
        keras.backend.eval(layer.states[0]),
        np.zeros(keras.backend.int_shape(layer.states[0])),
        atol=1e-4)
    state_shapes = [keras.backend.int_shape(state) for state in layer.states]
    values = [np.ones(shape) for shape in state_shapes]
    if len(values) == 1:
      values = values[0]
    layer.reset_states(values)
    self.assertAllClose(
        keras.backend.eval(layer.states[0]),
        np.ones(keras.backend.int_shape(layer.states[0])),
        atol=1e-4)

    # Test with invalid data
    with self.assertRaises(ValueError):
      layer.reset_states([1] * (len(layer.states) + 1))

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_specify_state_with_masking(self):
    num_states = 2
    timesteps = 3
    embedding_dim = 4
    units = 3
    num_samples = 2

    inputs = keras.Input((timesteps, embedding_dim))
    _ = keras.layers.Masking()(inputs)
    initial_state = [keras.Input((units,)) for _ in range(num_states)]
    output = keras.layers.UnifiedLSTM(units)(
        inputs, initial_state=initial_state)

    model = keras.models.Model([inputs] + initial_state, output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=gradient_descent.GradientDescentOptimizer(0.01))

    inputs = np.random.random((num_samples, timesteps, embedding_dim))
    initial_state = [
        np.random.random((num_samples, units)) for _ in range(num_states)
    ]
    targets = np.random.random((num_samples, units))
    model.train_on_batch([inputs] + initial_state, targets)

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_return_state(self):
    num_states = 2
    timesteps = 3
    embedding_dim = 4
    units = 3
    num_samples = 2

    inputs = keras.Input(batch_shape=(num_samples, timesteps, embedding_dim))
    layer = keras.layers.UnifiedLSTM(units, return_state=True, stateful=True)
    outputs = layer(inputs)
    state = outputs[1:]
    assert len(state) == num_states
    model = keras.models.Model(inputs, state[0])

    inputs = np.random.random((num_samples, timesteps, embedding_dim))
    state = model.predict(inputs)
    self.assertAllClose(keras.backend.eval(layer.states[0]), state, atol=1e-4)

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_state_reuse(self):
    timesteps = 3
    embedding_dim = 4
    units = 3
    num_samples = 2

    inputs = keras.Input(batch_shape=(num_samples, timesteps, embedding_dim))
    layer = keras.layers.UnifiedLSTM(
        units, return_state=True, return_sequences=True)
    outputs = layer(inputs)
    output, state = outputs[0], outputs[1:]
    output = keras.layers.UnifiedLSTM(units)(output, initial_state=state)
    model = keras.models.Model(inputs, output)

    inputs = np.random.random((num_samples, timesteps, embedding_dim))
    model.predict(inputs)

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_initial_states_as_other_inputs(self):
    timesteps = 3
    embedding_dim = 4
    units = 3
    num_samples = 2
    num_states = 2
    layer_class = keras.layers.UnifiedLSTM

    # Test with Keras tensor
    main_inputs = keras.Input((timesteps, embedding_dim))
    initial_state = [keras.Input((units,)) for _ in range(num_states)]
    inputs = [main_inputs] + initial_state

    layer = layer_class(units)
    output = layer(inputs)
    assert initial_state[0] in layer._inbound_nodes[0].input_tensors

    model = keras.models.Model(inputs, output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=gradient_descent.GradientDescentOptimizer(0.01))

    main_inputs = np.random.random((num_samples, timesteps, embedding_dim))
    initial_state = [
        np.random.random((num_samples, units)) for _ in range(num_states)
    ]
    targets = np.random.random((num_samples, units))
    model.train_on_batch([main_inputs] + initial_state, targets)


@test_util.run_v1_only('b/120545219')
class LSTMLayerGraphOnlyTest(test.TestCase):

  def test_statefulness_LSTM(self):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    layer_class = keras.layers.UnifiedLSTM
    with self.cached_session(config=_config):
      model = keras.models.Sequential()
      model.add(
          keras.layers.Embedding(
              4,
              embedding_dim,
              mask_zero=True,
              input_length=timesteps,
              batch_input_shape=(num_samples, timesteps)))
      layer = layer_class(
          units, return_sequences=False, stateful=True, weights=None)
      model.add(layer)
      model.compile(
          optimizer=gradient_descent.GradientDescentOptimizer(0.01), loss='mse')
      out1 = model.predict(np.ones((num_samples, timesteps)))
      self.assertEqual(out1.shape, (num_samples, units))

      # train once so that the states change
      model.train_on_batch(
          np.ones((num_samples, timesteps)), np.ones((num_samples, units)))
      out2 = model.predict(np.ones((num_samples, timesteps)))

      # if the state is not reset, output should be different
      self.assertNotEqual(out1.max(), out2.max())

      # check that output changes after states are reset
      # (even though the model itself didn't change)
      layer.reset_states()
      out3 = model.predict(np.ones((num_samples, timesteps)))
      self.assertNotEqual(out2.max(), out3.max())

      # check that container-level reset_states() works
      model.reset_states()
      out4 = model.predict(np.ones((num_samples, timesteps)))
      self.assertAllClose(out3, out4, atol=1e-5)

      # check that the call to `predict` updated the states
      out5 = model.predict(np.ones((num_samples, timesteps)))
      self.assertNotEqual(out4.max(), out5.max())

      # Check masking
      layer.reset_states()

      left_padded_input = np.ones((num_samples, timesteps))
      left_padded_input[0, :1] = 0
      left_padded_input[1, :2] = 0
      out6 = model.predict(left_padded_input)

      layer.reset_states()

      right_padded_input = np.ones((num_samples, timesteps))
      right_padded_input[0, -1:] = 0
      right_padded_input[1, -2:] = 0
      out7 = model.predict(right_padded_input)

      self.assertAllClose(out7, out6, atol=1e-5)

  def test_regularizers_LSTM(self):
    embedding_dim = 4
    layer_class = keras.layers.UnifiedLSTM
    with self.cached_session(config=_config):
      layer = layer_class(
          5,
          return_sequences=False,
          weights=None,
          input_shape=(None, embedding_dim),
          kernel_regularizer=keras.regularizers.l1(0.01),
          recurrent_regularizer=keras.regularizers.l1(0.01),
          bias_regularizer='l2',
          activity_regularizer='l1')
      layer.build((None, None, 2))
      self.assertEqual(len(layer.losses), 3)
      x = keras.backend.variable(np.ones((2, 3, 2)))
      layer(x)
      self.assertEqual(len(layer.get_losses_for(x)), 1)


class UnifiedLSTMPerformanceTest(test.Benchmark):

  def _measure_performance(self, test_config, model, x_train, y_train):
    batch = test_config['batch']
    epoch = test_config['epoch']
    warmup_epoch = test_config['warmup_epoch']

    # warm up the model
    model.fit(x_train, y_train, batch_size=batch, epochs=warmup_epoch)
    start_time = time.time()
    model.fit(x_train, y_train, batch_size=batch, epochs=epoch - warmup_epoch)
    end_time = time.time()
    return (end_time - start_time) / (epoch - warmup_epoch)

  def _time_performance_run_cudnn_lstm(self, test_config, x_train, y_train):
    # Get the performance number for standard Cudnn LSTM
    input_shape = test_config['input_shape']
    rnn_state_size = test_config['rnn_state_size']
    timestep = test_config['timestep']

    cudnn_lstm_layer = keras.layers.CuDNNLSTM(rnn_state_size)
    inputs = keras.layers.Input(
        shape=[timestep, input_shape], dtype=dtypes.float32)

    outputs = cudnn_lstm_layer(inputs)
    model = keras.models.Model(inputs, outputs)
    model.compile('sgd', 'mse')

    sec_per_epoch = self._measure_performance(
        test_config, model, x_train, y_train)
    logging.info('Average performance for %s per epoch is: %s',
                 'CuDNN LSTM', sec_per_epoch)
    return sec_per_epoch

  def _time_performance_run_unifed_lstm_gpu(
      self, test_config, x_train, y_train):
    # Get performance number for Unified_LSTM with grappler swap the impl
    input_shape = test_config['input_shape']
    rnn_state_size = test_config['rnn_state_size']
    timestep = test_config['timestep']

    layer = keras.layers.UnifiedLSTM(rnn_state_size)
    inputs = keras.layers.Input(
        shape=[timestep, input_shape], dtype=dtypes.float32)

    outputs = layer(inputs)
    model = keras.models.Model(inputs, outputs)
    model.compile('sgd', 'mse')

    sec_per_epoch = self._measure_performance(
        test_config, model, x_train, y_train)
    logging.info('Average performance for %s per epoch is: %s',
                 'Unified LSTM', sec_per_epoch)
    return sec_per_epoch

  def _time_performance_run_normal_lstm(
      self, test_config, x_train, y_train):
    # Get performance number for standard LSTM on GPU.
    input_shape = test_config['input_shape']
    rnn_state_size = test_config['rnn_state_size']
    timestep = test_config['timestep']

    layer = keras.layers.LSTM(rnn_state_size)
    inputs = keras.layers.Input(
        shape=[timestep, input_shape], dtype=dtypes.float32)

    outputs = layer(inputs)
    model = keras.models.Model(inputs, outputs)
    model.compile('sgd', 'mse')

    sec_per_epoch = self._measure_performance(
        test_config, model, x_train, y_train)
    logging.info('Average performance for %s per epoch is: %s',
                 'Normal LSTM', sec_per_epoch)
    return sec_per_epoch

  def _benchmark_performance_with_standard_cudnn_impl(self):
    if not test.is_gpu_available():
      self.skipTest('performance test will only run on GPU')

    mode = 'eager' if context.executing_eagerly() else 'graph'
    batch = 64
    num_batch = 10
    test_config = {
        'input_shape': 128,
        'rnn_state_size': 64,
        'output_shape': 64,
        'timestep': 50,
        'batch': batch,
        'epoch': 20,
        # The performance for warmup epoch is ignored.
        'warmup_epoch': 1,
    }
    (x_train, y_train), _ = testing_utils.get_test_data(
        train_samples=(batch * num_batch),
        test_samples=0,
        input_shape=(test_config['timestep'], test_config['input_shape']),
        num_classes=test_config['output_shape'])
    y_train = keras.utils.to_categorical(y_train, test_config['output_shape'])

    cudnn_sec_per_epoch = self._time_performance_run_cudnn_lstm(
        test_config, x_train, y_train)
    unified_lstm_sec_per_epoch = self._time_performance_run_unifed_lstm_gpu(
        test_config, x_train, y_train)
    normal_lstm_sec_per_epoch = self._time_performance_run_normal_lstm(
        test_config, x_train, y_train)

    cudnn_vs_unified = cudnn_sec_per_epoch / unified_lstm_sec_per_epoch
    unified_vs_normal = normal_lstm_sec_per_epoch / unified_lstm_sec_per_epoch

    self.report_benchmark(name='keras_cudnn_lstm_' + mode,
                          wall_time=cudnn_sec_per_epoch,
                          iters=test_config['epoch'],
                          extras=test_config)
    self.report_benchmark(name='keras_unified_lstm_' + mode,
                          wall_time=unified_lstm_sec_per_epoch,
                          iters=test_config['epoch'],
                          extras=test_config)
    self.report_benchmark(name='keras_canonical_lstm_' + mode,
                          wall_time=normal_lstm_sec_per_epoch,
                          iters=test_config['epoch'],
                          extras=test_config)

    logging.info('Expect the performance of Unified LSTM is within 80% of '
                 'CuDNN LSTM, got {0:.2f}%'.format(cudnn_vs_unified * 100))
    logging.info('Expect the performance of Unified LSTM is more than 5 times'
                 ' of normal LSTM, got {0:.2f}'.format(unified_vs_normal))

  def benchmark_performance_graph(self):
    with context.graph_mode(), session_lib.Session(config=_config):
      self._benchmark_performance_with_standard_cudnn_impl()

  def benchmark_performance_eager(self):
    with context.eager_mode():
      self._benchmark_performance_with_standard_cudnn_impl()


if __name__ == '__main__':
  test.main()
