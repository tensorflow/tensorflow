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
"""Tests for V2 LSTM layer."""

import copy
import os
import shutil
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import keras
from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers import recurrent as rnn_v1
from tensorflow.python.keras.layers import recurrent_v2 as rnn
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import gradient_descent
from tensorflow.python.util import nest


# Global config for grappler setting that is used for graph mode test.
_rewrites = rewriter_config_pb2.RewriterConfig()
_rewrites.implementation_selector = rewriter_config_pb2.RewriterConfig.ON
_rewrites.min_graph_nodes = -1
_graph_options = config_pb2.GraphOptions(rewrite_options=_rewrites)
_config = config_pb2.ConfigProto(graph_options=_graph_options)


@keras_parameterized.run_all_keras_modes(config=_config)
class LSTMV2Test(keras_parameterized.TestCase):

  @parameterized.named_parameters(
      ('non_tan_activation', 'relu', 'sigmoid', 0, False, True),
      ('non_sigmoid_recur_activation', 'tanh', 'relu', 0, False, True),
      ('use_recurrent_dropout', 'tanh', 'sigmoid', 0.1, False, True),
      ('unroll', 'tanh', 'sigmoid', 0, True, True),
      ('not_use_bias', 'tanh', 'sigmoid', 0, False, False),
  )
  def test_could_use_defun_backend(self, activation, recurrent_activation,
                                   recurrent_dropout, unroll, use_bias):
    layer = rnn.LSTM(
        1,
        activation=activation,
        recurrent_activation=recurrent_activation,
        recurrent_dropout=recurrent_dropout,
        unroll=unroll,
        use_bias=use_bias)
    self.assertFalse(layer._could_use_gpu_kernel)

  @testing_utils.run_v2_only
  def test_use_on_default_activation_with_gpu_kernel(self):
    layer = rnn.LSTM(1, activation=nn.tanh)
    self.assertTrue(layer._could_use_gpu_kernel)

    layer = rnn.LSTM(1, recurrent_activation=nn.sigmoid)
    self.assertTrue(layer._could_use_gpu_kernel)

  def test_static_shape_inference_LSTM(self):
    # Github issue: 15165
    timesteps = 3
    embedding_dim = 4
    units = 2

    model = keras.models.Sequential()
    inputs = keras.layers.Dense(
        embedding_dim, input_shape=(timesteps, embedding_dim))
    model.add(inputs)
    layer = rnn.LSTM(units, return_sequences=True)
    model.add(layer)
    outputs = model.layers[-1].output
    self.assertEqual(outputs.shape.as_list(), [None, timesteps, units])

  def test_dynamic_behavior_LSTM(self):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    layer = rnn.LSTM(units, input_shape=(None, embedding_dim))
    model = keras.models.Sequential()
    model.add(layer)
    model.compile(gradient_descent.GradientDescentOptimizer(0.001), 'mse')
    x = np.random.random((num_samples, timesteps, embedding_dim))
    y = np.random.random((num_samples, units))
    model.train_on_batch(x, y)

  def test_stacking_LSTM(self):
    inputs = np.random.random((2, 3, 4))
    targets = np.abs(np.random.random((2, 3, 5)))
    targets /= targets.sum(axis=-1, keepdims=True)
    model = keras.models.Sequential()
    model.add(rnn.LSTM(10, return_sequences=True, unroll=False))
    model.add(rnn.LSTM(5, return_sequences=True, unroll=False))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=gradient_descent.GradientDescentOptimizer(0.01))
    model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)

  def test_from_config_LSTM(self):
    layer_class = rnn.LSTM
    for stateful in (False, True):
      l1 = layer_class(units=1, stateful=stateful)
      l2 = layer_class.from_config(l1.get_config())
      assert l1.get_config() == l2.get_config()

  def test_specify_initial_state_keras_tensor(self):
    num_states = 2
    timesteps = 3
    embedding_dim = 4
    units = 3
    num_samples = 2

    # Test with Keras tensor
    inputs = keras.Input((timesteps, embedding_dim))
    initial_state = [keras.Input((units,)) for _ in range(num_states)]
    layer = rnn.LSTM(units)
    if len(initial_state) == 1:
      output = layer(inputs, initial_state=initial_state[0])
    else:
      output = layer(inputs, initial_state=initial_state)
    self.assertTrue(
        any(initial_state[0] is t
            for t in layer._inbound_nodes[0].input_tensors))

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

  def test_specify_initial_state_non_keras_tensor(self):
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
    layer = rnn.LSTM(units)
    output = layer(inputs, initial_state=initial_state)

    model = keras.models.Model(inputs, output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=gradient_descent.GradientDescentOptimizer(0.01))

    inputs = np.random.random((num_samples, timesteps, embedding_dim))
    targets = np.random.random((num_samples, units))
    model.train_on_batch(inputs, targets)

  def test_reset_states_with_values(self):
    num_states = 2
    timesteps = 3
    embedding_dim = 4
    units = 3
    num_samples = 2

    layer = rnn.LSTM(units, stateful=True)
    layer.build((num_samples, timesteps, embedding_dim))
    initial_weight_count = len(layer.weights)
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

    self.assertEqual(initial_weight_count, len(layer.weights))
    # Variables in "states" shouldn't show up in .weights
    layer.states = nest.map_structure(variables.Variable, values)
    layer.reset_states()
    self.assertEqual(initial_weight_count, len(layer.weights))

  def test_specify_state_with_masking(self):
    num_states = 2
    timesteps = 3
    embedding_dim = 4
    units = 3
    num_samples = 2

    inputs = keras.Input((timesteps, embedding_dim))
    _ = keras.layers.Masking()(inputs)
    initial_state = [keras.Input((units,)) for _ in range(num_states)]
    output = rnn.LSTM(units)(
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

  @test.disable_with_predicate(
      pred=test.is_built_with_rocm,
      skip_message='Skipping as ROCm MIOpen does not support padded input yet.')
  def test_return_state(self):
    num_states = 2
    timesteps = 3
    embedding_dim = 4
    units = 3
    num_samples = 2

    inputs = keras.Input(batch_shape=(num_samples, timesteps, embedding_dim))
    masked = keras.layers.Masking()(inputs)
    layer = rnn.LSTM(units, return_state=True, stateful=True)
    outputs = layer(masked)
    state = outputs[1:]
    assert len(state) == num_states
    model = keras.models.Model(inputs, state[0])

    inputs = np.random.random((num_samples, timesteps, embedding_dim))
    state = model.predict(inputs)
    self.assertAllClose(keras.backend.eval(layer.states[0]), state, atol=1e-4)

  def test_state_reuse(self):
    timesteps = 3
    embedding_dim = 4
    units = 3
    num_samples = 2

    inputs = keras.Input(batch_shape=(num_samples, timesteps, embedding_dim))
    layer = rnn.LSTM(
        units, return_state=True, return_sequences=True)
    outputs = layer(inputs)
    output, state = outputs[0], outputs[1:]
    output = rnn.LSTM(units)(output, initial_state=state)
    model = keras.models.Model(inputs, output)

    inputs = np.random.random((num_samples, timesteps, embedding_dim))
    model.predict(inputs)

  def test_initial_states_as_other_inputs(self):
    timesteps = 3
    embedding_dim = 4
    units = 3
    num_samples = 2
    num_states = 2
    layer_class = rnn.LSTM

    # Test with Keras tensor
    main_inputs = keras.Input((timesteps, embedding_dim))
    initial_state = [keras.Input((units,)) for _ in range(num_states)]
    inputs = [main_inputs] + initial_state

    layer = layer_class(units)
    output = layer(inputs)
    self.assertTrue(
        any(initial_state[0] is t
            for t in layer._inbound_nodes[0].input_tensors))

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

  @test.disable_with_predicate(
      pred=test.is_built_with_rocm,
      skip_message='Skipping as ROCm MIOpen does not support padded input yet.')
  @testing_utils.run_v2_only
  def test_lstm_v2_feature_parity_with_canonical_lstm(self):
    input_shape = 10
    rnn_state_size = 8
    timestep = 4
    batch = 20

    (x_train, y_train), _ = testing_utils.get_test_data(
        train_samples=batch,
        test_samples=0,
        input_shape=(timestep, input_shape),
        num_classes=rnn_state_size,
        random_seed=87654321)
    y_train = np_utils.to_categorical(y_train, rnn_state_size)
    # For the last batch item of the test data, we filter out the last
    # timestep to simulate the variable length sequence and masking test.
    x_train[-2:, -1, :] = 0.0
    y_train[-2:] = 0

    inputs = keras.layers.Input(
        shape=[timestep, input_shape], dtype=dtypes.float32)
    masked_input = keras.layers.Masking()(inputs)
    lstm_layer = rnn_v1.LSTM(rnn_state_size,
                             recurrent_activation='sigmoid')
    output = lstm_layer(masked_input)
    lstm_model = keras.models.Model(inputs, output)
    weights = lstm_model.get_weights()
    y_1 = lstm_model.predict(x_train)
    lstm_model.compile('rmsprop', 'mse')
    lstm_model.fit(x_train, y_train)
    y_2 = lstm_model.predict(x_train)

    with testing_utils.device(should_use_gpu=True):
      cudnn_layer = rnn.LSTM(rnn_state_size)
      cudnn_model = keras.models.Model(inputs, cudnn_layer(masked_input))
    cudnn_model.set_weights(weights)
    y_3 = cudnn_model.predict(x_train)
    cudnn_model.compile('rmsprop', 'mse')
    cudnn_model.fit(x_train, y_train)
    y_4 = cudnn_model.predict(x_train)

    self.assertAllClose(y_1, y_3, rtol=1e-5, atol=2e-5)
    self.assertAllClose(y_2, y_4, rtol=1e-5, atol=2e-5)

  @parameterized.named_parameters(('v0', 0), ('v1', 1), ('v2', 2))
  @test.disable_with_predicate(
      pred=test.is_built_with_rocm,
      skip_message='Skipping as ROCm MIOpen does not support padded input yet.')
  def test_implementation_mode_LSTM(self, implementation_mode):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    testing_utils.layer_test(
        rnn.LSTM,
        kwargs={
            'units': units,
            'implementation': implementation_mode
        },
        input_shape=(num_samples, timesteps, embedding_dim))

    layer_class = rnn.LSTM
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

    layer_class = rnn.LSTM
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

  @test.disable_with_predicate(
      pred=test.is_built_with_rocm,
      skip_message='Skipping as ROCm MIOpen does not support padded input yet.')
  def test_masking_with_stacking_LSTM(self):
    inputs = np.random.random((2, 3, 4))
    targets = np.abs(np.random.random((2, 3, 5)))
    targets /= targets.sum(axis=-1, keepdims=True)
    model = keras.models.Sequential()
    model.add(keras.layers.Masking(input_shape=(3, 4)))
    model.add(rnn.LSTM(10, return_sequences=True, unroll=False))
    model.add(rnn.LSTM(5, return_sequences=True, unroll=False))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=gradient_descent.GradientDescentOptimizer(0.01))
    model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)

  @parameterized.named_parameters(
      # test_name, time_major, go_backwards
      ('normal', False, False),
      ('time_major', True, False),
      ('go_backwards', False, True),
      ('both', True, True),
  )
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

    lstm_model = build_model(rnn_v1.LSTM)
    y_ref = lstm_model.predict(x_train)
    weights = lstm_model.get_weights()

    lstm_v2_model = build_model(rnn.LSTM)
    lstm_v2_model.set_weights(weights)
    y = lstm_v2_model.predict(x_train)

    self.assertAllClose(y, y_ref)

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
    y_train = np_utils.to_categorical(y_train, output_shape)

    layer = rnn.LSTM(rnn_state_size)

    inputs = keras.layers.Input(
        shape=[timestep, input_shape], dtype=dtypes.float32)

    outputs = layer(inputs)
    model = keras.models.Model(inputs, outputs)
    model.compile('rmsprop', loss='mse')
    model.fit(x_train, y_train, epochs=epoch)
    model.evaluate(x_train, y_train)
    model.predict(x_train)

  @parameterized.named_parameters(
      # test_name, use_bias, bias_initializer, activation
      ('normal', True, 'zeros'),
      ('no_bias', False, 'zeros'),
      ('random_bias', True, 'random_uniform'),
  )
  def test_lstm_model_save_load(self, use_bias, bias_initializer):
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
      layer = rnn.LSTM(
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

  def test_lstm_output_on_multiple_kernel(self):
    input_shape = 10
    rnn_state_size = 8
    timestep = 4
    batch = 100

    x_train = np.random.random((batch, timestep, input_shape))

    inputs = keras.layers.Input(
        shape=[timestep, input_shape], dtype=dtypes.float32)
    with testing_utils.device(should_use_gpu=False):
      layer = rnn.LSTM(rnn_state_size)
      output = layer(inputs)
      cpu_model = keras.models.Model(inputs, output)
      weights = cpu_model.get_weights()
    y_1 = cpu_model.predict(x_train)

    with testing_utils.device(should_use_gpu=True):
      layer = rnn.LSTM(rnn_state_size)
      output = layer(inputs)
      gpu_model = keras.models.Model(inputs, output)
      gpu_model.set_weights(weights)
    y_2 = gpu_model.predict(x_train)

    # Note that CuDNN uses 'sigmoid' as activation, so the LSTM V2 uses
    # 'sigmoid' as default. Construct the canonical LSTM with sigmoid to achieve
    # the same output.
    with testing_utils.device(should_use_gpu=True):
      layer = rnn_v1.LSTM(rnn_state_size, recurrent_activation='sigmoid')
      output = layer(inputs)
      canonical_model = keras.models.Model(inputs, output)
      # Remove the extra cudnn bias since canonical lstm will not use it.
      canonical_model.set_weights(weights[:3])
    y_3 = canonical_model.predict(x_train)

    self.assertAllClose(y_1, y_2)
    self.assertAllClose(y_2, y_3)

  def test_return_sequences_LSTM(self):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    testing_utils.layer_test(
        rnn.LSTM,
        kwargs={
            'units': units,
            'return_sequences': True
        },
        input_shape=(num_samples, timesteps, embedding_dim))

  @test.disable_with_predicate(
      pred=test.is_built_with_rocm,
      skip_message='Skipping as ROCm MIOpen does not support float64 yet.')
  @testing_utils.run_v2_only
  def test_float64_LSTM(self):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    testing_utils.layer_test(
        rnn.LSTM,
        kwargs={
            'units': units,
            'return_sequences': True,
            'dtype': 'float64'
        },
        input_shape=(num_samples, timesteps, embedding_dim),
        input_dtype='float64')

  def test_regularizers_LSTM(self):
    embedding_dim = 4
    layer_class = rnn.LSTM
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
    if context.executing_eagerly():
      self.assertEqual(len(layer.losses), 4)
    else:
      self.assertEqual(len(layer.get_losses_for(x)), 1)

  @test.disable_with_predicate(
      pred=test.is_built_with_rocm,
      skip_message='Skipping as ROCm MIOpen does not support padded input yet.')
  def test_statefulness_LSTM(self):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    layer_class = rnn.LSTM
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
        optimizer=gradient_descent.GradientDescentOptimizer(0.01),
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly())
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

    layer.reset_states()

    mix_padded_input = np.ones((num_samples, timesteps))
    mix_padded_input[0, 1] = 0
    mix_padded_input[1, 0] = 0
    mix_padded_input[1, 2] = 0
    out8 = model.predict(mix_padded_input)

    self.assertAllClose(out7, out6, atol=1e-5)
    self.assertAllClose(out8, out7, atol=1e-5)

  def test_stateful_LSTM_training(self):
    # See b/123587692 for more context.
    vocab_size = 20
    embedding_dim = 10
    batch_size = 8
    timestep = 12
    units = 5
    x = np.random.randint(0, vocab_size, size=(batch_size, timestep))
    y = np.random.randint(0, vocab_size, size=(batch_size, timestep))

    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim,
                               batch_input_shape=[batch_size, timestep]),
        rnn.LSTM(units, return_sequences=True, stateful=True),
        keras.layers.Dense(vocab_size)
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        run_eagerly=testing_utils.should_run_eagerly())
    model.fit(x, y, epochs=1, shuffle=False)

  def test_dropout_LSTM(self):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    testing_utils.layer_test(
        rnn.LSTM,
        kwargs={
            'units': units,
            'dropout': 0.1,
            'recurrent_dropout': 0.1
        },
        input_shape=(num_samples, timesteps, embedding_dim))

  def test_bidirectional(self):
    batch = 128
    timestep = 20
    vocab_size = 1000
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, 64),
        keras.layers.Bidirectional(rnn.LSTM(
            64, return_sequences=True)),
        keras.layers.Bidirectional(rnn.LSTM(32)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    x = np.random.randint(0, vocab_size, size=(batch, timestep))
    y = np.random.randint(0, 1, size=(batch))
    model.fit(x, y, epochs=1, shuffle=False)
    model.evaluate(x, y)
    model.predict(x)

  @test.disable_with_predicate(
      pred=test.is_built_with_rocm,
      skip_message='Skipping as ROCm MIOpen does not support padded input yet.')
  @testing_utils.run_v2_only
  def test_explicit_device_with_go_backward_and_mask(self):
    batch_size = 8
    timestep = 7
    masksteps = 5
    units = 4

    inputs = np.random.randn(batch_size, timestep, units).astype(np.float32)
    mask = np.ones((batch_size, timestep)).astype(np.bool)
    mask[:, masksteps:] = 0

    # Test for V1 behavior.
    lstm_v1 = rnn_v1.LSTM(units, return_sequences=True, go_backwards=True)
    with testing_utils.device(should_use_gpu=True):
      outputs_masked_v1 = lstm_v1(inputs, mask=constant_op.constant(mask))
      outputs_trimmed_v1 = lstm_v1(inputs[:, :masksteps])
    self.assertAllClose(outputs_masked_v1[:, -masksteps:], outputs_trimmed_v1)

    # Test for V2 behavior.
    lstm = rnn.LSTM(units, return_sequences=True, go_backwards=True)
    with testing_utils.device(should_use_gpu=True):
      outputs_masked = lstm(inputs, mask=constant_op.constant(mask))
      outputs_trimmed = lstm(inputs[:, :masksteps])
    self.assertAllClose(outputs_masked[:, -masksteps:], outputs_trimmed)

  @tf_test_util.enable_output_all_intermediates
  def test_v1_session_behavior(self):
    with ops.get_default_graph().as_default():
      # See b/139132348 for more details.
      x = np.random.uniform(size=(100, 4, 8))
      y = np.random.uniform(size=(100, 1))
      dataset = dataset_ops.Dataset.from_tensor_slices(
          (x, y)).shuffle(100).batch(32)

      inp = keras.layers.Input(shape=(4, 8))
      layer = rnn.LSTM(1)(inp)
      layer = keras.layers.Dense(1)(layer)

      model = keras.models.Model(inp, layer)

      model.compile(loss='mse', optimizer='sgd')
      model.fit(dataset)

  def test_with_fully_masked_inputs(self):
    num_samples = 8
    timestep = 5
    embedding_dim = 4
    vocab_size = 20
    units = 2

    inputs = np.random.randint(0, vocab_size, size=(num_samples, timestep))
    # Set the first inputs to be fully zero.
    inputs[0, :] = 0.0

    model = keras.models.Sequential()
    model.add(
        keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            mask_zero=True,
            input_length=timestep,
            batch_input_shape=(num_samples, timestep)))
    layer = rnn.LSTM(units)
    model.add(layer)
    model.compile(
        optimizer=gradient_descent.GradientDescentOptimizer(0.01),
        loss='mse',
        run_eagerly=testing_utils.should_run_eagerly())
    # Make sure it doesn't crash with cudnn kernel.
    model.predict(inputs)

  # TODO (b/169895267): test with xla_gpu is disabled.
  def test_deepcopy(self):
    if not context.executing_eagerly():
      self.skipTest('v2-only test')
    original_layer = rnn.LSTM(5)
    copied_layer = copy.deepcopy(original_layer)
    self.assertEqual(copied_layer.units, 5)
    self.assertEqual(original_layer.get_config(), original_layer.get_config())

    # Copy layer before layer call on inputs without weight initialization.
    inputs = np.random.normal(size=[32, 10, 8]).astype(np.float32)
    original_layer = rnn.LSTM(4)
    copied_layer = copy.deepcopy(original_layer)
    outputs = original_layer(inputs)
    copied_outputs = copied_layer(inputs)
    self.assertNotAllClose(
        self.evaluate(outputs), self.evaluate(copied_outputs))

    # Copy layer after layer call on inputs with weight initialization.
    original_layer = rnn.LSTM(4)
    outputs = original_layer(inputs)
    copied_layer = copy.deepcopy(original_layer)
    copied_outputs = copied_layer(inputs)
    self.assertAllClose(self.evaluate(outputs), self.evaluate(copied_outputs))


@keras_parameterized.run_all_keras_modes(config=_config)
class LSTMGraphRewriteTest(keras_parameterized.TestCase):

  input_shape = 10
  output_shape = 8
  rnn_state_size = 8
  timestep = 4
  batch = 100
  epoch = 1

  def _test_runtime_with_model(self, model):

    (x_train, y_train), _ = testing_utils.get_test_data(
        train_samples=self.batch,
        test_samples=0,
        input_shape=(self.timestep, self.input_shape),
        num_classes=self.output_shape)
    y_train = np_utils.to_categorical(y_train, self.output_shape)

    model.compile(
        optimizer='sgd',
        loss=['categorical_crossentropy', None],
        run_eagerly=testing_utils.should_run_eagerly())

    existing_loss = 0
    for _ in range(self.epoch):
      history = model.fit(x_train, y_train)
      loss_value = history.history['loss'][0]

      self.assertNotEqual(existing_loss, loss_value)
      existing_loss = loss_value

    _, runtime_value = model.predict(x_train)
    if test.is_gpu_available():
      self.assertEqual(runtime_value[0], rnn._RUNTIME_GPU)
    else:
      self.assertEqual(runtime_value[0], rnn._RUNTIME_CPU)

  @testing_utils.run_v2_only
  def test_LSTM_runtime(self):
    layer = rnn.LSTM(self.rnn_state_size, return_runtime=True)

    inputs = keras.layers.Input(
        shape=[self.timestep, self.input_shape], dtype=dtypes.float32)

    outputs, runtime = layer(inputs)
    # Expand the runtime so that it is a 1D tensor instead of scalar.
    # TF model does not work with scalar model output, specially during
    # aggregation.
    runtime = keras.layers.Lambda(
        lambda x: array_ops.expand_dims(x, axis=-1))(runtime)
    model = keras.models.Model(inputs=inputs, outputs=[outputs, runtime])
    self._test_runtime_with_model(model)

  @test.disable_with_predicate(
      pred=test.is_built_with_rocm,
      skip_message='Skipping as ROCm MIOpen does not support padded input yet.')
  @testing_utils.run_v2_only
  def test_LSTM_runtime_with_mask(self):
    # Masking will affect which backend is selected based on whether the mask
    # is strictly right padded.
    layer = rnn.LSTM(self.rnn_state_size, return_runtime=True)

    inputs = keras.layers.Input(
        shape=[self.timestep, self.input_shape], dtype=dtypes.float32)
    masked_inputs = keras.layers.Masking()(inputs)

    outputs, runtime = layer(masked_inputs)
    # Expand the runtime so that it is a 1D tensor instead of scalar.
    # TF model does not work with scalar model output, specially during
    # aggregation.
    runtime = keras.layers.Lambda(
        lambda x: array_ops.expand_dims(x, axis=-1))(runtime)
    model = keras.models.Model(inputs=inputs, outputs=[outputs, runtime])

    (x_train, y_train), _ = testing_utils.get_test_data(
        train_samples=self.batch,
        test_samples=0,
        input_shape=(self.timestep, self.input_shape),
        num_classes=self.output_shape)
    y_train = np_utils.to_categorical(y_train, self.output_shape)

    model.compile(
        optimizer='sgd',
        loss=['categorical_crossentropy', None],
        run_eagerly=testing_utils.should_run_eagerly())

    model.fit(x_train, y_train)

    # Verify unpadded data.
    _, runtime_value = model.predict(x_train)
    if test.is_gpu_available():
      self.assertEqual(runtime_value[0], rnn._RUNTIME_GPU)
    else:
      self.assertEqual(runtime_value[0], rnn._RUNTIME_CPU)

    # Update x/y to be right padded by setting the last timestep to 0
    x_train[:, -1, :] = 0
    y_train[:, -1] = 0
    _, runtime_value = model.predict(x_train)
    if test.is_gpu_available():
      self.assertEqual(runtime_value[0], rnn._RUNTIME_GPU)
    else:
      self.assertEqual(runtime_value[0], rnn._RUNTIME_CPU)

    # Further update x/y to be mix padded (masks in the middle), and verify
    # only cpu kernel can be selected.
    x_train[:, -3, :] = 0
    y_train[:, -3] = 0
    _, runtime_value = model.predict(x_train)
    self.assertEqual(runtime_value[0], rnn._RUNTIME_CPU)

  @testing_utils.run_v2_only
  def test_LSTM_runtime_with_cond(self):
    # This test is to demonstrate the graph rewrite of grappler plugin under
    # the condition that the function returns different number of internal
    # states.
    layer = rnn.LSTM(self.rnn_state_size, return_runtime=True)

    inputs = keras.layers.Input(
        shape=[self.timestep, self.input_shape], dtype=dtypes.float32)

    zeros = array_ops.zeros([self.batch, self.output_shape])
    dummy_runtime = rnn._runtime(rnn._RUNTIME_UNKNOWN)
    a = constant_op.constant(0)
    b = constant_op.constant(1)
    # Will always run the lstm layer.
    outputs, runtime = control_flow_ops.cond(
        gen_math_ops.less(a, b),
        lambda: layer(inputs),
        lambda: (zeros, dummy_runtime))

    # Expand the runtime so that it is a 1D tensor instead of scalar.
    # TF model does not work with scalar model output, specially during
    # aggregation.
    runtime = keras.layers.Lambda(
        lambda x: array_ops.expand_dims(x, axis=-1))(runtime)
    model = keras.models.Model(inputs=inputs, outputs=[outputs, runtime])
    self._test_runtime_with_model(model)


class LSTMPerformanceTest(test.Benchmark):

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
    # Get performance number for lstm_v2 with grappler swap the impl
    input_shape = test_config['input_shape']
    rnn_state_size = test_config['rnn_state_size']
    timestep = test_config['timestep']

    layer = rnn.LSTM(rnn_state_size)
    inputs = keras.layers.Input(
        shape=[timestep, input_shape], dtype=dtypes.float32)

    outputs = layer(inputs)
    model = keras.models.Model(inputs, outputs)
    model.compile('sgd', 'mse')

    sec_per_epoch = self._measure_performance(
        test_config, model, x_train, y_train)
    logging.info('Average performance for %s per epoch is: %s',
                 'LSTM V2', sec_per_epoch)
    return sec_per_epoch

  def _time_performance_run_normal_lstm(
      self, test_config, x_train, y_train):
    # Get performance number for standard LSTM on GPU.
    input_shape = test_config['input_shape']
    rnn_state_size = test_config['rnn_state_size']
    timestep = test_config['timestep']

    layer = rnn_v1.LSTM(rnn_state_size)
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
    y_train = np_utils.to_categorical(y_train, test_config['output_shape'])

    cudnn_sec_per_epoch = self._time_performance_run_cudnn_lstm(
        test_config, x_train, y_train)
    lstm_v2_sec_per_epoch = self._time_performance_run_unifed_lstm_gpu(
        test_config, x_train, y_train)
    normal_lstm_sec_per_epoch = self._time_performance_run_normal_lstm(
        test_config, x_train, y_train)

    cudnn_vs_v2 = cudnn_sec_per_epoch / lstm_v2_sec_per_epoch
    v2_vs_normal = normal_lstm_sec_per_epoch / lstm_v2_sec_per_epoch

    self.report_benchmark(name='keras_cudnn_lstm_' + mode,
                          wall_time=cudnn_sec_per_epoch,
                          iters=test_config['epoch'],
                          extras=test_config)
    self.report_benchmark(name='keras_lstm_v2_' + mode,
                          wall_time=lstm_v2_sec_per_epoch,
                          iters=test_config['epoch'],
                          extras=test_config)
    self.report_benchmark(name='keras_canonical_lstm_' + mode,
                          wall_time=normal_lstm_sec_per_epoch,
                          iters=test_config['epoch'],
                          extras=test_config)

    logging.info('Expect the performance of LSTM V2 is within 80% of '
                 'CuDNN LSTM, got {0:.2f}%'.format(cudnn_vs_v2 * 100))
    logging.info('Expect the performance of LSTM V2 is more than 5 times'
                 ' of normal LSTM, got {0:.2f}'.format(v2_vs_normal))

  def benchmark_performance_graph(self):
    with ops.get_default_graph().as_default():
      with session_lib.Session(config=_config):
        self._benchmark_performance_with_standard_cudnn_impl()

  def benchmark_performance_eager(self):
    with context.eager_mode():
      self._benchmark_performance_with_standard_cudnn_impl()


if __name__ == '__main__':
  test.main()
