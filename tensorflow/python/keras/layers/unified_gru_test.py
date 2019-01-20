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
"""Tests for UnifiedGRU layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import keras
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


# Global config for grappler setting that is used for graph mode test.
_rewrites = rewriter_config_pb2.RewriterConfig()
_rewrites.function_optimization = rewriter_config_pb2.RewriterConfig.OFF
_customer_optimizer = _rewrites.custom_optimizers.add()
_customer_optimizer.name = 'ExperimentalImplementationSelector'
_rewrites.min_graph_nodes = -1
_graph_options = config_pb2.GraphOptions(rewrite_options=_rewrites)
_config = config_pb2.ConfigProto(graph_options=_graph_options)


@keras_parameterized.run_all_keras_modes(config=_config)
class UnifiedGRUTest(keras_parameterized.TestCase):

  @parameterized.named_parameters(
      ('non_tan_activation', 'relu', 'sigmoid', 0, False, True, True),
      ('non_sigmoid_recur_activation', 'tanh', 'relu', 0, False, True, True),
      ('use_recurrent_dropout', 'tanh', 'sigmoid', 0.1, False, True, True),
      ('unroll', 'tanh', 'sigmoid', 0, True, True, True),
      ('not_use_bias', 'tanh', 'sigmoid', 0, False, False, True),
      ('not_reset_after', 'tanh', 'sigmoid', 0, False, True, False)
  )
  def test_could_use_defun_backend(self, activation, recurrent_activation,
                                   recurrent_dropout, unroll, use_bias,
                                   reset_after):
    layer = keras.layers.UnifiedGRU(1,
                                    activation=activation,
                                    recurrent_activation=recurrent_activation,
                                    recurrent_dropout=recurrent_dropout,
                                    unroll=unroll,
                                    use_bias=use_bias,
                                    reset_after=reset_after)
    self.assertFalse(layer.could_use_cudnn)

  def test_keras_model_with_gru(self):
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

    layer = keras.layers.UnifiedGRU(rnn_state_size)

    inputs = keras.layers.Input(
        shape=[timestep, input_shape], dtype=dtypes.float32)

    outputs = layer(inputs)
    model = keras.models.Model(inputs, outputs)
    model.compile('rmsprop', loss='mse')
    model.fit(x_train, y_train, epochs=epoch)
    model.evaluate(x_train, y_train)
    model.predict(x_train)

  def test_dynamic_behavior_GRU(self):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    layer = keras.layers.UnifiedGRU(units, input_shape=(None, embedding_dim))
    model = keras.models.Sequential()
    model.add(layer)
    model.compile(gradient_descent.GradientDescentOptimizer(0.001), 'mse')
    x = np.random.random((num_samples, timesteps, embedding_dim))
    y = np.random.random((num_samples, units))
    model.train_on_batch(x, y)

  def test_stacking_GRU(self):
    inputs = np.random.random((2, 3, 4))
    targets = np.abs(np.random.random((2, 3, 5)))
    targets /= targets.sum(axis=-1, keepdims=True)
    model = keras.models.Sequential()
    model.add(keras.layers.UnifiedGRU(10, return_sequences=True, unroll=False))
    model.add(keras.layers.UnifiedGRU(5, return_sequences=True, unroll=False))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=gradient_descent.GradientDescentOptimizer(0.01))
    model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)

  def test_from_config_GRU(self):
    layer_class = keras.layers.UnifiedGRU
    for stateful in (False, True):
      l1 = layer_class(units=1, stateful=stateful)
      l2 = layer_class.from_config(l1.get_config())
      assert l1.get_config() == l2.get_config()

  def test_unified_gru_feature_parity_with_canonical_gru(self):
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
      gru_layer = keras.layers.GRU(rnn_state_size,
                                   recurrent_activation='sigmoid',
                                   reset_after=True)
      output = gru_layer(inputs)
      gru_model = keras.models.Model(inputs, output)
      weights = gru_model.get_weights()
      y_1 = gru_model.predict(x_train)
      gru_model.compile('rmsprop', 'mse')
      gru_model.fit(x_train, y_train)
      y_2 = gru_model.predict(x_train)

      with test_util.device(use_gpu=True):
        cudnn_layer = keras.layers.UnifiedGRU(rnn_state_size,
                                              recurrent_activation='sigmoid',
                                              reset_after=True)
        cudnn_model = keras.models.Model(inputs, cudnn_layer(inputs))
      cudnn_model.set_weights(weights)
      y_3 = cudnn_model.predict(x_train)
      cudnn_model.compile('rmsprop', 'mse')
      cudnn_model.fit(x_train, y_train)
      y_4 = cudnn_model.predict(x_train)

      self.assertAllClose(y_1, y_3, rtol=1e-5, atol=1e-5)
      self.assertAllClose(y_2, y_4, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(
      # test_name, use_bias, bias_initializer, activation
      ('normal', True, 'zeros'),
      ('no_bias', False, 'zeros'),
      ('random_bias', True, 'random_uniform'),
  )
  def test_unified_gru_model_save_load(self, use_bias, bias_initializer):
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
      layer = keras.layers.UnifiedGRU(
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

  def test_unified_gru_output_on_multiple_kernel(self):
    input_shape = 10
    rnn_state_size = 8
    timestep = 4
    batch = 100

    x_train = np.random.random((batch, timestep, input_shape))

    inputs = keras.layers.Input(
        shape=[timestep, input_shape], dtype=dtypes.float32)
    with test_util.device(use_gpu=False):
      layer = keras.layers.UnifiedGRU(rnn_state_size)
      output = layer(inputs)
      cpu_model = keras.models.Model(inputs, output)
      weights = cpu_model.get_weights()
      y_1 = cpu_model.predict(x_train)

    with test_util.device(use_gpu=True):
      layer = keras.layers.UnifiedGRU(rnn_state_size)
      output = layer(inputs)
      gpu_model = keras.models.Model(inputs, output)
      gpu_model.set_weights(weights)
      y_2 = gpu_model.predict(x_train)

    # Note that CuDNN uses 'sigmoid' as activation, so the unified GRU uses
    # 'sigmoid' as default. Construct the canonical GRU with sigmoid to achieve
    # the same output.
    with test_util.device(use_gpu=True):
      layer = keras.layers.GRU(rnn_state_size,
                               recurrent_activation='sigmoid',
                               reset_after=True)
      output = layer(inputs)
      canonical_model = keras.models.Model(inputs, output)
      canonical_model.set_weights(weights)
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
                        go_backwards=go_backwards,
                        reset_after=True)
      if time_major:
        converted_input = keras.layers.Lambda(
            lambda t: array_ops.transpose(t, [1, 0, 2]))(inputs)
        outputs = layer(converted_input)
        outputs = keras.layers.Lambda(
            lambda t: array_ops.transpose(t, [1, 0, 2]))(outputs)
      else:
        outputs = layer(inputs)
      return keras.models.Model(inputs, outputs)

    gru_model = build_model(keras.layers.GRU)
    y_ref = gru_model.predict(x_train)
    weights = gru_model.get_weights()

    unified_gru_model = build_model(keras.layers.UnifiedGRU)
    unified_gru_model.set_weights(weights)
    y = unified_gru_model.predict(x_train)

    self.assertAllClose(y, y_ref)

  def test_with_masking_layer_GRU(self):
    layer_class = keras.layers.UnifiedGRU
    inputs = np.random.random((2, 3, 4))
    targets = np.abs(np.random.random((2, 3, 5)))
    targets /= targets.sum(axis=-1, keepdims=True)
    model = keras.models.Sequential()
    model.add(keras.layers.Masking(input_shape=(3, 4)))
    model.add(layer_class(units=5, return_sequences=True, unroll=False))
    model.compile(loss='categorical_crossentropy',
                  optimizer=gradient_descent.GradientDescentOptimizer(0.001))
    model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)

  def test_masking_with_stacking_GRU(self):
    inputs = np.random.random((2, 3, 4))
    targets = np.abs(np.random.random((2, 3, 5)))
    targets /= targets.sum(axis=-1, keepdims=True)
    model = keras.models.Sequential()
    model.add(keras.layers.Masking(input_shape=(3, 4)))
    model.add(keras.layers.UnifiedGRU(10, return_sequences=True, unroll=False))
    model.add(keras.layers.UnifiedGRU(5, return_sequences=True, unroll=False))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=gradient_descent.GradientDescentOptimizer(0.01))
    model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)

  def test_return_sequences_GRU(self):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    testing_utils.layer_test(
        keras.layers.UnifiedGRU,
        kwargs={'units': units,
                'return_sequences': True},
        input_shape=(num_samples, timesteps, embedding_dim))

  def test_dropout_GRU(self):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    testing_utils.layer_test(
        keras.layers.UnifiedGRU,
        kwargs={'units': units,
                'dropout': 0.1,
                'recurrent_dropout': 0.1},
        input_shape=(num_samples, timesteps, embedding_dim))

  def test_constraints_GRU(self):
    embedding_dim = 4
    layer_class = keras.layers.UnifiedGRU
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

  @parameterized.parameters([0, 1, 2])
  def test_implementation_mode_GRU(self, implementation_mode):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    testing_utils.layer_test(
        keras.layers.UnifiedGRU,
        kwargs={'units': units,
                'implementation': implementation_mode},
        input_shape=(num_samples, timesteps, embedding_dim))

  def test_regularizers_GRU(self):
    embedding_dim = 4
    layer_class = keras.layers.UnifiedGRU
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

  def test_statefulness_GRU(self):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    layer_class = keras.layers.UnifiedGRU
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
    model.compile(optimizer=gradient_descent.GradientDescentOptimizer(0.01),
                  loss='mse', run_eagerly=testing_utils.should_run_eagerly())
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
    np.testing.assert_allclose(out3, out4, atol=1e-5)

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

    np.testing.assert_allclose(out7, out6, atol=1e-5)


class GRULayerGradientTapeTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes(config=_config)
  def test_in_tape(self):
    if not context.executing_eagerly():
      self.skipTest('bloo')
    time_steps = 10
    embedding_size = 11
    gru_unit_size = 12

    gru = keras.layers.UnifiedGRU(gru_unit_size,
                                  return_sequences=True,
                                  return_state=True,
                                  recurrent_activation='sigmoid',
                                  recurrent_initializer='glorot_uniform')

    x = random_ops.random_uniform([1, time_steps, embedding_size])
    y = random_ops.random_uniform([1, gru_unit_size])

    with backprop.GradientTape() as tape:
      hidden_state = array_ops.zeros([1, gru_unit_size], dtype=dtypes.float32)
      _, state = gru(x, initial_state=hidden_state)

      loss = math_ops.reduce_mean(math_ops.square(state - y))

    tape.gradient(loss, gru.variables)


class GRULayerGraphOnlyTest(test.TestCase):

  # Need session for test
  @test_util.run_deprecated_v1
  def test_unifiedGRU(self):
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

      layer = keras.layers.UnifiedGRU(rnn_state_size, return_runtime=True)

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

  # Need session for test
  @test_util.run_deprecated_v1
  def test_UnifiedGRU_with_cond(self):
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

      layer = keras.layers.UnifiedGRU(rnn_state_size, return_runtime=True)

      inputs = array_ops.placeholder(
          dtypes.float32, shape=(None, timestep, input_shape), name='inputs')
      predict = array_ops.placeholder(
          dtypes.float32, shape=(None, output_shape), name='predict')

      zeros = array_ops.zeros([batch, output_shape])
      dummy_runtime = constant_op.constant(
          'unknown', dtype=dtypes.string, name='runtime')
      a = constant_op.constant(0)
      b = constant_op.constant(1)
      # Will always run the GRU layer.
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


if __name__ == '__main__':
  test.main()
