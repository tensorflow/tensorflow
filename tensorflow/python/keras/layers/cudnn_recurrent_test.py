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
"""Tests for cudnn recurrent layers."""

import os
import tempfile

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


@keras_parameterized.run_all_keras_modes
class CuDNNTest(keras_parameterized.TestCase):

  @parameterized.named_parameters(
      *testing_utils.generate_combinations_with_testcase_name(
          layer_class=[keras.layers.CuDNNGRU, keras.layers.CuDNNLSTM],
          return_sequences=[True, False]))
  @test_util.run_gpu_only
  def test_cudnn_rnn_return_sequence(self, layer_class, return_sequences):
    input_size = 10
    timesteps = 6
    units = 2
    num_samples = 32
    testing_utils.layer_test(
        layer_class,
        kwargs={'units': units,
                'return_sequences': return_sequences},
        input_shape=(num_samples, timesteps, input_size))

  @parameterized.named_parameters(
      *testing_utils.generate_combinations_with_testcase_name(
          layer_class=[keras.layers.CuDNNGRU, keras.layers.CuDNNLSTM],
          go_backwards=[True, False]))
  @test_util.run_gpu_only
  def test_cudnn_rnn_go_backward(self, layer_class, go_backwards):
    input_size = 10
    timesteps = 6
    units = 2
    num_samples = 32
    testing_utils.layer_test(
        layer_class,
        kwargs={'units': units,
                'go_backwards': go_backwards},
        input_shape=(num_samples, timesteps, input_size))

  @parameterized.named_parameters(
      ('cudnngru', keras.layers.CuDNNGRU),
      ('cudnnlstm', keras.layers.CuDNNLSTM),
  )
  @test_util.run_gpu_only
  def test_return_state(self, layer_class):
    input_size = 10
    timesteps = 6
    units = 2
    num_samples = 32
    num_states = 2 if layer_class is keras.layers.CuDNNLSTM else 1

    inputs = keras.Input(batch_shape=(num_samples, timesteps, input_size))
    layer = layer_class(units, return_state=True, stateful=True)
    outputs = layer(inputs)
    _, state = outputs[0], outputs[1:]
    self.assertEqual(len(state), num_states)
    model = keras.models.Model(inputs, state[0])
    model.run_eagerly = testing_utils.should_run_eagerly()

    inputs = np.random.random((num_samples, timesteps, input_size))
    state = model.predict(inputs)
    np.testing.assert_allclose(
        keras.backend.eval(layer.states[0]), state, atol=1e-4)

  @parameterized.named_parameters(
      ('cudnngru', keras.layers.CuDNNGRU),
      ('cudnnlstm', keras.layers.CuDNNLSTM),
  )
  @test_util.run_gpu_only
  def test_time_major_input(self, layer_class):
    input_size = 10
    timesteps = 6
    units = 2
    num_samples = 32

    model = keras.models.Sequential()
    model.add(
        keras.layers.Lambda(lambda t: array_ops.transpose(t, [1, 0, 2])))
    layer = layer_class(units, time_major=True, return_sequences=True)
    model.add(layer)
    model.add(
        keras.layers.Lambda(lambda t: array_ops.transpose(t, [1, 0, 2])))
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(learning_rate=0.001))
    model.fit(
        np.ones((num_samples, timesteps, input_size)),
        np.ones((num_samples, timesteps, units)))
    out = model.predict(np.ones((num_samples, timesteps, input_size)))
    self.assertEqual(out.shape, (num_samples, timesteps, units))

  @parameterized.named_parameters(
      ('cudnngru', keras.layers.CuDNNGRU),
      ('cudnnlstm', keras.layers.CuDNNLSTM),
  )
  @test_util.run_gpu_only
  def test_specify_initial_state_keras_tensor(self, layer_class):
    input_size = 10
    timesteps = 6
    units = 2
    num_samples = 32
    num_states = 2 if layer_class is keras.layers.CuDNNLSTM else 1

    inputs = keras.Input((timesteps, input_size))
    initial_state = [keras.Input((units,)) for _ in range(num_states)]
    layer = layer_class(units)
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
        optimizer=RMSprop(learning_rate=0.001),
        run_eagerly=testing_utils.should_run_eagerly())

    inputs = np.random.random((num_samples, timesteps, input_size))
    initial_state = [
        np.random.random((num_samples, units)) for _ in range(num_states)
    ]
    targets = np.random.random((num_samples, units))
    model.fit([inputs] + initial_state, targets)


class CuDNNGraphOnlyTest(keras_parameterized.TestCase):

  @parameterized.named_parameters(
      ('cudnngru', keras.layers.CuDNNGRU),
      ('cudnnlstm', keras.layers.CuDNNLSTM),
  )
  @test_util.run_gpu_only
  def test_regularizer(self, layer_class):
    input_size = 10
    timesteps = 6
    units = 2
    num_samples = 32
    with ops.Graph().as_default():
      layer = layer_class(
          units,
          return_sequences=False,
          input_shape=(timesteps, input_size),
          kernel_regularizer=keras.regularizers.l1(0.01),
          recurrent_regularizer=keras.regularizers.l1(0.01),
          bias_regularizer='l2')
      layer.build((None, None, input_size))
      self.assertEqual(len(layer.losses), 3)

      layer = layer_class(
          units,
          return_sequences=False,
          input_shape=(timesteps, input_size),
          activity_regularizer='l2')
      self.assertTrue(layer.activity_regularizer)
      x = keras.backend.variable(
          np.ones((num_samples, timesteps, input_size)))
      layer(x)
      self.assertEqual(len(layer.get_losses_for(x)), 1)

  @parameterized.named_parameters(
      ('cudnngru', keras.layers.CuDNNGRU),
      ('cudnnlstm', keras.layers.CuDNNLSTM),
  )
  @test_util.run_gpu_only
  @test_util.run_v1_only('b/120941292')
  def test_statefulness(self, layer_class):
    input_size = 10
    timesteps = 6
    units = 2
    num_samples = 32

    with self.cached_session():
      model = keras.models.Sequential()
      model.add(
          keras.layers.Embedding(
              10,
              input_size,
              input_length=timesteps,
              batch_input_shape=(num_samples, timesteps)))
      layer = layer_class(
          units, return_sequences=False, stateful=True, weights=None)
      model.add(layer)
      model.compile(optimizer=gradient_descent.GradientDescentOptimizer(0.01),
                    loss='mse')
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


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class CuDNNV1OnlyTest(keras_parameterized.TestCase):

  @test_util.run_gpu_only
  def test_trainability(self):
    input_size = 10
    units = 2
    for layer_class in [keras.layers.CuDNNGRU, keras.layers.CuDNNLSTM]:
      layer = layer_class(units)
      layer.build((None, None, input_size))
      self.assertEqual(len(layer.weights), 3)
      self.assertEqual(len(layer.trainable_weights), 3)
      self.assertEqual(len(layer.non_trainable_weights), 0)
      layer.trainable = False
      self.assertEqual(len(layer.weights), 3)
      self.assertEqual(len(layer.non_trainable_weights), 3)
      self.assertEqual(len(layer.trainable_weights), 0)
      layer.trainable = True
      self.assertEqual(len(layer.weights), 3)
      self.assertEqual(len(layer.trainable_weights), 3)
      self.assertEqual(len(layer.non_trainable_weights), 0)

  @parameterized.named_parameters(
      *testing_utils.generate_combinations_with_testcase_name(
          rnn_type=['LSTM', 'GRU'], to_cudnn=[True, False],
          bidirectional=[False], implementation=[1, 2],
          model_nest_level=[1, 2], model_type=['seq', 'func']))
  @test_util.run_v1_only('b/120911602, b/112083752')
  @test_util.run_gpu_only
  def test_load_weights_between_noncudnn_rnn(self, rnn_type, to_cudnn,
                                             bidirectional, implementation,
                                             model_nest_level, model_type):
    input_size = 10
    timesteps = 6
    input_shape = (timesteps, input_size)
    units = 2
    num_samples = 32
    inputs = np.random.random((num_samples, timesteps, input_size))

    rnn_layer_kwargs = {
        'recurrent_activation': 'sigmoid',
        # ensure biases are non-zero and properly converted
        'bias_initializer': 'random_uniform',
        'implementation': implementation
    }
    if rnn_type == 'LSTM':
      rnn_layer_class = keras.layers.LSTM
      cudnn_rnn_layer_class = keras.layers.CuDNNLSTM
    else:
      rnn_layer_class = keras.layers.GRU
      cudnn_rnn_layer_class = keras.layers.CuDNNGRU
      rnn_layer_kwargs['reset_after'] = True

    layer = rnn_layer_class(units, **rnn_layer_kwargs)
    cudnn_layer = cudnn_rnn_layer_class(units)

    model = self._make_nested_model(input_shape, layer, model_nest_level,
                                    model_type)
    cudnn_model = self._make_nested_model(input_shape, cudnn_layer,
                                          model_nest_level, model_type)

    if to_cudnn:
      self._convert_model_weights(model, cudnn_model)
    else:
      self._convert_model_weights(cudnn_model, model)

    self.assertAllClose(model.predict(inputs), cudnn_model.predict(inputs),
                        atol=1e-4)

  def _make_nested_model(self, input_shape, layer, level=1, model_type='func'):
    # example: make_nested_seq_model((1,), Dense(10), level=2).summary()
    def make_nested_seq_model(input_shape, layer, level=1):
      model = layer
      for i in range(1, level + 1):
        layers = [keras.layers.InputLayer(input_shape),
                  model] if (i == 1) else [model]
        model = keras.models.Sequential(layers)
        if i > 1:
          model.build((None,) + input_shape)
      return model

    # example: make_nested_func_model((1,), Dense(10), level=2).summary()
    def make_nested_func_model(input_shape, layer, level=1):
      model_input = keras.layers.Input(input_shape)
      model = layer
      for _ in range(level):
        model = keras.models.Model(model_input, model(model_input))
      return model

    if model_type == 'func':
      return make_nested_func_model(input_shape, layer, level)
    elif model_type == 'seq':
      return make_nested_seq_model(input_shape, layer, level)

  def _convert_model_weights(self, source_model, target_model):
    _, fname = tempfile.mkstemp('.h5')
    source_model.save_weights(fname)
    target_model.load_weights(fname)
    os.remove(fname)

  @test_util.run_gpu_only
  def test_preprocess_weights_for_loading_gru_incompatible(self):
    """Test loading weights between incompatible layers.

    Should fail fast with an exception.
    """
    input_shape = (3, 5)

    def gru(cudnn=False, **kwargs):
      layer_class = keras.layers.CuDNNGRU if cudnn else keras.layers.GRUV1
      return layer_class(2, input_shape=input_shape, **kwargs)

    def get_layer_weights(layer):
      layer.build(input_shape=input_shape)
      return layer.get_weights()

    def assert_not_compatible(src, dest, message):
      with self.assertRaises(ValueError) as ex:
        keras.saving.hdf5_format.preprocess_weights_for_loading(
            dest,
            get_layer_weights(src))
      self.assertIn(message, str(ex.exception))

    assert_not_compatible(
        gru(),
        gru(cudnn=True),
        'GRU(reset_after=False) is not compatible with CuDNNGRU')
    assert_not_compatible(
        gru(cudnn=True),
        gru(),
        'CuDNNGRU is not compatible with GRU(reset_after=False)')
    assert_not_compatible(
        gru(),
        gru(reset_after=True),
        'GRU(reset_after=False) is not compatible with '
        'GRU(reset_after=True)')
    assert_not_compatible(
        gru(reset_after=True),
        gru(),
        'GRU(reset_after=True) is not compatible with '
        'GRU(reset_after=False)')


if __name__ == '__main__':
  test.main()
