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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.keras._impl import keras
from tensorflow.python.keras._impl.keras import testing_utils
from tensorflow.python.platform import test
from tensorflow.python.training.rmsprop import RMSPropOptimizer


class CuDNNTest(test.TestCase, parameterized.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def test_cudnn_rnn_timing(self):
    if test.is_gpu_available(cuda_only=True):
      with self.test_session(use_gpu=True):
        input_size = 10
        timesteps = 6
        units = 2
        num_samples = 32

        for rnn_type in ['lstm', 'gru']:
          times = []
          for use_cudnn in [True, False]:
            start_time = time.time()
            inputs = keras.layers.Input(shape=(None, input_size))
            if use_cudnn:
              if rnn_type == 'lstm':
                layer = keras.layers.CuDNNLSTM(units)
              else:
                layer = keras.layers.CuDNNGRU(units)
            else:
              if rnn_type == 'lstm':
                layer = keras.layers.LSTM(units)
              else:
                layer = keras.layers.GRU(units)
            outputs = layer(inputs)

            optimizer = RMSPropOptimizer(learning_rate=0.001)
            model = keras.models.Model(inputs, outputs)
            model.compile(optimizer, 'mse')

            x = np.random.random((num_samples, timesteps, input_size))
            y = np.random.random((num_samples, units))
            model.fit(x, y, epochs=4, batch_size=32)

            times.append(time.time() - start_time)
          self.assertGreater(times[1], times[0])

  @test_util.run_in_graph_and_eager_modes()
  def test_cudnn_rnn_basics(self):
    if test.is_gpu_available(cuda_only=True):
      with self.test_session(use_gpu=True):
        input_size = 10
        timesteps = 6
        units = 2
        num_samples = 32
        for layer_class in [keras.layers.CuDNNGRU, keras.layers.CuDNNLSTM]:
          for return_sequences in [True, False]:
            with keras.utils.CustomObjectScope(
                {'keras.layers.CuDNNGRU': keras.layers.CuDNNGRU,
                 'keras.layers.CuDNNLSTM': keras.layers.CuDNNLSTM}):
              testing_utils.layer_test(
                  layer_class,
                  kwargs={'units': units,
                          'return_sequences': return_sequences},
                  input_shape=(num_samples, timesteps, input_size))
          for go_backwards in [True, False]:
            with keras.utils.CustomObjectScope(
                {'keras.layers.CuDNNGRU': keras.layers.CuDNNGRU,
                 'keras.layers.CuDNNLSTM': keras.layers.CuDNNLSTM}):
              testing_utils.layer_test(
                  layer_class,
                  kwargs={'units': units,
                          'go_backwards': go_backwards},
                  input_shape=(num_samples, timesteps, input_size))

  @test_util.run_in_graph_and_eager_modes()
  def test_trainability(self):
    if test.is_gpu_available(cuda_only=True):
      with self.test_session(use_gpu=True):
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
      ('cudnngru', keras.layers.CuDNNGRU),
      ('cudnnlstm', keras.layers.CuDNNLSTM),
  )
  def test_regularizer(self, layer_class):
    if test.is_gpu_available(cuda_only=True):
      with self.test_session(use_gpu=True):
        input_size = 10
        timesteps = 6
        units = 2
        num_samples = 32
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
  def test_return_state(self, layer_class):
    if test.is_gpu_available(cuda_only=True):
      with self.test_session(use_gpu=True):
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

        inputs = np.random.random((num_samples, timesteps, input_size))
        state = model.predict(inputs)
        np.testing.assert_allclose(
            keras.backend.eval(layer.states[0]), state, atol=1e-4)

  @parameterized.named_parameters(
      ('cudnngru', keras.layers.CuDNNGRU),
      ('cudnnlstm', keras.layers.CuDNNLSTM),
  )
  def test_specify_initial_state_keras_tensor(self, layer_class):
    if test.is_gpu_available(cuda_only=True):
      with self.test_session(use_gpu=True):
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
        self.assertIn(initial_state[0], layer._inbound_nodes[0].input_tensors)

        model = keras.models.Model([inputs] + initial_state, output)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        inputs = np.random.random((num_samples, timesteps, input_size))
        initial_state = [
            np.random.random((num_samples, units)) for _ in range(num_states)
        ]
        targets = np.random.random((num_samples, units))
        model.fit([inputs] + initial_state, targets)

  @parameterized.named_parameters(
      ('cudnngru', keras.layers.CuDNNGRU),
      ('cudnnlstm', keras.layers.CuDNNLSTM),
  )
  def test_statefulness(self, layer_class):
    if test.is_gpu_available(cuda_only=True):
      with self.test_session(use_gpu=True):
        input_size = 10
        timesteps = 6
        units = 2
        num_samples = 32

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
        model.compile(optimizer='sgd', loss='mse')
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

  # TODO(psv): Add generic cross product helper function for parametrized tests.
  @parameterized.named_parameters(
      ('cudnnlstm_to_lstm_unidirectional_impl_1', 'LSTM', False, False, 1),
      ('cudnnlstm_to_lstm_bidirectional_impl_1', 'LSTM', False, True, 1),
      ('lstm_to_cudnnlstm_unidirectional_impl_1', 'LSTM', True, False, 1),
      ('lstm_to_cudnnlstm_bidirectional_impl_1', 'LSTM', True, True, 1),
      ('cudnngru_to_gru_unidirectional_impl_1', 'GRU', False, False, 1),
      ('cudnngru_to_gru_bidirectional_impl_1', 'GRU', False, True, 1),
      ('gru_to_cudnngru_unidirectional_impl_1', 'GRU', True, False, 1),
      ('gru_to_cudnngru_bidirectional_impl_1', 'GRU', True, True, 1),
      ('cudnnlstm_to_lstm_unidirectional_impl_2', 'LSTM', False, False, 2),
      ('cudnnlstm_to_lstm_bidirectional_impl_2', 'LSTM', False, True, 2),
      ('lstm_to_cudnnlstm_unidirectional_impl_2', 'LSTM', True, False, 2),
      ('lstm_to_cudnnlstm_bidirectional_impl_2', 'LSTM', True, True, 2),
      ('cudnngru_to_gru_unidirectional_impl_2', 'GRU', False, False, 2),
      ('cudnngru_to_gru_bidirectional_impl_2', 'GRU', False, True, 2),
      ('gru_to_cudnngru_unidirectional_impl_2', 'GRU', True, False, 2),
      ('gru_to_cudnngru_bidirectional_impl_2', 'GRU', True, True, 2),
  )
  def test_load_weights_between_noncudnn_rnn(self, rnn_type, to_cudnn,
                                             bidirectional, implementation):
    if test.is_gpu_available(cuda_only=True):
      with self.test_session(use_gpu=True):
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

        def convert_weights(source_layer, target_layer):
          weights = source_layer.get_weights()
          weights = keras.engine.saving.preprocess_weights_for_loading(
              target_layer, weights)
          target_layer.set_weights(weights)

        input_layer = keras.layers.InputLayer(input_shape)

        layer = rnn_layer_class(units, **rnn_layer_kwargs)
        if bidirectional:
          layer = keras.layers.Bidirectional(layer)

        cudnn_layer = cudnn_rnn_layer_class(units)
        if bidirectional:
          cudnn_layer = keras.layers.Bidirectional(cudnn_layer)

        model = keras.models.Sequential([input_layer, layer])
        cudnn_model = keras.models.Sequential([input_layer, cudnn_layer])

        if to_cudnn:
          convert_weights(layer, cudnn_layer)
        else:
          convert_weights(cudnn_layer, layer)

        self.assertAllClose(
            model.predict(inputs), cudnn_model.predict(inputs), atol=1e-4)

  @test_util.run_in_graph_and_eager_modes()
  def test_cudnnrnn_bidirectional(self):
    if test.is_gpu_available(cuda_only=True):
      with self.test_session(use_gpu=True):
        rnn = keras.layers.CuDNNGRU
        samples = 2
        dim = 2
        timesteps = 2
        output_dim = 2
        mode = 'concat'

        x = np.random.random((samples, timesteps, dim))
        target_dim = 2 * output_dim if mode == 'concat' else output_dim
        y = np.random.random((samples, target_dim))

        # test with Sequential model
        model = keras.Sequential()
        model.add(
            keras.layers.Bidirectional(
                rnn(output_dim), merge_mode=mode, input_shape=(None, dim)))
        model.compile(
            loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))
        model.fit(x, y, epochs=1, batch_size=1)

        # test config
        model.get_config()
        model = keras.models.model_from_json(model.to_json())
        model.summary()

        # test stacked bidirectional layers
        model = keras.Sequential()
        model.add(
            keras.layers.Bidirectional(
                rnn(output_dim, return_sequences=True),
                merge_mode=mode,
                input_shape=(None, dim)))
        model.add(keras.layers.Bidirectional(rnn(output_dim), merge_mode=mode))
        model.compile(
            loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))
        model.fit(x, y, epochs=1, batch_size=1)

        # test with functional API
        inputs = keras.Input((timesteps, dim))
        outputs = keras.layers.Bidirectional(
            rnn(output_dim), merge_mode=mode)(
                inputs)
        model = keras.Model(inputs, outputs)
        model.compile(
            loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))
        model.fit(x, y, epochs=1, batch_size=1)

        # Bidirectional and stateful
        inputs = keras.Input(batch_shape=(1, timesteps, dim))
        outputs = keras.layers.Bidirectional(
            rnn(output_dim, stateful=True), merge_mode=mode)(
                inputs)
        model = keras.Model(inputs, outputs)
        model.compile(
            loss='mse', optimizer=RMSPropOptimizer(learning_rate=0.001))
        model.fit(x, y, epochs=1, batch_size=1)

  def test_preprocess_weights_for_loading_gru_incompatible(self):
    """Test loading weights between incompatible layers.

    Should fail fast with an exception.
    """
    if test.is_gpu_available(cuda_only=True):
      with self.test_session(use_gpu=True):
        input_shape = (3, 5)

        def gru(cudnn=False, **kwargs):
          layer_class = keras.layers.CuDNNGRU if cudnn else keras.layers.GRU
          return layer_class(2, input_shape=input_shape, **kwargs)

        def get_layer_weights(layer):
          layer.build(input_shape=input_shape)
          return layer.get_weights()

        def assert_not_compatible(src, dest, message):
          with self.assertRaises(ValueError) as ex:
            keras.engine.saving.preprocess_weights_for_loading(
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
