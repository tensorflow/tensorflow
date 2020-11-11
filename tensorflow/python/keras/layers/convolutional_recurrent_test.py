# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for convolutional recurrent layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes
class ConvLSTMTest(keras_parameterized.TestCase):

  @parameterized.named_parameters(
      *testing_utils.generate_combinations_with_testcase_name(
          data_format=['channels_first', 'channels_last'],
          return_sequences=[True, False]))
  def test_conv_lstm(self, data_format, return_sequences):
    num_row = 3
    num_col = 3
    filters = 2
    num_samples = 1
    input_channel = 2
    input_num_row = 5
    input_num_col = 5
    sequence_len = 2
    if data_format == 'channels_first':
      inputs = np.random.rand(num_samples, sequence_len,
                              input_channel,
                              input_num_row, input_num_col)
    else:
      inputs = np.random.rand(num_samples, sequence_len,
                              input_num_row, input_num_col,
                              input_channel)

    # test for return state:
    x = keras.Input(batch_shape=inputs.shape)
    kwargs = {'data_format': data_format,
              'return_sequences': return_sequences,
              'return_state': True,
              'stateful': True,
              'filters': filters,
              'kernel_size': (num_row, num_col),
              'padding': 'valid'}
    layer = keras.layers.ConvLSTM2D(**kwargs)
    layer.build(inputs.shape)
    outputs = layer(x)
    _, states = outputs[0], outputs[1:]
    self.assertEqual(len(states), 2)
    model = keras.models.Model(x, states[0])
    state = model.predict(inputs)

    self.assertAllClose(
        keras.backend.eval(layer.states[0]), state, atol=1e-4)

    # test for output shape:
    testing_utils.layer_test(
        keras.layers.ConvLSTM2D,
        kwargs={'data_format': data_format,
                'return_sequences': return_sequences,
                'filters': filters,
                'kernel_size': (num_row, num_col),
                'padding': 'valid'},
        input_shape=inputs.shape)

  def test_conv_lstm_statefulness(self):
    # Tests for statefulness
    num_row = 3
    num_col = 3
    filters = 2
    num_samples = 1
    input_channel = 2
    input_num_row = 5
    input_num_col = 5
    sequence_len = 2
    inputs = np.random.rand(num_samples, sequence_len,
                            input_num_row, input_num_col,
                            input_channel)

    with self.cached_session():
      model = keras.models.Sequential()
      kwargs = {'data_format': 'channels_last',
                'return_sequences': False,
                'filters': filters,
                'kernel_size': (num_row, num_col),
                'stateful': True,
                'batch_input_shape': inputs.shape,
                'padding': 'same'}
      layer = keras.layers.ConvLSTM2D(**kwargs)

      model.add(layer)
      model.compile(optimizer='sgd', loss='mse')
      out1 = model.predict(np.ones_like(inputs))

      # train once so that the states change
      model.train_on_batch(np.ones_like(inputs),
                           np.random.random(out1.shape))
      out2 = model.predict(np.ones_like(inputs))

      # if the state is not reset, output should be different
      self.assertNotEqual(out1.max(), out2.max())

      # check that output changes after states are reset
      # (even though the model itself didn't change)
      layer.reset_states()
      out3 = model.predict(np.ones_like(inputs))
      self.assertNotEqual(out3.max(), out2.max())

      # check that container-level reset_states() works
      model.reset_states()
      out4 = model.predict(np.ones_like(inputs))
      self.assertAllClose(out3, out4, atol=1e-5)

      # check that the call to `predict` updated the states
      out5 = model.predict(np.ones_like(inputs))
      self.assertNotEqual(out4.max(), out5.max())

  def test_conv_lstm_regularizers(self):
    # check regularizers
    num_row = 3
    num_col = 3
    filters = 2
    num_samples = 1
    input_channel = 2
    input_num_row = 5
    input_num_col = 5
    sequence_len = 2
    inputs = np.random.rand(num_samples, sequence_len,
                            input_num_row, input_num_col,
                            input_channel)

    with self.cached_session():
      kwargs = {'data_format': 'channels_last',
                'return_sequences': False,
                'kernel_size': (num_row, num_col),
                'stateful': True,
                'filters': filters,
                'batch_input_shape': inputs.shape,
                'kernel_regularizer': keras.regularizers.L1L2(l1=0.01),
                'recurrent_regularizer': keras.regularizers.L1L2(l1=0.01),
                'activity_regularizer': 'l2',
                'bias_regularizer': 'l2',
                'kernel_constraint': 'max_norm',
                'recurrent_constraint': 'max_norm',
                'bias_constraint': 'max_norm',
                'padding': 'same'}

      layer = keras.layers.ConvLSTM2D(**kwargs)
      layer.build(inputs.shape)
      self.assertEqual(len(layer.losses), 3)
      layer(keras.backend.variable(np.ones(inputs.shape)))
      self.assertEqual(len(layer.losses), 4)

  def test_conv_lstm_dropout(self):
    # check dropout
    with self.cached_session():
      testing_utils.layer_test(
          keras.layers.ConvLSTM2D,
          kwargs={'data_format': 'channels_last',
                  'return_sequences': False,
                  'filters': 2,
                  'kernel_size': (3, 3),
                  'padding': 'same',
                  'dropout': 0.1,
                  'recurrent_dropout': 0.1},
          input_shape=(1, 2, 5, 5, 2))

  def test_conv_lstm_cloning(self):
    with self.cached_session():
      model = keras.models.Sequential()
      model.add(keras.layers.ConvLSTM2D(5, 3, input_shape=(None, 5, 5, 3)))

      test_inputs = np.random.random((2, 4, 5, 5, 3))
      reference_outputs = model.predict(test_inputs)
      weights = model.get_weights()

    # Use a new graph to clone the model
    with self.cached_session():
      clone = keras.models.clone_model(model)
      clone.set_weights(weights)

      outputs = clone.predict(test_inputs)
      self.assertAllClose(reference_outputs, outputs, atol=1e-5)

  def test_conv_lstm_with_initial_state(self):
    num_samples = 32
    sequence_len = 5
    encoder_inputs = keras.layers.Input((None, 32, 32, 3))
    encoder = keras.layers.ConvLSTM2D(
        filters=32, kernel_size=(3, 3), padding='same',
        return_sequences=False, return_state=True)
    _, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = keras.layers.Input((None, 32, 32, 4))
    decoder_lstm = keras.layers.ConvLSTM2D(
        filters=32, kernel_size=(3, 3), padding='same',
        return_sequences=False, return_state=False)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    output = keras.layers.Conv2D(
        1, (3, 3), padding='same', activation='relu')(decoder_outputs)
    model = keras.Model([encoder_inputs, decoder_inputs], output)

    model.compile(
        optimizer='sgd', loss='mse',
        run_eagerly=testing_utils.should_run_eagerly())
    x_1 = np.random.rand(num_samples, sequence_len, 32, 32, 3)
    x_2 = np.random.rand(num_samples, sequence_len, 32, 32, 4)
    y = np.random.rand(num_samples, 32, 32, 1)
    model.fit([x_1, x_2], y)

    model.predict([x_1, x_2])


if __name__ == '__main__':
  test.main()
