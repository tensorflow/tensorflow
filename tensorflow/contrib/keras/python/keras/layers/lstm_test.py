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
"""Tests for LSTM layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import testing_utils
from tensorflow.python.platform import test


class LSTMLayerTest(test.TestCase):

  def test_return_sequences_LSTM(self):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    with self.test_session():
      testing_utils.layer_test(
          keras.layers.LSTM,
          kwargs={'units': units,
                  'return_sequences': True},
          input_shape=(num_samples, timesteps, embedding_dim))

  def test_dynamic_behavior_LSTM(self):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    with self.test_session():
      layer = keras.layers.LSTM(units, input_shape=(None, embedding_dim))
      model = keras.models.Sequential()
      model.add(layer)
      model.compile('sgd', 'mse')
      x = np.random.random((num_samples, timesteps, embedding_dim))
      y = np.random.random((num_samples, units))
      model.train_on_batch(x, y)

  def test_dropout_LSTM(self):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    with self.test_session():
      testing_utils.layer_test(
          keras.layers.LSTM,
          kwargs={'units': units,
                  'dropout': 0.1,
                  'recurrent_dropout': 0.1},
          input_shape=(num_samples, timesteps, embedding_dim))

  def test_implementation_mode_LSTM(self):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    with self.test_session():
      for mode in [0, 1, 2]:
        testing_utils.layer_test(
            keras.layers.LSTM,
            kwargs={'units': units,
                    'implementation': mode},
            input_shape=(num_samples, timesteps, embedding_dim))

  def test_statefulness_LSTM(self):
    num_samples = 2
    timesteps = 3
    embedding_dim = 4
    units = 2
    layer_class = keras.layers.LSTM
    with self.test_session():
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

  def test_regularization_LSTM(self):
    embedding_dim = 4
    layer_class = keras.layers.LSTM
    with self.test_session():
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
      layer(keras.backend.variable(np.ones((2, 3, 2))))
      self.assertEqual(len(layer.losses), 4)

      layer = layer_class(
          5,
          return_sequences=False,
          weights=None,
          input_shape=(None, embedding_dim),
          kernel_constraint=keras.constraints.max_norm(0.01),
          recurrent_constraint=keras.constraints.max_norm(0.01),
          bias_constraint='max_norm')
      layer.build((None, None, embedding_dim))
      self.assertEqual(len(layer.constraints), 3)

  def test_with_masking_layer_LSTM(self):
    layer_class = keras.layers.LSTM
    with self.test_session():
      inputs = np.random.random((2, 3, 4))
      targets = np.abs(np.random.random((2, 3, 5)))
      targets /= targets.sum(axis=-1, keepdims=True)
      model = keras.models.Sequential()
      model.add(keras.layers.Masking(input_shape=(3, 4)))
      model.add(layer_class(units=5, return_sequences=True, unroll=False))
      model.compile(loss='categorical_crossentropy', optimizer='adam')
      model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)

  def test_from_config_LSTM(self):
    layer_class = keras.layers.LSTM
    for stateful in (False, True):
      l1 = layer_class(units=1, stateful=stateful)
      l2 = layer_class.from_config(l1.get_config())
      assert l1.get_config() == l2.get_config()


if __name__ == '__main__':
  test.main()
