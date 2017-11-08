# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for recurrent layers functionality other than GRU, LSTM, SimpleRNN.

See also: lstm_test.py, gru_test.py, simplernn_test.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras._impl import keras
from tensorflow.python.platform import test


class RNNTest(test.TestCase):

  def test_minimal_rnn_cell_non_layer(self):

    class MinimalRNNCell(object):

      def __init__(self, units, input_dim):
        self.units = units
        self.state_size = units
        self.kernel = keras.backend.variable(
            np.random.random((input_dim, units)))

      def call(self, inputs, states):
        prev_output = states[0]
        output = keras.backend.dot(inputs, self.kernel) + prev_output
        return output, [output]

    with self.test_session():
      # Basic test case.
      cell = MinimalRNNCell(32, 5)
      x = keras.Input((None, 5))
      layer = keras.layers.RNN(cell)
      y = layer(x)
      model = keras.models.Model(x, y)
      model.compile(optimizer='rmsprop', loss='mse')
      model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

      # Test stacking.
      cells = [MinimalRNNCell(8, 5),
               MinimalRNNCell(32, 8),
               MinimalRNNCell(32, 32)]
      layer = keras.layers.RNN(cells)
      y = layer(x)
      model = keras.models.Model(x, y)
      model.compile(optimizer='rmsprop', loss='mse')
      model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

  def test_minimal_rnn_cell_non_layer_multiple_states(self):

    class MinimalRNNCell(object):

      def __init__(self, units, input_dim):
        self.units = units
        self.state_size = (units, units)
        self.kernel = keras.backend.variable(
            np.random.random((input_dim, units)))

      def call(self, inputs, states):
        prev_output_1 = states[0]
        prev_output_2 = states[1]
        output = keras.backend.dot(inputs, self.kernel)
        output += prev_output_1
        output -= prev_output_2
        return output, [output * 2, output * 3]

    with self.test_session():
      # Basic test case.
      cell = MinimalRNNCell(32, 5)
      x = keras.Input((None, 5))
      layer = keras.layers.RNN(cell)
      y = layer(x)
      model = keras.models.Model(x, y)
      model.compile(optimizer='rmsprop', loss='mse')
      model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

      # Test stacking.
      cells = [MinimalRNNCell(8, 5),
               MinimalRNNCell(16, 8),
               MinimalRNNCell(32, 16)]
      layer = keras.layers.RNN(cells)
      assert layer.cell.state_size == (32, 32, 16, 16, 8, 8)
      y = layer(x)
      model = keras.models.Model(x, y)
      model.compile(optimizer='rmsprop', loss='mse')
      model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

  def test_minimal_rnn_cell_layer(self):

    class MinimalRNNCell(keras.layers.Layer):

      def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

      def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

      def call(self, inputs, states):
        prev_output = states[0]
        h = keras.backend.dot(inputs, self.kernel)
        output = h + keras.backend.dot(prev_output, self.recurrent_kernel)
        return output, [output]

      def get_config(self):
        config = {'units': self.units}
        base_config = super(MinimalRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    with self.test_session():
      # Test basic case.
      x = keras.Input((None, 5))
      cell = MinimalRNNCell(32)
      layer = keras.layers.RNN(cell)
      y = layer(x)
      model = keras.models.Model(x, y)
      model.compile(optimizer='rmsprop', loss='mse')
      model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

      # Test basic case serialization.
      x_np = np.random.random((6, 5, 5))
      y_np = model.predict(x_np)
      weights = model.get_weights()
      config = layer.get_config()
      with keras.utils.CustomObjectScope({'MinimalRNNCell': MinimalRNNCell}):
        layer = keras.layers.RNN.from_config(config)
      y = layer(x)
      model = keras.models.Model(x, y)
      model.set_weights(weights)
      y_np_2 = model.predict(x_np)
      self.assertAllClose(y_np, y_np_2, atol=1e-4)

      # Test stacking.
      cells = [MinimalRNNCell(8),
               MinimalRNNCell(12),
               MinimalRNNCell(32)]
      layer = keras.layers.RNN(cells)
      y = layer(x)
      model = keras.models.Model(x, y)
      model.compile(optimizer='rmsprop', loss='mse')
      model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

      # Test stacked RNN serialization.
      x_np = np.random.random((6, 5, 5))
      y_np = model.predict(x_np)
      weights = model.get_weights()
      config = layer.get_config()
      with keras.utils.CustomObjectScope({'MinimalRNNCell': MinimalRNNCell}):
        layer = keras.layers.RNN.from_config(config)
      y = layer(x)
      model = keras.models.Model(x, y)
      model.set_weights(weights)
      y_np_2 = model.predict(x_np)
      self.assertAllClose(y_np, y_np_2, atol=1e-4)

  def test_rnn_cell_with_constants_layer(self):

    class RNNCellWithConstants(keras.layers.Layer):

      def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(RNNCellWithConstants, self).__init__(**kwargs)

      def build(self, input_shape):
        if not isinstance(input_shape, list):
          raise TypeError('expects constants shape')
        [input_shape, constant_shape] = input_shape
        # will (and should) raise if more than one constant passed

        self.input_kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='uniform',
            name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.constant_kernel = self.add_weight(
            shape=(constant_shape[-1], self.units),
            initializer='uniform',
            name='constant_kernel')
        self.built = True

      def call(self, inputs, states, constants):
        [prev_output] = states
        [constant] = constants
        h_input = keras.backend.dot(inputs, self.input_kernel)
        h_state = keras.backend.dot(prev_output, self.recurrent_kernel)
        h_const = keras.backend.dot(constant, self.constant_kernel)
        output = h_input + h_state + h_const
        return output, [output]

      def get_config(self):
        config = {'units': self.units}
        base_config = super(RNNCellWithConstants, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    with self.test_session():
      # Test basic case.
      x = keras.Input((None, 5))
      c = keras.Input((3,))
      cell = RNNCellWithConstants(32)
      layer = keras.layers.RNN(cell)
      y = layer(x, constants=c)
      model = keras.models.Model([x, c], y)
      model.compile(optimizer='rmsprop', loss='mse')
      model.train_on_batch(
          [np.zeros((6, 5, 5)), np.zeros((6, 3))],
          np.zeros((6, 32))
      )

    with self.test_session():
      # Test basic case serialization.
      x_np = np.random.random((6, 5, 5))
      c_np = np.random.random((6, 3))
      y_np = model.predict([x_np, c_np])
      weights = model.get_weights()
      config = layer.get_config()
      custom_objects = {'RNNCellWithConstants': RNNCellWithConstants}
      with keras.utils.CustomObjectScope(custom_objects):
        layer = keras.layers.RNN.from_config(config.copy())
      y = layer(x, constants=c)
      model = keras.models.Model([x, c], y)
      model.set_weights(weights)
      y_np_2 = model.predict([x_np, c_np])
      self.assertAllClose(y_np, y_np_2, atol=1e-4)

    with self.test_session():
      # test flat list inputs
      with keras.utils.CustomObjectScope(custom_objects):
        layer = keras.layers.RNN.from_config(config.copy())
      y = layer([x, c])
      model = keras.models.Model([x, c], y)
      model.set_weights(weights)
      y_np_3 = model.predict([x_np, c_np])
      self.assertAllClose(y_np, y_np_3, atol=1e-4)

  def test_rnn_cell_with_constants_layer_passing_initial_state(self):

    class RNNCellWithConstants(keras.layers.Layer):

      def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(RNNCellWithConstants, self).__init__(**kwargs)

      def build(self, input_shape):
        if not isinstance(input_shape, list):
          raise TypeError('expects constants shape')
        [input_shape, constant_shape] = input_shape
        # will (and should) raise if more than one constant passed

        self.input_kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='uniform',
            name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.constant_kernel = self.add_weight(
            shape=(constant_shape[-1], self.units),
            initializer='uniform',
            name='constant_kernel')
        self.built = True

      def call(self, inputs, states, constants):
        [prev_output] = states
        [constant] = constants
        h_input = keras.backend.dot(inputs, self.input_kernel)
        h_state = keras.backend.dot(prev_output, self.recurrent_kernel)
        h_const = keras.backend.dot(constant, self.constant_kernel)
        output = h_input + h_state + h_const
        return output, [output]

      def get_config(self):
        config = {'units': self.units}
        base_config = super(RNNCellWithConstants, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    with self.test_session():
      # Test basic case.
      x = keras.Input((None, 5))
      c = keras.Input((3,))
      s = keras.Input((32,))
      cell = RNNCellWithConstants(32)
      layer = keras.layers.RNN(cell)
      y = layer(x, initial_state=s, constants=c)
      model = keras.models.Model([x, s, c], y)
      model.compile(optimizer='rmsprop', loss='mse')
      model.train_on_batch(
          [np.zeros((6, 5, 5)), np.zeros((6, 32)), np.zeros((6, 3))],
          np.zeros((6, 32))
      )

    with self.test_session():
      # Test basic case serialization.
      x_np = np.random.random((6, 5, 5))
      s_np = np.random.random((6, 32))
      c_np = np.random.random((6, 3))
      y_np = model.predict([x_np, s_np, c_np])
      weights = model.get_weights()
      config = layer.get_config()
      custom_objects = {'RNNCellWithConstants': RNNCellWithConstants}
      with keras.utils.CustomObjectScope(custom_objects):
        layer = keras.layers.RNN.from_config(config.copy())
      y = layer(x, initial_state=s, constants=c)
      model = keras.models.Model([x, s, c], y)
      model.set_weights(weights)
      y_np_2 = model.predict([x_np, s_np, c_np])
      self.assertAllClose(y_np, y_np_2, atol=1e-4)

      # verify that state is used
      y_np_2_different_s = model.predict([x_np, s_np + 10., c_np])
      with self.assertRaises(AssertionError):
        self.assertAllClose(y_np, y_np_2_different_s, atol=1e-4)

    with self.test_session():
      # test flat list inputs
      with keras.utils.CustomObjectScope(custom_objects):
        layer = keras.layers.RNN.from_config(config.copy())
      y = layer([x, s, c])
      model = keras.models.Model([x, s, c], y)
      model.set_weights(weights)
      y_np_3 = model.predict([x_np, s_np, c_np])
      self.assertAllClose(y_np, y_np_3, atol=1e-4)

  def test_stacked_rnn_attributes(self):
    cells = [keras.layers.LSTMCell(3),
             keras.layers.LSTMCell(3, kernel_regularizer='l2')]
    layer = keras.layers.RNN(cells)
    layer.build((None, None, 5))

    # Test regularization losses
    assert len(layer.losses) == 1

    # Test weights
    assert len(layer.trainable_weights) == 6
    cells[0].trainable = False
    assert len(layer.trainable_weights) == 3
    assert len(layer.non_trainable_weights) == 3

    # Test `get_losses_for`
    x = keras.Input((None, 5))
    y = keras.backend.sum(x)
    cells[0].add_loss(y, inputs=x)
    assert layer.get_losses_for(x) == [y]


if __name__ == '__main__':
  test.main()
