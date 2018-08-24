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

from tensorflow.python import keras
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import test
from tensorflow.python.training.checkpointable import util as checkpointable_util


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

    with self.cached_session():
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

    with self.cached_session():
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
      self.assertEqual(layer.cell.state_size, (8, 8, 16, 16, 32, 32))
      self.assertEqual(layer.cell.output_size, 32)
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

    with self.cached_session():
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

    with self.cached_session():
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

    with self.cached_session():
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

    with self.cached_session():
      # test flat list inputs.
      with keras.utils.CustomObjectScope(custom_objects):
        layer = keras.layers.RNN.from_config(config.copy())
      y = layer([x, c])
      model = keras.models.Model([x, c], y)
      model.set_weights(weights)
      y_np_3 = model.predict([x_np, c_np])
      self.assertAllClose(y_np, y_np_3, atol=1e-4)

    with self.cached_session():
      # Test stacking.
      cells = [keras.layers.recurrent.GRUCell(8),
               RNNCellWithConstants(12),
               RNNCellWithConstants(32)]
      layer = keras.layers.recurrent.RNN(cells)
      y = layer(x, constants=c)
      model = keras.models.Model([x, c], y)
      model.compile(optimizer='rmsprop', loss='mse')
      model.train_on_batch(
          [np.zeros((6, 5, 5)), np.zeros((6, 3))],
          np.zeros((6, 32))
      )

    with self.cached_session():
      # Test GRUCell reset_after property.
      x = keras.Input((None, 5))
      c = keras.Input((3,))
      cells = [keras.layers.recurrent.GRUCell(32, reset_after=True)]
      layer = keras.layers.recurrent.RNN(cells)
      y = layer(x, constants=c)
      model = keras.models.Model([x, c], y)
      model.compile(optimizer='rmsprop', loss='mse')
      model.train_on_batch(
          [np.zeros((6, 5, 5)), np.zeros((6, 3))],
          np.zeros((6, 32))
      )

    with self.cached_session():
      # Test stacked RNN serialization
      x_np = np.random.random((6, 5, 5))
      c_np = np.random.random((6, 3))
      y_np = model.predict([x_np, c_np])
      weights = model.get_weights()
      config = layer.get_config()
      with keras.utils.CustomObjectScope(custom_objects):
        layer = keras.layers.recurrent.RNN.from_config(config.copy())
      y = layer(x, constants=c)
      model = keras.models.Model([x, c], y)
      model.set_weights(weights)
      y_np_2 = model.predict([x_np, c_np])
      self.assertAllClose(y_np, y_np_2, atol=1e-4)

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

    with self.cached_session():
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

    with self.cached_session():
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

    with self.cached_session():
      # test flat list inputs
      with keras.utils.CustomObjectScope(custom_objects):
        layer = keras.layers.RNN.from_config(config.copy())
      y = layer([x, s, c])
      model = keras.models.Model([x, s, c], y)
      model.set_weights(weights)
      y_np_3 = model.predict([x_np, s_np, c_np])
      self.assertAllClose(y_np, y_np_3, atol=1e-4)

  def test_stacked_rnn_attributes(self):
    cells = [keras.layers.LSTMCell(1),
             keras.layers.LSTMCell(1)]
    layer = keras.layers.RNN(cells)
    layer.build((None, None, 1))

    # Test weights
    self.assertEqual(len(layer.trainable_weights), 6)
    cells[0].trainable = False
    self.assertEqual(len(layer.trainable_weights), 3)
    self.assertEqual(len(layer.non_trainable_weights), 3)

    # Test `get_losses_for` and `losses`
    x = keras.Input((None, 1))
    loss_1 = math_ops.reduce_sum(x)
    loss_2 = math_ops.reduce_sum(cells[0].kernel)
    cells[0].add_loss(loss_1, inputs=x)
    cells[0].add_loss(loss_2)
    self.assertEqual(len(layer.losses), 2)
    self.assertEqual(layer.get_losses_for(None), [loss_2])
    self.assertEqual(layer.get_losses_for(x), [loss_1])

    # Test `get_updates_for` and `updates`
    cells = [keras.layers.LSTMCell(1),
             keras.layers.LSTMCell(1)]
    layer = keras.layers.RNN(cells)
    layer.build((None, None, 1))

    x = keras.Input((None, 1))
    update_1 = state_ops.assign_add(cells[0].kernel,
                                    x[0, 0, 0] * cells[0].kernel)
    update_2 = state_ops.assign_add(cells[0].kernel,
                                    array_ops.ones_like(cells[0].kernel))
    cells[0].add_update(update_1, inputs=x)
    cells[0].add_update(update_2)
    self.assertEqual(len(layer.updates), 2)
    self.assertEqual(len(layer.get_updates_for(None)), 1)
    self.assertEqual(len(layer.get_updates_for(x)), 1)

  def test_rnn_dynamic_trainability(self):
    layer_class = keras.layers.SimpleRNN
    embedding_dim = 4
    units = 3

    layer = layer_class(units)
    layer.build((None, None, embedding_dim))
    self.assertEqual(len(layer.weights), 3)
    self.assertEqual(len(layer.trainable_weights), 3)
    self.assertEqual(len(layer.non_trainable_weights), 0)
    layer.trainable = False
    self.assertEqual(len(layer.weights), 3)
    self.assertEqual(len(layer.trainable_weights), 0)
    self.assertEqual(len(layer.non_trainable_weights), 3)
    layer.trainable = True
    self.assertEqual(len(layer.weights), 3)
    self.assertEqual(len(layer.trainable_weights), 3)
    self.assertEqual(len(layer.non_trainable_weights), 0)

  def test_state_reuse_with_dropout(self):
    layer_class = keras.layers.SimpleRNN
    embedding_dim = 4
    units = 3
    timesteps = 2
    num_samples = 2

    with self.cached_session():
      input1 = keras.Input(batch_shape=(num_samples, timesteps, embedding_dim))
      layer = layer_class(units,
                          return_state=True,
                          return_sequences=True,
                          dropout=0.2)
      state = layer(input1)[1:]

      input2 = keras.Input(batch_shape=(num_samples, timesteps, embedding_dim))
      output = layer_class(units)(input2, initial_state=state)
      model = keras.Model([input1, input2], output)

      inputs = [np.random.random((num_samples, timesteps, embedding_dim)),
                np.random.random((num_samples, timesteps, embedding_dim))]
      model.predict(inputs)

  def test_builtin_rnn_cell_serialization(self):
    for cell_class in [keras.layers.SimpleRNNCell,
                       keras.layers.GRUCell,
                       keras.layers.LSTMCell]:
      with self.cached_session():
        # Test basic case.
        x = keras.Input((None, 5))
        cell = cell_class(32)
        layer = keras.layers.RNN(cell)
        y = layer(x)
        model = keras.models.Model(x, y)
        model.compile(optimizer='rmsprop', loss='mse')

        # Test basic case serialization.
        x_np = np.random.random((6, 5, 5))
        y_np = model.predict(x_np)
        weights = model.get_weights()
        config = layer.get_config()
        layer = keras.layers.RNN.from_config(config)
        y = layer(x)
        model = keras.models.Model(x, y)
        model.set_weights(weights)
        y_np_2 = model.predict(x_np)
        self.assertAllClose(y_np, y_np_2, atol=1e-4)

        # Test stacking.
        cells = [cell_class(8),
                 cell_class(12),
                 cell_class(32)]
        layer = keras.layers.RNN(cells)
        y = layer(x)
        model = keras.models.Model(x, y)
        model.compile(optimizer='rmsprop', loss='mse')

        # Test stacked RNN serialization.
        x_np = np.random.random((6, 5, 5))
        y_np = model.predict(x_np)
        weights = model.get_weights()
        config = layer.get_config()
        layer = keras.layers.RNN.from_config(config)
        y = layer(x)
        model = keras.models.Model(x, y)
        model.set_weights(weights)
        y_np_2 = model.predict(x_np)
        self.assertAllClose(y_np, y_np_2, atol=1e-4)

  def test_stacked_rnn_dropout(self):
    cells = [keras.layers.LSTMCell(3, dropout=0.1, recurrent_dropout=0.1),
             keras.layers.LSTMCell(3, dropout=0.1, recurrent_dropout=0.1)]
    layer = keras.layers.RNN(cells)

    with self.cached_session():
      x = keras.Input((None, 5))
      y = layer(x)
      model = keras.models.Model(x, y)
      model.compile('sgd', 'mse')
      x_np = np.random.random((6, 5, 5))
      y_np = np.random.random((6, 3))
      model.train_on_batch(x_np, y_np)

  def test_stacked_rnn_compute_output_shape(self):
    cells = [keras.layers.LSTMCell(3),
             keras.layers.LSTMCell(6)]
    embedding_dim = 4
    timesteps = 2
    layer = keras.layers.RNN(cells, return_state=True, return_sequences=True)
    output_shape = layer.compute_output_shape((None, timesteps, embedding_dim))
    expected_output_shape = [(None, timesteps, 6),
                             (None, 3),
                             (None, 3),
                             (None, 6),
                             (None, 6)]
    self.assertEqual(
        [tuple(o.as_list()) for o in output_shape],
        expected_output_shape)

    # Test reverse_state_order = True for stacked cell.
    stacked_cell = keras.layers.StackedRNNCells(
        cells, reverse_state_order=True)
    layer = keras.layers.RNN(
        stacked_cell, return_state=True, return_sequences=True)
    output_shape = layer.compute_output_shape((None, timesteps, embedding_dim))
    expected_output_shape = [(None, timesteps, 6),
                             (None, 6),
                             (None, 6),
                             (None, 3),
                             (None, 3)]
    self.assertEqual(
        [tuple(o.as_list()) for o in output_shape],
        expected_output_shape)

  def test_checkpointable_dependencies(self):
    rnn = keras.layers.SimpleRNN
    with self.cached_session():
      x = np.random.random((2, 2, 2))
      y = np.random.random((2, 2))
      model = keras.models.Sequential()
      model.add(rnn(2))
      model.compile(optimizer='rmsprop', loss='mse')
      model.fit(x, y, epochs=1, batch_size=1)

      # check whether the model variables are present in the
      # checkpointable list of objects
      checkpointed_objects = set(checkpointable_util.list_objects(model))
      for v in model.variables:
        self.assertIn(v, checkpointed_objects)

  def test_high_dimension_RNN(self):
    with self.cached_session():
      # Basic test case.
      unit_a = 10
      unit_b = 20
      input_a = 5
      input_b = 10
      batch = 32
      time_step = 4

      cell = Minimal2DRNNCell(unit_a, unit_b)
      x = keras.Input((None, input_a, input_b))
      layer = keras.layers.RNN(cell)
      y = layer(x)

      self.assertEqual(cell.state_size.as_list(), [unit_a, unit_b])
      init_state = layer.get_initial_state(x)
      self.assertEqual(len(init_state), 1)
      self.assertEqual(init_state[0].get_shape().as_list(),
                       [None, unit_a, unit_b])

      model = keras.models.Model(x, y)
      model.compile(optimizer='rmsprop', loss='mse')
      model.train_on_batch(
          np.zeros((batch, time_step, input_a, input_b)),
          np.zeros((batch, unit_a, unit_b)))
      self.assertEqual(model.output_shape, (None, unit_a, unit_b))

      # Test stacking.
      cells = [
          Minimal2DRNNCell(unit_a, unit_b),
          Minimal2DRNNCell(unit_a * 2, unit_b * 2),
          Minimal2DRNNCell(unit_a * 4, unit_b * 4)
      ]
      layer = keras.layers.RNN(cells)
      y = layer(x)
      model = keras.models.Model(x, y)
      model.compile(optimizer='rmsprop', loss='mse')
      model.train_on_batch(
          np.zeros((batch, time_step, input_a, input_b)),
          np.zeros((batch, unit_a * 4, unit_b * 4)))
      self.assertEqual(model.output_shape, (None, unit_a * 4, unit_b * 4))

  def test_high_dimension_RNN_with_init_state(self):
    unit_a = 10
    unit_b = 20
    input_a = 5
    input_b = 10
    batch = 32
    time_step = 4

    with self.cached_session():
      # Basic test case.
      cell = Minimal2DRNNCell(unit_a, unit_b)
      x = keras.Input((None, input_a, input_b))
      s = keras.Input((unit_a, unit_b))
      layer = keras.layers.RNN(cell)
      y = layer(x, initial_state=s)

      model = keras.models.Model([x, s], y)
      model.compile(optimizer='rmsprop', loss='mse')
      model.train_on_batch([
          np.zeros((batch, time_step, input_a, input_b)),
          np.zeros((batch, unit_a, unit_b))
      ], np.zeros((batch, unit_a, unit_b)))
      self.assertEqual(model.output_shape, (None, unit_a, unit_b))

    with self.cached_session():
      # Bad init state shape.
      bad_shape_a = unit_a * 2
      bad_shape_b = unit_b * 2
      cell = Minimal2DRNNCell(unit_a, unit_b)
      x = keras.Input((None, input_a, input_b))
      s = keras.Input((bad_shape_a, bad_shape_b))
      layer = keras.layers.RNN(cell)
      with self.assertRaisesWithPredicateMatch(ValueError,
                                               'however `cell.state_size` is'):
        layer(x, initial_state=s)

  def test_inconsistent_output_state_size(self):
    with self.cached_session():
      batch = 32
      time_step = 4
      state_size = 5
      input_size = 6
      cell = PlusOneRNNCell(state_size)
      x = keras.Input((None, input_size))
      layer = keras.layers.RNN(cell)
      y = layer(x)

      self.assertEqual(cell.state_size, state_size)
      init_state = layer.get_initial_state(x)
      self.assertEqual(len(init_state), 1)
      self.assertEqual(init_state[0].get_shape().as_list(),
                       [None, state_size])

      model = keras.models.Model(x, y)
      model.compile(optimizer='rmsprop', loss='mse')
      model.train_on_batch(
          np.zeros((batch, time_step, input_size)),
          np.zeros((batch, input_size)))
      self.assertEqual(model.output_shape, (None, input_size))

  def test_get_initial_state(self):
    cell = keras.layers.SimpleRNNCell(5)
    with self.assertRaisesRegexp(ValueError,
                                 'batch_size and dtype cannot be None'):
      cell.get_initial_state(None, None, None)

    inputs = keras.Input((None, 2, 10))
    initial_state = cell.get_initial_state(inputs, None, None)
    self.assertEqual(initial_state.shape.as_list(), [None, 5])
    self.assertEqual(initial_state.dtype, inputs.dtype)

    batch = array_ops.shape(inputs)[0]
    dtype = inputs.dtype
    initial_state = cell.get_initial_state(None, batch, dtype)
    self.assertEqual(initial_state.shape.as_list(), [None, 5])
    self.assertEqual(initial_state.dtype, inputs.dtype)


class Minimal2DRNNCell(keras.layers.Layer):
  """The minimal 2D RNN cell is a simple combination of 2 1-D RNN cell.

  Both internal state and output have 2 dimensions and are orthogonal
  between each other.
  """

  def __init__(self, unit_a, unit_b, **kwargs):
    self.unit_a = unit_a
    self.unit_b = unit_b
    self.state_size = tensor_shape.as_shape([unit_a, unit_b])
    self.output_size = tensor_shape.as_shape([unit_a, unit_b])
    super(Minimal2DRNNCell, self).__init__(**kwargs)

  def build(self, input_shape):
    input_a = input_shape[-2]
    input_b = input_shape[-1]
    self.kernel = self.add_weight(
        shape=(input_a, input_b, self.unit_a, self.unit_b),
        initializer='uniform',
        name='kernel')
    self.recurring_kernel = self.add_weight(
        shape=(self.unit_a, self.unit_b, self.unit_a, self.unit_b),
        initializer='uniform',
        name='recurring_kernel')
    self.bias = self.add_weight(
        shape=(self.unit_a, self.unit_b), initializer='uniform', name='bias')
    self.built = True

  def call(self, inputs, states):
    prev_output = states[0]
    h = special_math_ops.einsum('bij,ijkl->bkl', inputs, self.kernel)
    h += array_ops.expand_dims(self.bias, axis=0)
    output = h + special_math_ops.einsum('bij,ijkl->bkl', prev_output,
                                         self.recurring_kernel)
    return output, [output]


class PlusOneRNNCell(keras.layers.Layer):
  """Add one to the input and state.

  This cell is used for testing state_size and output_size."""

  def __init__(self, num_unit, **kwargs):
    self.state_size = num_unit
    super(PlusOneRNNCell, self).__init__(**kwargs)

  def build(self, input_shape):
    self.output_size = input_shape[-1]

  def call(self, inputs, states):
    return inputs + 1, [states[0] + 1]


if __name__ == '__main__':
  test.main()
