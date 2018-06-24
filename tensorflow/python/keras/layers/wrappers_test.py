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
"""Tests for layer wrappers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.platform import test
from tensorflow.python.training.rmsprop import RMSPropOptimizer


class _RNNCellWithConstants(keras.layers.Layer):

  def __init__(self, units, **kwargs):
    self.units = units
    self.state_size = units
    super(_RNNCellWithConstants, self).__init__(**kwargs)

  def build(self, input_shape):
    [input_shape, constant_shape] = input_shape

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
    base_config = super(_RNNCellWithConstants, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class TimeDistributedTest(test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes()
  def test_timedistributed_dense(self):
    model = keras.models.Sequential()
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(2), input_shape=(3, 4)))
    model.compile(optimizer=RMSPropOptimizer(0.01), loss='mse')
    model.fit(
        np.random.random((10, 3, 4)),
        np.random.random((10, 3, 2)),
        epochs=1,
        batch_size=10)

    # test config
    model.get_config()

  def test_timedistributed_static_batch_size(self):
    model = keras.models.Sequential()
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(2), input_shape=(3, 4), batch_size=10))
    model.compile(optimizer=RMSPropOptimizer(0.01), loss='mse')
    model.fit(
        np.random.random((10, 3, 4)),
        np.random.random((10, 3, 2)),
        epochs=1,
        batch_size=10)

  def test_timedistributed_conv2d(self):
    with self.test_session():
      model = keras.models.Sequential()
      model.add(
          keras.layers.TimeDistributed(
              keras.layers.Conv2D(5, (2, 2), padding='same'),
              input_shape=(2, 4, 4, 3)))
      model.add(keras.layers.Activation('relu'))
      model.compile(optimizer='rmsprop', loss='mse')
      model.train_on_batch(
          np.random.random((1, 2, 4, 4, 3)), np.random.random((1, 2, 4, 4, 5)))

      model = keras.models.model_from_json(model.to_json())
      model.summary()

  def test_timedistributed_stacked(self):
    with self.test_session():
      model = keras.models.Sequential()
      model.add(
          keras.layers.TimeDistributed(
              keras.layers.Dense(2), input_shape=(3, 4)))
      model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
      model.add(keras.layers.Activation('relu'))
      model.compile(optimizer='rmsprop', loss='mse')

      model.fit(
          np.random.random((10, 3, 4)),
          np.random.random((10, 3, 3)),
          epochs=1,
          batch_size=10)

  def test_regularizers(self):
    with self.test_session():
      model = keras.models.Sequential()
      model.add(
          keras.layers.TimeDistributed(
              keras.layers.Dense(2, kernel_regularizer='l1'),
              input_shape=(3, 4)))
      model.add(keras.layers.Activation('relu'))
      model.compile(optimizer='rmsprop', loss='mse')
      self.assertEqual(len(model.losses), 1)

  def test_TimeDistributed_learning_phase(self):
    with self.test_session():
      # test layers that need learning_phase to be set
      np.random.seed(1234)
      x = keras.layers.Input(shape=(3, 2))
      y = keras.layers.TimeDistributed(
          keras.layers.Dropout(.999))(x, training=True)
      model = keras.models.Model(x, y)
      y = model.predict(np.random.random((10, 3, 2)))
      self.assertAllClose(np.mean(y), 0., atol=1e-1, rtol=1e-1)

  def test_TimeDistributed_batchnorm(self):
    with self.test_session():
      # test that wrapped BN updates still work.
      model = keras.models.Sequential()
      model.add(keras.layers.TimeDistributed(
          keras.layers.BatchNormalization(center=True, scale=True),
          name='bn',
          input_shape=(10, 2)))
      model.compile(optimizer='rmsprop', loss='mse')
      # Assert that mean and variance are 0 and 1.
      td = model.layers[0]
      self.assertAllClose(td.get_weights()[2], np.array([0, 0]))
      assert np.array_equal(td.get_weights()[3], np.array([1, 1]))
      # Train
      model.train_on_batch(np.random.normal(loc=2, scale=2, size=(1, 10, 2)),
                           np.broadcast_to(np.array([0, 1]), (1, 10, 2)))
      # Assert that mean and variance changed.
      assert not np.array_equal(td.get_weights()[2], np.array([0, 0]))
      assert not np.array_equal(td.get_weights()[3], np.array([1, 1]))
      # Verify input_map has one mapping from inputs to reshaped inputs.
      self.assertEqual(len(td._input_map.keys()), 1)

  def test_TimeDistributed_trainable(self):
    # test layers that need learning_phase to be set
    x = keras.layers.Input(shape=(3, 2))
    layer = keras.layers.TimeDistributed(keras.layers.BatchNormalization())
    _ = layer(x)
    assert len(layer.updates) == 2
    assert len(layer.trainable_weights) == 2
    layer.trainable = False
    assert not layer.updates
    assert not layer.trainable_weights
    layer.trainable = True
    assert len(layer.updates) == 2
    assert len(layer.trainable_weights) == 2


class BidirectionalTest(test.TestCase):

  def test_bidirectional(self):
    rnn = keras.layers.SimpleRNN
    samples = 2
    dim = 2
    timesteps = 2
    output_dim = 2
    with self.test_session():
      for mode in ['sum', 'concat', 'ave', 'mul']:
        x = np.random.random((samples, timesteps, dim))
        target_dim = 2 * output_dim if mode == 'concat' else output_dim
        y = np.random.random((samples, target_dim))

        # test with Sequential model
        model = keras.models.Sequential()
        model.add(
            keras.layers.Bidirectional(
                rnn(output_dim), merge_mode=mode, input_shape=(timesteps, dim)))
        model.compile(optimizer=RMSPropOptimizer(0.01), loss='mse')
        model.fit(x, y, epochs=1, batch_size=1)

        # test compute output shape
        ref_shape = model.layers[-1].output.get_shape()
        shape = model.layers[-1].compute_output_shape(
            (None, timesteps, dim))
        self.assertListEqual(shape.as_list(), ref_shape.as_list())

        # test config
        model.get_config()
        model = keras.models.model_from_json(model.to_json())
        model.summary()

  def test_bidirectional_weight_loading(self):
    rnn = keras.layers.SimpleRNN
    samples = 2
    dim = 2
    timesteps = 2
    output_dim = 2
    with self.test_session():
      x = np.random.random((samples, timesteps, dim))
      model = keras.models.Sequential()
      model.add(
          keras.layers.Bidirectional(
              rnn(output_dim), input_shape=(timesteps, dim)))
      y_ref = model.predict(x)
      weights = model.layers[-1].get_weights()
      model.layers[-1].set_weights(weights)
      y = model.predict(x)
      self.assertAllClose(y, y_ref)

  def test_bidirectional_stacked(self):
    # test stacked bidirectional layers
    rnn = keras.layers.SimpleRNN
    samples = 2
    dim = 2
    timesteps = 2
    output_dim = 2
    mode = 'sum'

    with self.test_session():
      x = np.random.random((samples, timesteps, dim))
      target_dim = 2 * output_dim if mode == 'concat' else output_dim
      y = np.random.random((samples, target_dim))

      model = keras.models.Sequential()
      model.add(
          keras.layers.Bidirectional(
              rnn(output_dim, return_sequences=True),
              merge_mode=mode,
              input_shape=(timesteps, dim)))
      model.add(keras.layers.Bidirectional(rnn(output_dim), merge_mode=mode))
      model.compile(loss='mse', optimizer='sgd')
      model.fit(x, y, epochs=1, batch_size=1)

      # test with functional API
      inputs = keras.layers.Input((timesteps, dim))
      output = keras.layers.Bidirectional(
          rnn(output_dim), merge_mode=mode)(inputs)
      model = keras.models.Model(inputs, output)
      model.compile(loss='mse', optimizer='sgd')
      model.fit(x, y, epochs=1, batch_size=1)

  def test_bidirectional_statefulness(self):
    # Bidirectional and stateful
    rnn = keras.layers.SimpleRNN
    samples = 2
    dim = 2
    timesteps = 2
    output_dim = 2
    mode = 'sum'

    with self.test_session():
      x = np.random.random((samples, timesteps, dim))
      target_dim = 2 * output_dim if mode == 'concat' else output_dim
      y = np.random.random((samples, target_dim))

      inputs = keras.layers.Input(batch_shape=(1, timesteps, dim))
      output = keras.layers.Bidirectional(
          rnn(output_dim, stateful=True), merge_mode=mode)(inputs)
      model = keras.models.Model(inputs, output)
      model.compile(loss='mse', optimizer='sgd')
      model.fit(x, y, epochs=1, batch_size=1)

  def test_Bidirectional_merged_value(self):
    rnn = keras.layers.LSTM
    samples = 2
    dim = 5
    timesteps = 3
    units = 3
    x = [np.random.rand(samples, timesteps, dim)]

    with self.test_session():
      for merge_mode in ['sum', 'mul', 'ave', 'concat', None]:
        if merge_mode == 'sum':
          merge_func = lambda y, y_rev: y + y_rev
        elif merge_mode == 'mul':
          merge_func = lambda y, y_rev: y * y_rev
        elif merge_mode == 'ave':
          merge_func = lambda y, y_rev: (y + y_rev) / 2
        elif merge_mode == 'concat':
          merge_func = lambda y, y_rev: np.concatenate((y, y_rev), axis=-1)
        else:
          merge_func = lambda y, y_rev: [y, y_rev]

        # basic case
        inputs = keras.Input((timesteps, dim))
        layer = keras.layers.Bidirectional(
            rnn(units, return_sequences=True), merge_mode=merge_mode)
        f_merged = keras.backend.function([inputs], _to_list(layer(inputs)))
        f_forward = keras.backend.function([inputs],
                                           [layer.forward_layer.call(inputs)])
        f_backward = keras.backend.function(
            [inputs],
            [keras.backend.reverse(layer.backward_layer.call(inputs), 1)])

        y_merged = f_merged(x)
        y_expected = _to_list(merge_func(f_forward(x)[0], f_backward(x)[0]))
        assert len(y_merged) == len(y_expected)
        for x1, x2 in zip(y_merged, y_expected):
          self.assertAllClose(x1, x2, atol=1e-5)

        # test return_state
        inputs = keras.Input((timesteps, dim))
        layer = keras.layers.Bidirectional(
            rnn(units, return_state=True), merge_mode=merge_mode)
        f_merged = keras.backend.function([inputs], layer(inputs))
        f_forward = keras.backend.function([inputs],
                                           layer.forward_layer.call(inputs))
        f_backward = keras.backend.function([inputs],
                                            layer.backward_layer.call(inputs))
        n_states = len(layer.layer.states)

        y_merged = f_merged(x)
        y_forward = f_forward(x)
        y_backward = f_backward(x)
        y_expected = _to_list(merge_func(y_forward[0], y_backward[0]))
        assert len(y_merged) == len(y_expected) + n_states * 2
        for x1, x2 in zip(y_merged, y_expected):
          self.assertAllClose(x1, x2, atol=1e-5)

        y_merged = y_merged[-n_states * 2:]
        y_forward = y_forward[-n_states:]
        y_backward = y_backward[-n_states:]
        for state_birnn, state_inner in zip(y_merged, y_forward + y_backward):
          self.assertAllClose(state_birnn, state_inner, atol=1e-5)

  def test_Bidirectional_dropout(self):
    rnn = keras.layers.LSTM
    samples = 2
    dim = 5
    timesteps = 3
    units = 3
    merge_mode = 'sum'
    x = [np.random.rand(samples, timesteps, dim)]

    with self.test_session():
      inputs = keras.Input((timesteps, dim))
      wrapped = keras.layers.Bidirectional(
          rnn(units, dropout=0.2, recurrent_dropout=0.2), merge_mode=merge_mode)
      outputs = _to_list(wrapped(inputs, training=True))
      assert all(not getattr(x, '_uses_learning_phase') for x in outputs)

      inputs = keras.Input((timesteps, dim))
      wrapped = keras.layers.Bidirectional(
          rnn(units, dropout=0.2, return_state=True), merge_mode=merge_mode)
      outputs = _to_list(wrapped(inputs))
      assert all(x._uses_learning_phase for x in outputs)

      model = keras.Model(inputs, outputs)
      assert model.uses_learning_phase
      y1 = _to_list(model.predict(x))
      y2 = _to_list(model.predict(x))
      for x1, x2 in zip(y1, y2):
        self.assertAllClose(x1, x2, atol=1e-5)

  def test_Bidirectional_state_reuse(self):
    rnn = keras.layers.LSTM
    samples = 2
    dim = 5
    timesteps = 3
    units = 3

    with self.test_session():
      input1 = keras.layers.Input((timesteps, dim))
      layer = keras.layers.Bidirectional(
          rnn(units, return_state=True, return_sequences=True))
      state = layer(input1)[1:]

      # test passing invalid initial_state: passing a tensor
      input2 = keras.layers.Input((timesteps, dim))
      with self.assertRaises(ValueError):
        output = keras.layers.Bidirectional(
            rnn(units))(input2, initial_state=state[0])

      # test valid usage: passing a list
      output = keras.layers.Bidirectional(rnn(units))(input2,
                                                      initial_state=state)
      model = keras.models.Model([input1, input2], output)
      assert len(model.layers) == 4
      assert isinstance(model.layers[-1].input, list)
      inputs = [np.random.rand(samples, timesteps, dim),
                np.random.rand(samples, timesteps, dim)]
      model.predict(inputs)

  def test_Bidirectional_trainable(self):
    # test layers that need learning_phase to be set
    with self.test_session():
      x = keras.layers.Input(shape=(3, 2))
      layer = keras.layers.Bidirectional(keras.layers.SimpleRNN(3))
      _ = layer(x)
      assert len(layer.trainable_weights) == 6
      layer.trainable = False
      assert not layer.trainable_weights
      layer.trainable = True
      assert len(layer.trainable_weights) == 6

  def test_Bidirectional_with_constants(self):
    with self.test_session():
      # Test basic case.
      x = keras.Input((5, 5))
      c = keras.Input((3,))
      cell = _RNNCellWithConstants(32)
      custom_objects = {'_RNNCellWithConstants': _RNNCellWithConstants}
      with keras.utils.CustomObjectScope(custom_objects):
        layer = keras.layers.Bidirectional(keras.layers.RNN(cell))
      y = layer(x, constants=c)
      model = keras.Model([x, c], y)
      model.compile(optimizer='rmsprop', loss='mse')
      model.train_on_batch(
          [np.zeros((6, 5, 5)), np.zeros((6, 3))],
          np.zeros((6, 64))
      )

      # Test basic case serialization.
      x_np = np.random.random((6, 5, 5))
      c_np = np.random.random((6, 3))
      y_np = model.predict([x_np, c_np])
      weights = model.get_weights()
      config = layer.get_config()

      with keras.utils.CustomObjectScope(custom_objects):
        layer = keras.layers.Bidirectional.from_config(copy.deepcopy(config))
      y = layer(x, constants=c)
      model = keras.Model([x, c], y)
      model.set_weights(weights)
      y_np_2 = model.predict([x_np, c_np])
      self.assertAllClose(y_np, y_np_2, atol=1e-4)

      # Test flat list inputs
      with keras.utils.CustomObjectScope(custom_objects):
        layer = keras.layers.Bidirectional.from_config(copy.deepcopy(config))
      y = layer([x, c])
      model = keras.Model([x, c], y)
      model.set_weights(weights)
      y_np_3 = model.predict([x_np, c_np])
      self.assertAllClose(y_np, y_np_3, atol=1e-4)

  def test_Bidirectional_with_constants_layer_passing_initial_state(self):
    with self.test_session():
      # Test basic case.
      x = keras.Input((5, 5))
      c = keras.Input((3,))
      s_for = keras.Input((32,))
      s_bac = keras.Input((32,))
      cell = _RNNCellWithConstants(32)
      custom_objects = {'_RNNCellWithConstants': _RNNCellWithConstants}
      with keras.utils.CustomObjectScope(custom_objects):
        layer = keras.layers.Bidirectional(keras.layers.RNN(cell))
      y = layer(x, initial_state=[s_for, s_bac], constants=c)
      model = keras.Model([x, s_for, s_bac, c], y)
      model.compile(optimizer='rmsprop', loss='mse')
      model.train_on_batch(
          [np.zeros((6, 5, 5)),
           np.zeros((6, 32)),
           np.zeros((6, 32)),
           np.zeros((6, 3))],
          np.zeros((6, 64))
      )

      # Test basic case serialization.
      x_np = np.random.random((6, 5, 5))
      s_fw_np = np.random.random((6, 32))
      s_bk_np = np.random.random((6, 32))
      c_np = np.random.random((6, 3))
      y_np = model.predict([x_np, s_fw_np, s_bk_np, c_np])
      weights = model.get_weights()
      config = layer.get_config()

      with keras.utils.CustomObjectScope(custom_objects):
        layer = keras.layers.Bidirectional.from_config(copy.deepcopy(config))
      y = layer(x, initial_state=[s_for, s_bac], constants=c)
      model = keras.Model([x, s_for, s_bac, c], y)
      model.set_weights(weights)
      y_np_2 = model.predict([x_np, s_fw_np, s_bk_np, c_np])
      self.assertAllClose(y_np, y_np_2, atol=1e-4)

      # Verify that state is used
      y_np_2_different_s = model.predict(
          [x_np, s_fw_np + 10., s_bk_np + 10., c_np])
      assert np.mean(y_np - y_np_2_different_s) != 0

      # Test flat list inputs
      with keras.utils.CustomObjectScope(custom_objects):
        layer = keras.layers.Bidirectional.from_config(copy.deepcopy(config))
      y = layer([x, s_for, s_bac, c])
      model = keras.Model([x, s_for, s_bac, c], y)
      model.set_weights(weights)
      y_np_3 = model.predict([x_np, s_fw_np, s_bk_np, c_np])
      self.assertAllClose(y_np, y_np_3, atol=1e-4)


def _to_list(ls):
  if isinstance(ls, list):
    return ls
  else:
    return [ls]


if __name__ == '__main__':
  test.main()
