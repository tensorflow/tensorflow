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

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers.rnn_cell_wrapper_v2 import ResidualWrapper
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.training.tracking import util as trackable_util
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity


class _RNNCellWithConstants(keras.layers.Layer):

  def __init__(self, units, constant_size, **kwargs):
    self.units = units
    self.state_size = units
    self.constant_size = constant_size
    super(_RNNCellWithConstants, self).__init__(**kwargs)

  def build(self, input_shape):
    self.input_kernel = self.add_weight(
        shape=(input_shape[-1], self.units),
        initializer='uniform',
        name='kernel')
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units),
        initializer='uniform',
        name='recurrent_kernel')
    self.constant_kernel = self.add_weight(
        shape=(self.constant_size, self.units),
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
    config = {'units': self.units, 'constant_size': self.constant_size}
    base_config = super(_RNNCellWithConstants, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class _ResidualLSTMCell(keras.layers.LSTMCell):

  def call(self, inputs, states, training=None):
    output, states = super(_ResidualLSTMCell, self).call(inputs, states)
    return output + inputs, states


class _AddOneCell(keras.layers.AbstractRNNCell):
  """Increments inputs and state by one on each call."""

  @property
  def state_size(self):
    return 1

  @property
  def output_size(self):
    return 1

  def call(self, inputs, state):
    inputs = math_ops.reduce_mean(inputs, axis=1, keepdims=True)
    outputs = inputs + 1.0
    state = nest.map_structure(lambda t: t + 1.0, state)
    return outputs, state


class TimeDistributedTest(keras_parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_timedistributed_dense(self):
    model = keras.models.Sequential()
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(2), input_shape=(3, 4)))
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(
        np.random.random((10, 3, 4)),
        np.random.random((10, 3, 2)),
        epochs=1,
        batch_size=10)

    # test config
    model.get_config()

    # check whether the model variables are present in the
    # trackable list of objects
    checkpointed_objects = object_identity.ObjectIdentitySet(
        trackable_util.list_objects(model))
    for v in model.variables:
      self.assertIn(v, checkpointed_objects)

  def test_timedistributed_static_batch_size(self):
    model = keras.models.Sequential()
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(2), input_shape=(3, 4), batch_size=10))
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(
        np.random.random((10, 3, 4)),
        np.random.random((10, 3, 2)),
        epochs=1,
        batch_size=10)

  def test_timedistributed_invalid_init(self):
    x = constant_op.constant(np.zeros((1, 1)).astype('float32'))
    with self.assertRaisesRegex(
        ValueError, 'Please initialize `TimeDistributed` layer with a '
        '`tf.keras.layers.Layer` instance.'):
      keras.layers.TimeDistributed(x)

  def test_timedistributed_conv2d(self):
    with self.cached_session():
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
    with self.cached_session():
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
    with self.cached_session():
      model = keras.models.Sequential()
      model.add(
          keras.layers.TimeDistributed(
              keras.layers.Dense(2, kernel_regularizer='l1',
                                 activity_regularizer='l1'),
              input_shape=(3, 4)))
      model.add(keras.layers.Activation('relu'))
      model.compile(optimizer='rmsprop', loss='mse')
      self.assertEqual(len(model.losses), 2)

  def test_TimeDistributed_learning_phase(self):
    with self.cached_session():
      # test layers that need learning_phase to be set
      np.random.seed(1234)
      x = keras.layers.Input(shape=(3, 2))
      y = keras.layers.TimeDistributed(keras.layers.Dropout(.999))(
          x, training=True)
      model = keras.models.Model(x, y)
      y = model.predict(np.random.random((10, 3, 2)))
      self.assertAllClose(np.mean(y), 0., atol=1e-1, rtol=1e-1)

  def test_TimeDistributed_batchnorm(self):
    with self.cached_session():
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

  def test_TimeDistributed_trainable(self):
    # test layers that need learning_phase to be set
    x = keras.layers.Input(shape=(3, 2))
    layer = keras.layers.TimeDistributed(keras.layers.BatchNormalization())
    _ = layer(x)
    self.assertEqual(len(layer.trainable_weights), 2)
    layer.trainable = False
    assert not layer.trainable_weights
    layer.trainable = True
    assert len(layer.trainable_weights) == 2

  def test_TimeDistributed_with_masked_embedding_and_unspecified_shape(self):
    with self.cached_session():
      # test with unspecified shape and Embeddings with mask_zero
      model = keras.models.Sequential()
      model.add(keras.layers.TimeDistributed(
          keras.layers.Embedding(5, 6, mask_zero=True),
          input_shape=(None, None)))  # N by t_1 by t_2 by 6
      model.add(keras.layers.TimeDistributed(
          keras.layers.SimpleRNN(7, return_sequences=True)))
      model.add(keras.layers.TimeDistributed(
          keras.layers.SimpleRNN(8, return_sequences=False)))
      model.add(keras.layers.SimpleRNN(1, return_sequences=False))
      model.compile(optimizer='rmsprop', loss='mse')
      model_input = np.random.randint(low=1, high=5, size=(10, 3, 4),
                                      dtype='int32')
      for i in range(4):
        model_input[i, i:, i:] = 0
      model.fit(model_input,
                np.random.random((10, 1)), epochs=1, batch_size=10)
      mask_outputs = [model.layers[0].compute_mask(model.input)]
      for layer in model.layers[1:]:
        mask_outputs.append(layer.compute_mask(layer.input, mask_outputs[-1]))
      func = keras.backend.function([model.input], mask_outputs[:-1])
      mask_outputs_val = func([model_input])
      ref_mask_val_0 = model_input > 0         # embedding layer
      ref_mask_val_1 = ref_mask_val_0          # first RNN layer
      ref_mask_val_2 = np.any(ref_mask_val_1, axis=-1)     # second RNN layer
      ref_mask_val = [ref_mask_val_0, ref_mask_val_1, ref_mask_val_2]
      for i in range(3):
        self.assertAllEqual(mask_outputs_val[i], ref_mask_val[i])
      self.assertIs(mask_outputs[-1], None)  # final layer

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_TimeDistributed_with_masking_layer(self):
    # test with Masking layer
    model = keras.models.Sequential()
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Masking(mask_value=0.,), input_shape=(None, 4)))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(5)))
    model.compile(optimizer='rmsprop', loss='mse')
    model_input = np.random.randint(low=1, high=5, size=(10, 3, 4))
    for i in range(4):
      model_input[i, i:, :] = 0.
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(model_input, np.random.random((10, 3, 5)), epochs=1, batch_size=6)
    mask_outputs = [model.layers[0].compute_mask(model.input)]
    mask_outputs += [
        model.layers[1].compute_mask(model.layers[1].input, mask_outputs[-1])
    ]
    func = keras.backend.function([model.input], mask_outputs)
    mask_outputs_val = func([model_input])
    self.assertEqual((mask_outputs_val[0]).all(), model_input.all())
    self.assertEqual((mask_outputs_val[1]).all(), model_input.all())

  def test_TimeDistributed_with_different_time_shapes(self):
    time_dist = keras.layers.TimeDistributed(keras.layers.Dense(5))
    ph_1 = keras.backend.placeholder(shape=(None, 10, 13))
    out_1 = time_dist(ph_1)
    self.assertEqual(out_1.shape.as_list(), [None, 10, 5])

    ph_2 = keras.backend.placeholder(shape=(None, 1, 13))
    out_2 = time_dist(ph_2)
    self.assertEqual(out_2.shape.as_list(), [None, 1, 5])

    ph_3 = keras.backend.placeholder(shape=(None, 1, 18))
    with self.assertRaisesRegex(ValueError, 'is incompatible with layer'):
      time_dist(ph_3)

  def test_TimeDistributed_with_invalid_dimensions(self):
    time_dist = keras.layers.TimeDistributed(keras.layers.Dense(5))
    ph = keras.backend.placeholder(shape=(None, 10))
    with self.assertRaisesRegex(
        ValueError,
        '`TimeDistributed` Layer should be passed an `input_shape `'):
      time_dist(ph)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_TimeDistributed_reshape(self):

    class NoReshapeLayer(keras.layers.Layer):

      def call(self, inputs):
        return inputs

    # Built-in layers that aren't stateful use the reshape implementation.
    td1 = keras.layers.TimeDistributed(keras.layers.Dense(5))
    self.assertTrue(td1._always_use_reshape)

    # Built-in layers that are stateful don't use the reshape implementation.
    td2 = keras.layers.TimeDistributed(
        keras.layers.RNN(keras.layers.SimpleRNNCell(10), stateful=True))
    self.assertFalse(td2._always_use_reshape)

    # Custom layers are not whitelisted for the fast reshape implementation.
    td3 = keras.layers.TimeDistributed(NoReshapeLayer())
    self.assertFalse(td3._always_use_reshape)

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_TimeDistributed_output_shape_return_types(self):

    class TestLayer(keras.layers.Layer):

      def call(self, inputs):
        return array_ops.concat([inputs, inputs], axis=-1)

      def compute_output_shape(self, input_shape):
        output_shape = tensor_shape.TensorShape(input_shape).as_list()
        output_shape[-1] = output_shape[-1] * 2
        output_shape = tensor_shape.TensorShape(output_shape)
        return output_shape

    class TestListLayer(TestLayer):

      def compute_output_shape(self, input_shape):
        shape = super(TestListLayer, self).compute_output_shape(input_shape)
        return shape.as_list()

    class TestTupleLayer(TestLayer):

      def compute_output_shape(self, input_shape):
        shape = super(TestTupleLayer, self).compute_output_shape(input_shape)
        return tuple(shape.as_list())

    # Layers can specify output shape as list/tuple/TensorShape
    test_layers = [TestLayer, TestListLayer, TestTupleLayer]
    for layer in test_layers:
      input_layer = keras.layers.TimeDistributed(layer())
      inputs = keras.backend.placeholder(shape=(None, 2, 4))
      output = input_layer(inputs)
      self.assertEqual(output.shape.as_list(), [None, 2, 8])
      self.assertEqual(
          input_layer.compute_output_shape([None, 2, 4]).as_list(),
          [None, 2, 8])

  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  # TODO(scottzhu): check why v1 session failed.
  def test_TimeDistributed_with_mask_first_implementation(self):
    np.random.seed(100)
    rnn_layer = keras.layers.LSTM(4, return_sequences=True, stateful=True)

    data = np.array([[[[1.0], [1.0]], [[0.0], [1.0]]],
                     [[[1.0], [0.0]], [[1.0], [1.0]]],
                     [[[1.0], [0.0]], [[1.0], [1.0]]]])
    x = keras.layers.Input(shape=(2, 2, 1), batch_size=3)
    x_masking = keras.layers.Masking()(x)
    y = keras.layers.TimeDistributed(rnn_layer)(x_masking)
    model_1 = keras.models.Model(x, y)
    model_1.compile(
        'rmsprop',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    output_with_mask = model_1.predict(data, steps=1)

    y = keras.layers.TimeDistributed(rnn_layer)(x)
    model_2 = keras.models.Model(x, y)
    model_2.compile(
        'rmsprop',
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    output = model_2.predict(data, steps=1)

    self.assertNotAllClose(output_with_mask, output, atol=1e-7)

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(
      *tf_test_util.generate_combinations_with_testcase_name(
          layer=[keras.layers.LSTM,
                 keras.layers.Dense]))
  def test_TimeDistributed_with_ragged_input(self, layer):
    if context.executing_eagerly():
      self.skipTest('b/143103634')
    np.random.seed(100)
    layer = layer(4)
    ragged_data = ragged_factory_ops.constant(
        [[[[1.0], [1.0]], [[2.0], [2.0]]],
         [[[4.0], [4.0]], [[5.0], [5.0]], [[6.0], [6.0]]],
         [[[7.0], [7.0]], [[8.0], [8.0]], [[9.0], [9.0]]]],
        ragged_rank=1)

    x_ragged = keras.Input(shape=(None, 2, 1), dtype='float32', ragged=True)
    y_ragged = keras.layers.TimeDistributed(layer)(x_ragged)
    model_1 = keras.models.Model(x_ragged, y_ragged)
    model_1._run_eagerly = testing_utils.should_run_eagerly()
    output_ragged = model_1.predict(ragged_data, steps=1)

    x_dense = keras.Input(shape=(None, 2, 1), dtype='float32')
    masking = keras.layers.Masking()(x_dense)
    y_dense = keras.layers.TimeDistributed(layer)(masking)
    model_2 = keras.models.Model(x_dense, y_dense)
    dense_data = ragged_data.to_tensor()
    model_2._run_eagerly = testing_utils.should_run_eagerly()
    output_dense = model_2.predict(dense_data, steps=1)

    output_ragged = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        output_ragged, name='tensor')
    self.assertAllEqual(output_ragged.to_tensor(), output_dense)

  @keras_parameterized.run_all_keras_modes
  def test_TimeDistributed_with_ragged_input_with_batch_size(self):
    np.random.seed(100)
    layer = keras.layers.Dense(16)

    ragged_data = ragged_factory_ops.constant(
        [[[[1.0], [1.0]], [[2.0], [2.0]]],
         [[[4.0], [4.0]], [[5.0], [5.0]], [[6.0], [6.0]]],
         [[[7.0], [7.0]], [[8.0], [8.0]], [[9.0], [9.0]]]],
        ragged_rank=1)

    # Use the first implementation by specifying batch_size
    x_ragged = keras.Input(shape=(None, 2, 1), batch_size=3, dtype='float32',
                           ragged=True)
    y_ragged = keras.layers.TimeDistributed(layer)(x_ragged)
    model_1 = keras.models.Model(x_ragged, y_ragged)
    output_ragged = model_1.predict(ragged_data, steps=1)

    x_dense = keras.Input(shape=(None, 2, 1), batch_size=3, dtype='float32')
    masking = keras.layers.Masking()(x_dense)
    y_dense = keras.layers.TimeDistributed(layer)(masking)
    model_2 = keras.models.Model(x_dense, y_dense)
    dense_data = ragged_data.to_tensor()
    output_dense = model_2.predict(dense_data, steps=1)

    output_ragged = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        output_ragged, name='tensor')
    self.assertAllEqual(output_ragged.to_tensor(), output_dense)


@combinations.generate(combinations.combine(mode=['graph', 'eager']))
class BidirectionalTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(['sum', 'concat', 'ave', 'mul'])
  def test_bidirectional(self, mode):
    rnn = keras.layers.SimpleRNN
    samples = 2
    dim = 2
    timesteps = 2
    output_dim = 2
    with self.cached_session():
      x = np.random.random((samples, timesteps, dim))
      target_dim = 2 * output_dim if mode == 'concat' else output_dim
      y = np.random.random((samples, target_dim))

      # test with Sequential model
      model = keras.models.Sequential()
      model.add(
          keras.layers.Bidirectional(
              rnn(output_dim), merge_mode=mode, input_shape=(timesteps, dim)))
      model.compile(optimizer='rmsprop', loss='mse')
      model.fit(x, y, epochs=1, batch_size=1)

      # check whether the model variables are present in the
      # trackable list of objects
      checkpointed_objects = object_identity.ObjectIdentitySet(
          trackable_util.list_objects(model))
      for v in model.variables:
        self.assertIn(v, checkpointed_objects)

      # test compute output shape
      ref_shape = model.layers[-1].output.shape
      shape = model.layers[-1].compute_output_shape(
          (None, timesteps, dim))
      self.assertListEqual(shape.as_list(), ref_shape.as_list())

      # test config
      model.get_config()
      model = keras.models.model_from_json(model.to_json())
      model.summary()

  def test_bidirectional_invalid_init(self):
    x = constant_op.constant(np.zeros((1, 1)).astype('float32'))
    with self.assertRaisesRegex(
        ValueError,
        'Please initialize `Bidirectional` layer with a `Layer` instance.'):
      keras.layers.Bidirectional(x)

  def test_bidirectional_weight_loading(self):
    rnn = keras.layers.SimpleRNN
    samples = 2
    dim = 2
    timesteps = 2
    output_dim = 2
    with self.cached_session():
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

    with self.cached_session():
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

    with self.cached_session():
      x = np.random.random((samples, timesteps, dim))
      target_dim = 2 * output_dim if mode == 'concat' else output_dim
      y = np.random.random((samples, target_dim))

      inputs = keras.layers.Input(batch_shape=(1, timesteps, dim))
      bidi_rnn = keras.layers.Bidirectional(
          rnn(output_dim, stateful=True), merge_mode=mode)
      self.assertTrue(bidi_rnn.stateful)
      output = bidi_rnn(inputs)
      model = keras.models.Model(inputs, output)

      y_1 = model.predict(x, batch_size=1)
      model.reset_states()
      y_2 = model.predict(x, batch_size=1)

      self.assertAllClose(y_1, y_2)

      model.compile(loss='mse', optimizer='sgd')
      model.fit(x, y, epochs=1, batch_size=1)

  @parameterized.parameters(['sum', 'mul', 'ave', 'concat', None])
  def test_Bidirectional_merged_value(self, merge_mode):
    rnn = keras.layers.LSTM
    samples = 2
    dim = 5
    timesteps = 3
    units = 3
    x = [np.random.rand(samples, timesteps, dim)]

    with self.cached_session():
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
                                         [layer.forward_layer(inputs)])
      f_backward = keras.backend.function(
          [inputs],
          [keras.backend.reverse(layer.backward_layer(inputs), 1)])

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
                                         layer.forward_layer(inputs))
      f_backward = keras.backend.function([inputs],
                                          layer.backward_layer(inputs))
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

  @parameterized.parameters([True, False])
  def test_Bidirectional_with_time_major_input(self, time_major):
    batch_size, time, input_dim = 2, 3, 1
    inputs = array_ops.zeros((batch_size, time, input_dim))
    # length is [1 2]. Within the batch, the first element has 1 step, and the
    # second element as 2 steps.
    lengths = math_ops.range(1, 1 + batch_size)
    mask = array_ops.sequence_mask(lengths, maxlen=time, dtype=dtypes.float32)

    forward_cell = _AddOneCell(name='forward')
    backward_cell = _AddOneCell(name='backward')

    layer = keras.layers.Bidirectional(
        layer=keras.layers.RNN(
            forward_cell, time_major=time_major, return_sequences=True),
        backward_layer=keras.layers.RNN(
            backward_cell, time_major=time_major, return_sequences=True,
            go_backwards=True))

    # Switch to time-major.
    if time_major:
      inputs = array_ops.transpose(inputs, [1, 0, 2])
      mask = array_ops.transpose(mask, [1, 0])

    keras_outputs = layer(inputs, mask=mask)
    if time_major:
      keras_outputs = array_ops.transpose(keras_outputs, [1, 0, 2])

    # expect the first element in batch has 1 step and second element in batch
    # has 2 steps.
    expected_result = np.array([[[1., 1.], [0., 0.], [0., 0.]],
                                [[1., 1.], [1., 1.], [0., 0.]]])
    self.assertAllClose(expected_result, keras_outputs)

  def test_Bidirectional_dropout(self):
    rnn = keras.layers.LSTM
    samples = 2
    dim = 5
    timesteps = 3
    units = 3
    merge_mode = 'sum'
    x = [np.random.rand(samples, timesteps, dim)]

    with self.cached_session():
      inputs = keras.Input((timesteps, dim))
      wrapped = keras.layers.Bidirectional(
          rnn(units, dropout=0.2, recurrent_dropout=0.2), merge_mode=merge_mode)
      outputs = _to_list(wrapped(inputs, training=True))

      inputs = keras.Input((timesteps, dim))
      wrapped = keras.layers.Bidirectional(
          rnn(units, dropout=0.2, return_state=True), merge_mode=merge_mode)
      outputs = _to_list(wrapped(inputs))

      model = keras.Model(inputs, outputs)
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

    with self.cached_session():
      input1 = keras.layers.Input((timesteps, dim))
      layer = keras.layers.Bidirectional(
          rnn(units, return_state=True, return_sequences=True))
      state = layer(input1)[1:]

      # test passing invalid initial_state: passing a tensor
      input2 = keras.layers.Input((timesteps, dim))
      with self.assertRaises(ValueError):
        keras.layers.Bidirectional(rnn(units))(input2, initial_state=state[0])

      # test valid usage: passing a list
      output = keras.layers.Bidirectional(rnn(units))(input2,
                                                      initial_state=state)
      model = keras.models.Model([input1, input2], output)
      assert len(model.layers) == 4
      assert isinstance(model.layers[-1].input, list)
      inputs = [np.random.rand(samples, timesteps, dim),
                np.random.rand(samples, timesteps, dim)]
      model.predict(inputs)

  def test_Bidirectional_state_reuse_with_np_input(self):
    # See https://github.com/tensorflow/tensorflow/issues/28761 for more detail.
    rnn = keras.layers.LSTM
    samples = 2
    dim = 5
    timesteps = 3
    units = 3

    with self.cached_session():
      input1 = np.random.rand(samples, timesteps, dim).astype(np.float32)
      layer = keras.layers.Bidirectional(
          rnn(units, return_state=True, return_sequences=True))
      state = layer(input1)[1:]

      input2 = np.random.rand(samples, timesteps, dim).astype(np.float32)
      keras.layers.Bidirectional(rnn(units))(input2, initial_state=state)

  def test_Bidirectional_trainable(self):
    # test layers that need learning_phase to be set
    with self.cached_session():
      x = keras.layers.Input(shape=(3, 2))
      layer = keras.layers.Bidirectional(keras.layers.SimpleRNN(3))
      _ = layer(x)
      assert len(layer.trainable_weights) == 6
      layer.trainable = False
      assert not layer.trainable_weights
      layer.trainable = True
      assert len(layer.trainable_weights) == 6

  def test_Bidirectional_updates(self):
    if context.executing_eagerly():
      self.skipTest('layer.updates is only available in graph mode.')

    with self.cached_session():
      x = keras.layers.Input(shape=(3, 2))
      x_reachable_update = x * x
      layer = keras.layers.Bidirectional(keras.layers.SimpleRNN(3))
      _ = layer(x)
      assert not layer.updates
      # TODO(b/128684069): Remove when Wrapper sublayers are __call__'d.
      with base_layer_utils.call_context().enter(layer, x, True, None):
        layer.forward_layer.add_update(x_reachable_update, inputs=x)
        layer.forward_layer.add_update(1, inputs=None)
        layer.backward_layer.add_update(x_reachable_update, inputs=x)
        layer.backward_layer.add_update(1, inputs=None)
      assert len(layer.updates) == 4

  def test_Bidirectional_losses(self):
    x = keras.layers.Input(shape=(3, 2))
    layer = keras.layers.Bidirectional(
        keras.layers.SimpleRNN(
            3,
            kernel_regularizer='l1',
            bias_regularizer='l1',
            activity_regularizer='l1'))
    _ = layer(x)
    assert len(layer.losses) == 6

    loss = x * x
    layer.forward_layer.add_loss(loss)
    layer.backward_layer.add_loss(loss, inputs=x)
    assert len(layer.losses) == 8

  def test_Bidirectional_with_constants(self):
    with self.cached_session():
      # Test basic case.
      x = keras.Input((5, 5))
      c = keras.Input((3,))
      cell = _RNNCellWithConstants(32, 3)
      custom_objects = {'_RNNCellWithConstants': _RNNCellWithConstants}
      with generic_utils.CustomObjectScope(custom_objects):
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

      with generic_utils.CustomObjectScope(custom_objects):
        layer = keras.layers.Bidirectional.from_config(copy.deepcopy(config))
      y = layer(x, constants=c)
      model = keras.Model([x, c], y)
      model.set_weights(weights)
      y_np_2 = model.predict([x_np, c_np])
      self.assertAllClose(y_np, y_np_2, atol=1e-4)

      # Test flat list inputs
      with generic_utils.CustomObjectScope(custom_objects):
        layer = keras.layers.Bidirectional.from_config(copy.deepcopy(config))
      y = layer([x, c])
      model = keras.Model([x, c], y)
      model.set_weights(weights)
      y_np_3 = model.predict([x_np, c_np])
      self.assertAllClose(y_np, y_np_3, atol=1e-4)

  def test_Bidirectional_with_constants_layer_passing_initial_state(self):
    with self.cached_session():
      # Test basic case.
      x = keras.Input((5, 5))
      c = keras.Input((3,))
      s_for = keras.Input((32,))
      s_bac = keras.Input((32,))
      cell = _RNNCellWithConstants(32, 3)
      custom_objects = {'_RNNCellWithConstants': _RNNCellWithConstants}
      with generic_utils.CustomObjectScope(custom_objects):
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

      with generic_utils.CustomObjectScope(custom_objects):
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
      with generic_utils.CustomObjectScope(custom_objects):
        layer = keras.layers.Bidirectional.from_config(copy.deepcopy(config))
      y = layer([x, s_for, s_bac, c])
      model = keras.Model([x, s_for, s_bac, c], y)
      model.set_weights(weights)
      y_np_3 = model.predict([x_np, s_fw_np, s_bk_np, c_np])
      self.assertAllClose(y_np, y_np_3, atol=1e-4)

  def test_Bidirectional_output_shape_return_types(self):

    class TestLayer(keras.layers.SimpleRNN):

      def call(self, inputs):
        return array_ops.concat([inputs, inputs], axis=-1)

      def compute_output_shape(self, input_shape):
        output_shape = tensor_shape.TensorShape(input_shape).as_list()
        output_shape[-1] = output_shape[-1] * 2
        return tensor_shape.TensorShape(output_shape)

    class TestListLayer(TestLayer):

      def compute_output_shape(self, input_shape):
        shape = super(TestListLayer, self).compute_output_shape(input_shape)
        return shape.as_list()

    class TestTupleLayer(TestLayer):

      def compute_output_shape(self, input_shape):
        shape = super(TestTupleLayer, self).compute_output_shape(input_shape)
        return tuple(shape.as_list())

    # Layers can specify output shape as list/tuple/TensorShape
    test_layers = [TestLayer, TestListLayer, TestTupleLayer]
    for layer in test_layers:
      input_layer = keras.layers.Bidirectional(layer(1))
      inputs = keras.backend.placeholder(shape=(None, 2, 4))
      output = input_layer(inputs)
      self.assertEqual(output.shape.as_list(), [None, 2, 16])
      self.assertEqual(
          input_layer.compute_output_shape([None, 2, 4]).as_list(),
          [None, 2, 16])

  def test_Bidirectional_last_output_with_masking(self):
    if test.is_built_with_rocm():
      # testcase uses input and/or output sequences which require padding
      # leading to the following error on ROCm platform
      # ROCm MIOpen only supports packed input output
      # Skip this subtest for now
      self.skipTest('Test not supported on the ROCm platform')
    rnn = keras.layers.LSTM
    samples = 2
    dim = 5
    timesteps = 3
    units = 3
    merge_mode = 'concat'
    x = np.random.rand(samples, timesteps, dim)
    # clear the first record's timestep 2. Last output should be same as state,
    # not zeroed.
    x[0, 2] = 0

    with self.cached_session():
      inputs = keras.Input((timesteps, dim))
      masked_inputs = keras.layers.Masking()(inputs)
      wrapped = keras.layers.Bidirectional(
          rnn(units, return_state=True), merge_mode=merge_mode)
      outputs = _to_list(wrapped(masked_inputs, training=True))
      self.assertLen(outputs, 5)
      self.assertEqual(outputs[0].shape.as_list(), [None, units * 2])

      model = keras.Model(inputs, outputs)
      y = _to_list(model.predict(x))
      self.assertLen(y, 5)
      self.assertAllClose(y[0], np.concatenate([y[1], y[3]], axis=1))

  @parameterized.parameters([keras.layers.LSTM, keras.layers.GRU])
  def test_Bidirectional_sequence_output_with_masking(self, rnn):
    if test.is_built_with_rocm():
      # testcase uses input and/or output sequences which require padding
      # leading to the following error on ROCm platform
      # ROCm MIOpen only supports packed input output
      # Skip this subtest for now
      self.skipTest('Test not supported on the ROCm platform')
    samples = 2
    dim = 5
    timesteps = 3
    units = 3
    merge_mode = 'concat'
    x = np.random.rand(samples, timesteps, dim)
    # clear the first record's timestep 2, and expect the output of timestep 2
    # is also 0s.
    x[0, 2] = 0

    with self.cached_session():
      inputs = keras.Input((timesteps, dim))
      masked_inputs = keras.layers.Masking()(inputs)
      wrapped = keras.layers.Bidirectional(
          rnn(units, return_sequences=True),
          merge_mode=merge_mode)
      outputs = _to_list(wrapped(masked_inputs, training=True))
      self.assertLen(outputs, 1)
      self.assertEqual(outputs[0].shape.as_list(), [None, timesteps, units * 2])

      model = keras.Model(inputs, outputs)
      y = _to_list(model.predict(x))
      self.assertLen(y, 1)
      self.assertAllClose(y[0][0, 2], np.zeros(units * 2))

  @parameterized.parameters(['sum', 'concat'])
  def test_custom_backward_layer(self, mode):
    rnn = keras.layers.SimpleRNN
    samples = 2
    dim = 2
    timesteps = 2
    output_dim = 2

    x = np.random.random((samples, timesteps, dim))
    target_dim = 2 * output_dim if mode == 'concat' else output_dim
    y = np.random.random((samples, target_dim))
    forward_layer = rnn(output_dim)
    backward_layer = rnn(output_dim, go_backwards=True)

    # test with Sequential model
    model = keras.models.Sequential()
    model.add(
        keras.layers.Bidirectional(
            forward_layer,
            merge_mode=mode,
            backward_layer=backward_layer,
            input_shape=(timesteps, dim)))
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(x, y, epochs=1, batch_size=1)

    # check whether the model variables are present in the
    # trackable list of objects
    checkpointed_objects = object_identity.ObjectIdentitySet(
        trackable_util.list_objects(model))
    for v in model.variables:
      self.assertIn(v, checkpointed_objects)

    # test compute output shape
    ref_shape = model.layers[-1].output.shape
    shape = model.layers[-1].compute_output_shape((None, timesteps, dim))
    self.assertListEqual(shape.as_list(), ref_shape.as_list())

    # test config
    model.get_config()
    model = keras.models.model_from_json(model.to_json())
    model.summary()

  def test_custom_backward_layer_error_check(self):
    rnn = keras.layers.LSTM
    units = 2

    forward_layer = rnn(units)
    backward_layer = rnn(units)

    with self.assertRaisesRegex(ValueError,
                                'should have different `go_backwards` value.'):
      keras.layers.Bidirectional(
          forward_layer, merge_mode='concat', backward_layer=backward_layer)

    for attr in ('stateful', 'return_sequences', 'return_state'):
      kwargs = {attr: True}
      backward_layer = rnn(units, go_backwards=True, **kwargs)
      with self.assertRaisesRegex(
          ValueError, 'expected to have the same value for attribute ' + attr):
        keras.layers.Bidirectional(
            forward_layer, merge_mode='concat', backward_layer=backward_layer)

  def test_custom_backward_layer_serialization(self):
    rnn = keras.layers.LSTM
    units = 2

    forward_layer = rnn(units)
    backward_layer = rnn(units, go_backwards=True)
    layer = keras.layers.Bidirectional(
        forward_layer, merge_mode='concat', backward_layer=backward_layer)
    config = layer.get_config()
    layer_from_config = keras.layers.Bidirectional.from_config(config)
    new_config = layer_from_config.get_config()
    self.assertDictEqual(config, new_config)

  def test_rnn_layer_name(self):
    rnn = keras.layers.LSTM
    units = 2

    layer = keras.layers.Bidirectional(rnn(units, name='rnn'))
    config = layer.get_config()

    self.assertEqual(config['layer']['config']['name'], 'rnn')

    layer_from_config = keras.layers.Bidirectional.from_config(config)
    self.assertEqual(layer_from_config.forward_layer.name, 'forward_rnn')
    self.assertEqual(layer_from_config.backward_layer.name, 'backward_rnn')

  def test_custom_backward_rnn_layer_name(self):
    rnn = keras.layers.LSTM
    units = 2

    forward_layer = rnn(units)
    backward_layer = rnn(units, go_backwards=True)
    layer = keras.layers.Bidirectional(
        forward_layer, merge_mode='concat', backward_layer=backward_layer)
    config = layer.get_config()

    self.assertEqual(config['layer']['config']['name'], 'lstm')
    self.assertEqual(config['backward_layer']['config']['name'], 'lstm_1')

    layer_from_config = keras.layers.Bidirectional.from_config(config)
    self.assertEqual(layer_from_config.forward_layer.name, 'forward_lstm')
    self.assertEqual(layer_from_config.backward_layer.name, 'backward_lstm_1')

  def test_rnn_with_customized_cell(self):
    batch = 20
    dim = 5
    timesteps = 3
    units = 5
    merge_mode = 'sum'

    cell = _ResidualLSTMCell(units)
    forward_layer = keras.layers.RNN(cell)
    inputs = keras.Input((timesteps, dim))
    bidirectional_rnn = keras.layers.Bidirectional(
        forward_layer, merge_mode=merge_mode)
    outputs = _to_list(bidirectional_rnn(inputs))

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(
        np.random.random((batch, timesteps, dim)),
        np.random.random((batch, units)),
        epochs=1,
        batch_size=10)

  def test_rnn_with_customized_cell_stacking(self):
    batch = 20
    dim = 5
    timesteps = 3
    units = 5
    merge_mode = 'sum'

    cell = [_ResidualLSTMCell(units), _ResidualLSTMCell(units)]
    forward_layer = keras.layers.RNN(cell)
    inputs = keras.Input((timesteps, dim))
    bidirectional_rnn = keras.layers.Bidirectional(
        forward_layer, merge_mode=merge_mode)
    outputs = _to_list(bidirectional_rnn(inputs))

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(
        np.random.random((batch, timesteps, dim)),
        np.random.random((batch, units)),
        epochs=1,
        batch_size=10)

  @tf_test_util.run_v2_only
  def test_wrapped_rnn_cell(self):
    # See https://github.com/tensorflow/tensorflow/issues/26581.
    batch = 20
    dim = 5
    timesteps = 3
    units = 5
    merge_mode = 'sum'

    cell = keras.layers.LSTMCell(units)
    cell = ResidualWrapper(cell)
    rnn = keras.layers.RNN(cell)

    inputs = keras.Input((timesteps, dim))
    wrapped = keras.layers.Bidirectional(rnn, merge_mode=merge_mode)
    outputs = _to_list(wrapped(inputs))

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(
        np.random.random((batch, timesteps, dim)),
        np.random.random((batch, units)),
        epochs=1,
        batch_size=10)

  @parameterized.parameters(['ave', 'concat', 'mul'])
  def test_Bidirectional_ragged_input(self, merge_mode):
    if test.is_built_with_rocm():
      # ragged tenors are not supported in ROCM RNN implementation
      self.skipTest('Test not supported on the ROCm platform')
    np.random.seed(100)
    rnn = keras.layers.LSTM
    units = 3
    x = ragged_factory_ops.constant(
        [[[1, 1, 1], [1, 1, 1]], [[1, 1, 1]],
         [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
         [[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
        ragged_rank=1)
    x = math_ops.cast(x, 'float32')

    # pylint: disable=g-long-lambda
    with self.cached_session():
      if merge_mode == 'ave':
        merge_func = lambda y, y_rev: (y + y_rev) / 2
      elif merge_mode == 'concat':
        merge_func = lambda y, y_rev: ragged_concat_ops.concat(
            (y, y_rev), axis=-1)
      elif merge_mode == 'mul':
        merge_func = lambda y, y_rev: (y * y_rev)
        # pylint: enable=g-long-lambda

      inputs = keras.Input(
          shape=(None, 3), batch_size=4, dtype='float32', ragged=True)
      layer = keras.layers.Bidirectional(
          rnn(units, return_sequences=True), merge_mode=merge_mode)
      f_merged = keras.backend.function([inputs], layer(inputs))
      f_forward = keras.backend.function([inputs],
                                         layer.forward_layer(inputs))

      # TODO(kaftan): after KerasTensor refactor TF op layers should work
      # with many composite tensors, and this shouldn't need to be a lambda
      # layer.
      reverse_layer = core.Lambda(array_ops.reverse, arguments=dict(axis=[1]))
      f_backward = keras.backend.function(
          [inputs],
          reverse_layer(layer.backward_layer(inputs)))

      y_merged = f_merged(x)
      y_expected = merge_func(
          ragged_tensor.convert_to_tensor_or_ragged_tensor(f_forward(x)),
          ragged_tensor.convert_to_tensor_or_ragged_tensor(f_backward(x)))

      y_merged = ragged_tensor.convert_to_tensor_or_ragged_tensor(y_merged)
      self.assertAllClose(y_merged.flat_values, y_expected.flat_values)

  def test_full_input_spec(self):
    # See https://github.com/tensorflow/tensorflow/issues/38403
    inputs = keras.layers.Input(batch_shape=(1, 1, 1))
    fw_state = keras.layers.Input(batch_shape=(1, 1))
    bw_state = keras.layers.Input(batch_shape=(1, 1))
    states = [fw_state, bw_state]
    bidirectional_rnn = keras.layers.Bidirectional(
        keras.layers.SimpleRNN(1, stateful=True))

    rnn_output = bidirectional_rnn(inputs, initial_state=states)
    model = keras.Model([inputs, fw_state, bw_state], rnn_output)
    output1 = model.predict(
        [np.ones((1, 1, 1)), np.ones((1, 1)), np.ones((1, 1))])
    output2 = model.predict(
        [np.ones((1, 1, 1)), np.ones((1, 1)), np.ones((1, 1))])
    model.reset_states()
    output3 = model.predict(
        [np.ones((1, 1, 1)), np.ones((1, 1)), np.ones((1, 1))])
    self.assertAllClose(output1, output3)
    self.assertNotAllClose(output1, output2)


class ExampleWrapper(keras.layers.Wrapper):
  """Simple Wrapper subclass."""

  def call(self, inputs, *args, **kwargs):
    return self.layer(inputs, *args, **kwargs)


class WrapperTest(parameterized.TestCase):

  def test_wrapper_from_config_no_mutation(self):
    wrapper = ExampleWrapper(keras.layers.Dense(1))
    config = wrapper.get_config()
    config_copy = config.copy()
    self.assertEqual(config, config_copy)

    wrapper_from_config = ExampleWrapper.from_config(config)
    new_config = wrapper.get_config()
    self.assertEqual(new_config, config_copy)
    self.assertEqual(config, config_copy)


def _to_list(ls):
  if isinstance(ls, list):
    return ls
  else:
    return [ls]


if __name__ == '__main__':
  test.main()
