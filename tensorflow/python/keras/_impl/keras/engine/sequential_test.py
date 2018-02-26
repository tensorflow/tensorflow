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
"""Tests specific to `Sequential` model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras._impl import keras
from tensorflow.python.platform import test


class TestSequential(test.TestCase):
  """Most Sequential model API tests are covered in `training_test.py`.
  """

  def test_basic_methods(self):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, input_dim=2))
    model.add(keras.layers.Dropout(0.3, name='dp'))
    model.add(keras.layers.Dense(2, kernel_regularizer='l2',
                                 kernel_constraint='max_norm'))
    model.build()
    self.assertEqual(model.state_updates, model.model.state_updates)
    self.assertEqual(model.get_layer(name='dp').name, 'dp')

  def test_sequential_pop(self):
    num_hidden = 5
    input_dim = 3
    batch_size = 5
    num_classes = 2
    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(num_hidden, input_dim=input_dim))
      model.add(keras.layers.Dense(num_classes))
      model.compile(loss='mse', optimizer='sgd')
      x = np.random.random((batch_size, input_dim))
      y = np.random.random((batch_size, num_classes))
      model.fit(x, y, epochs=1)
      model.pop()
      self.assertEqual(len(model.layers), 1)
      self.assertEqual(model.output_shape, (None, num_hidden))
      model.compile(loss='mse', optimizer='sgd')
      y = np.random.random((batch_size, num_hidden))
      model.fit(x, y, epochs=1)

      # Test popping single-layer model
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(num_hidden, input_dim=input_dim))
      model.pop()
      self.assertEqual(len(model.layers), 0)
      self.assertEqual(len(model.outputs), 0)

      # Invalid use case
      model = keras.models.Sequential()
      with self.assertRaises(TypeError):
        model.pop()

  def test_invalid_use_cases(self):
    with self.test_session():
      # Added objects must be layer instances
      with self.assertRaises(TypeError):
        model = keras.models.Sequential()
        model.add(None)

      # Added layers must have an inputs shape
      with self.assertRaises(ValueError):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(1))

      # Added layers cannot have multiple outputs
      class MyLayer(keras.layers.Layer):

        def call(self, inputs):
          return [3 * inputs, 2 * inputs]

        def compute_output_shape(self, input_shape):
          return [input_shape, input_shape]

      with self.assertRaises(ValueError):
        model = keras.models.Sequential()
        model.add(MyLayer(input_shape=(3,)))
      with self.assertRaises(TypeError):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(1, input_dim=1))
        model.add(MyLayer())

      # Building empty model
      model = keras.models.Sequential()
      with self.assertRaises(TypeError):
        model.build()

  def test_nested_sequential_trainability(self):
    input_dim = 20
    num_units = 10
    num_classes = 2

    inner_model = keras.models.Sequential()
    inner_model.add(keras.layers.Dense(num_units, input_shape=(input_dim,)))

    model = keras.models.Sequential()
    model.add(inner_model)
    model.add(keras.layers.Dense(num_classes))

    self.assertEqual(len(model.trainable_weights), 4)
    inner_model.trainable = False
    self.assertEqual(len(model.trainable_weights), 2)
    inner_model.trainable = True
    self.assertEqual(len(model.trainable_weights), 4)

  def test_sequential_update_disabling(self):
    val_a = np.random.random((10, 4))
    val_out = np.random.random((10, 4))

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.BatchNormalization(input_shape=(4,)))

      model.trainable = False
      assert not model.updates

      model.compile('sgd', 'mse')
      assert not model.updates
      assert not model.model.updates

      x1 = model.predict(val_a)
      model.train_on_batch(val_a, val_out)
      x2 = model.predict(val_a)
      self.assertAllClose(x1, x2, atol=1e-7)

      model.trainable = True
      model.compile('sgd', 'mse')
      assert model.updates
      assert model.model.updates

      model.train_on_batch(val_a, val_out)
      x2 = model.predict(val_a)
      assert np.abs(np.sum(x1 - x2)) > 1e-5
