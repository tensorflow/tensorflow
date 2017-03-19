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
"""Tests for training routines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import numpy as np

from tensorflow.contrib.keras.python import keras
from tensorflow.python.platform import test

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None


class TestModelSaving(test.TestCase):

  def test_sequential_model_saving(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.RepeatVector(3))
      model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
      model.compile(loss=keras.losses.MSE,
                    optimizer=keras.optimizers.RMSprop(lr=0.0001),
                    metrics=[keras.metrics.categorical_accuracy],
                    sample_weight_mode='temporal')
      x = np.random.random((1, 3))
      y = np.random.random((1, 3, 3))
      model.train_on_batch(x, y)

      out = model.predict(x)
      _, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)

      new_model = keras.models.load_model(fname)
      os.remove(fname)

      out2 = new_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

      # test that new updates are the same with both models
      x = np.random.random((1, 3))
      y = np.random.random((1, 3, 3))
      model.train_on_batch(x, y)
      new_model.train_on_batch(x, y)
      out = model.predict(x)
      out2 = new_model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

  def test_sequential_model_saving_2(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    with self.test_session():
      # test with custom optimizer, loss

      class CustomOp(keras.optimizers.RMSprop):
        pass

      def custom_loss(y_true, y_pred):
        return keras.losses.mse(y_true, y_pred)

      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss=custom_loss, optimizer=CustomOp(), metrics=['acc'])

      x = np.random.random((1, 3))
      y = np.random.random((1, 3))
      model.train_on_batch(x, y)

      out = model.predict(x)
      _, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)

      model = keras.models.load_model(
          fname,
          custom_objects={'CustomOp': CustomOp,
                          'custom_loss': custom_loss})
      os.remove(fname)

      out2 = model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

  def test_fuctional_model_saving(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    with self.test_session():
      inputs = keras.layers.Input(shape=(3,))
      x = keras.layers.Dense(2)(inputs)
      output = keras.layers.Dense(3)(x)

      model = keras.models.Model(inputs, output)
      model.compile(loss=keras.losses.MSE,
                    optimizer=keras.optimizers.RMSprop(lr=0.0001),
                    metrics=[keras.metrics.categorical_accuracy])
      x = np.random.random((1, 3))
      y = np.random.random((1, 3))
      model.train_on_batch(x, y)

      out = model.predict(x)
      _, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)

      model = keras.models.load_model(fname)
      os.remove(fname)

      out2 = model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

  def test_saving_without_compilation(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

      _, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)
      model = keras.models.load_model(fname)
      os.remove(fname)

  def test_saving_right_after_compilation(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss='mse', optimizer='sgd', metrics=['acc'])
      model.model._make_train_function()

      _, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)
      model = keras.models.load_model(fname)
      os.remove(fname)


class TestSequential(test.TestCase):
  """Most Sequential model API tests are covered in `training_test.py`.
  """

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


if __name__ == '__main__':
  test.main()
