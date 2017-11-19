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
import shutil
import tempfile

import numpy as np

from tensorflow.python.keras._impl import keras
from tensorflow.python.platform import test
from tensorflow.python.training import training as training_module

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
      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)

      new_model = keras.models.load_model(fname)
      os.close(fd)
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
      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)

      model = keras.models.load_model(
          fname,
          custom_objects={'CustomOp': CustomOp,
                          'custom_loss': custom_loss})
      os.close(fd)
      os.remove(fname)

      out2 = model.predict(x)
      self.assertAllClose(out, out2, atol=1e-05)

  def test_functional_model_saving(self):
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
      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)

      model = keras.models.load_model(fname)
      os.close(fd)
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

      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)
      model = keras.models.load_model(fname)
      os.close(fd)
      os.remove(fname)

  def test_saving_with_tf_optimizer(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss='mse',
                    optimizer=training_module.AdadeltaOptimizer(0.1),
                    metrics=['acc'])

      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)
      model = keras.models.load_model(fname)
      os.close(fd)
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

      fd, fname = tempfile.mkstemp('.h5')
      keras.models.save_model(model, fname)
      model = keras.models.load_model(fname)
      os.close(fd)
      os.remove(fname)

  def test_saving_lambda_numpy_array_arguments(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    mean = np.random.random((4, 2, 3))
    std = np.abs(np.random.random((4, 2, 3))) + 1e-5
    inputs = keras.layers.Input(shape=(4, 2, 3))
    output = keras.layers.Lambda(lambda image, mu, std: (image - mu) / std,
                                 arguments={'mu': mean, 'std': std})(inputs)
    model = keras.models.Model(inputs, output)
    model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

    fd, fname = tempfile.mkstemp('.h5')
    keras.models.save_model(model, fname)

    model = keras.models.load_model(fname)
    os.close(fd)
    os.remove(fname)

    self.assertAllClose(mean, model.layers[1].arguments['mu'])
    self.assertAllClose(std, model.layers[1].arguments['std'])


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

  def test_sequential_weight_loading(self):
    if h5py is None:
      return

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)
    h5_path = os.path.join(temp_dir, 'test.h5')

    num_hidden = 5
    input_dim = 3
    batch_size = 5
    num_classes = 2

    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(num_hidden, input_dim=input_dim))
      model.add(keras.layers.Dense(num_classes))

      x = np.random.random((batch_size, input_dim))
      ref_y = model.predict(x)

      model.save_weights(h5_path)

      model = keras.models.Sequential()
      model.add(keras.layers.Dense(num_hidden, input_dim=input_dim))
      model.add(keras.layers.Dense(num_classes))
      model.load_weights(h5_path)
      y = model.predict(x)

      self.assertAllClose(y, ref_y)

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

        def _compute_output_shape(self, input_shape):
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


class TestModelCloning(test.TestCase):

  def test_clone_sequential_model(self):
    with self.test_session():
      val_a = np.random.random((10, 4))
      val_out = np.random.random((10, 4))

      model = keras.models.Sequential()
      model.add(keras.layers.Dense(4, input_shape=(4,)))
      model.add(keras.layers.Dropout(0.5))
      model.add(keras.layers.Dense(4))

    # Everything should work in a new session.
    keras.backend.clear_session()

    with self.test_session():
      # With placeholder creation
      new_model = keras.models.clone_model(model)
      new_model.compile('rmsprop', 'mse')
      new_model.train_on_batch(val_a, val_out)

      # On top of new tensor
      input_a = keras.Input(shape=(4,))
      new_model = keras.models.clone_model(
          model, input_tensors=input_a)
      new_model.compile('rmsprop', 'mse')
      new_model.train_on_batch(val_a, val_out)

      # On top of new, non-Keras tensor
      input_a = keras.backend.variable(val_a)
      new_model = keras.models.clone_model(
          model, input_tensors=input_a)
      new_model.compile('rmsprop', 'mse')
      new_model.train_on_batch(None, val_out)

  def test_clone_functional_model(self):
    with self.test_session():
      val_a = np.random.random((10, 4))
      val_b = np.random.random((10, 4))
      val_out = np.random.random((10, 4))

      input_a = keras.Input(shape=(4,))
      input_b = keras.Input(shape=(4,))
      dense_1 = keras.layers.Dense(4,)
      dense_2 = keras.layers.Dense(4,)

      x_a = dense_1(input_a)
      x_a = keras.layers.Dropout(0.5)(x_a)
      x_b = dense_1(input_b)
      x_a = dense_2(x_a)
      outputs = keras.layers.add([x_a, x_b])
      model = keras.models.Model([input_a, input_b], outputs)

    # Everything should work in a new session.
    keras.backend.clear_session()

    with self.test_session():
      # With placeholder creation
      new_model = keras.models.clone_model(model)
      new_model.compile('rmsprop', 'mse')
      new_model.train_on_batch([val_a, val_b], val_out)

      # On top of new tensors
      input_a = keras.Input(shape=(4,), name='a')
      input_b = keras.Input(shape=(4,), name='b')
      new_model = keras.models.clone_model(
          model, input_tensors=[input_a, input_b])
      new_model.compile('rmsprop', 'mse')
      new_model.train_on_batch([val_a, val_b], val_out)

      # On top of new, non-Keras tensors
      input_a = keras.backend.variable(val_a)
      input_b = keras.backend.variable(val_b)
      new_model = keras.models.clone_model(
          model, input_tensors=[input_a, input_b])
      new_model.compile('rmsprop', 'mse')
      new_model.train_on_batch(None, val_out)

  def test_model_cloning_invalid_use_cases(self):
    seq_model = keras.models.Sequential()
    seq_model.add(keras.layers.Dense(4, input_shape=(4,)))

    x = keras.Input((4,))
    y = keras.layers.Dense(4)(x)
    fn_model = keras.models.Model(x, y)

    with self.assertRaises(ValueError):
      keras.models._clone_functional_model(seq_model)
    with self.assertRaises(ValueError):
      keras.models._clone_functional_model(None)
    with self.assertRaises(ValueError):
      keras.models._clone_sequential_model(fn_model)

    with self.assertRaises(ValueError):
      keras.models._clone_sequential_model(seq_model, input_tensors=[x, x])
    with self.assertRaises(ValueError):
      keras.models._clone_sequential_model(seq_model, input_tensors=y)


if __name__ == '__main__':
  test.main()
