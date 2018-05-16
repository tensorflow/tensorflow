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
"""Tests for `models.py` (model cloning, mainly)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import adam


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


class CheckpointingTests(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def test_optimizer_dependency(self):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, input_shape=(4,)))
    opt = adam.AdamOptimizer(0.01)
    model.compile(optimizer=opt, loss='mse')
    model.fit(x=np.array([[1., 2., 3., 4.]]), y=[1.], epochs=2)
    save_prefix = os.path.join(self.get_temp_dir(), 'ckpt')
    beta1_power, _ = opt._get_beta_accumulators()
    self.evaluate(beta1_power.assign(12.))
    model.save_weights(save_prefix)
    self.evaluate(beta1_power.assign(13.))
    model.load_weights(save_prefix)
    self.assertEqual(12., self.evaluate(beta1_power))

class TestModelBackend(test.TestCase):

  def test_model_backend_float64_use_cases(self):
    # Test case for GitHub issue 19318
    floatx = keras.backend.floatx()
    keras.backend.set_floatx('float64')

    x = keras.Input((5,))
    y = keras.layers.Dense(1)(x)
    model = keras.models.Model(x, y)
    model.compile('rmsprop', 'mse')

    keras.backend.set_floatx(floatx)

if __name__ == '__main__':
  test.main()
