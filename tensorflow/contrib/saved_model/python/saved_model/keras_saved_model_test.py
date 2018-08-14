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
# pylint: disable=protected-access
"""Tests for saving/loading function for keras Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import numpy as np

from tensorflow.contrib.saved_model.python.saved_model import keras_saved_model
from tensorflow.python import keras
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import training
from tensorflow.python.platform import test
from tensorflow.python.training import training as training_module


class TestModelSavingandLoading(test.TestCase):

  def test_saving_sequential_model(self):
    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.RepeatVector(3))
      model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
      model.compile(
          loss=keras.losses.MSE,
          optimizer=keras.optimizers.RMSprop(lr=0.0001),
          metrics=[keras.metrics.categorical_accuracy],
          sample_weight_mode='temporal')
      x = np.random.random((1, 3))
      y = np.random.random((1, 3, 3))
      model.train_on_batch(x, y)

      ref_y = model.predict(x)
      temp_dir = self.get_temp_dir()
      self.addCleanup(shutil.rmtree, temp_dir)

      temp_saved_model = os.path.join(temp_dir, 'saved_model')
      keras_saved_model.save_model(model, temp_saved_model)

      loaded_model = keras_saved_model.load_model(temp_saved_model)
      y = loaded_model.predict(x)
      self.assertAllClose(ref_y, y, atol=1e-05)

  @test_util.run_in_graph_and_eager_modes
  def test_saving_sequential_model_without_compile(self):
    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.RepeatVector(3))
      model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))

      x = np.random.random((1, 3))
      ref_y = model.predict(x)

      temp_dir = self.get_temp_dir()
      self.addCleanup(shutil.rmtree, temp_dir)

      temp_saved_model = os.path.join(temp_dir, 'saved_model')
      keras_saved_model.save_model(model, temp_saved_model)
      loaded_model = keras_saved_model.load_model(temp_saved_model)

      y = loaded_model.predict(x)
      self.assertAllClose(ref_y, y, atol=1e-05)

  def test_saving_functional_model(self):
    with self.test_session():
      inputs = keras.layers.Input(shape=(3,))
      x = keras.layers.Dense(2)(inputs)
      output = keras.layers.Dense(3)(x)

      model = keras.models.Model(inputs, output)
      model.compile(
          loss=keras.losses.MSE,
          optimizer=keras.optimizers.RMSprop(lr=0.0001),
          metrics=[keras.metrics.categorical_accuracy])
      x = np.random.random((1, 3))
      y = np.random.random((1, 3))
      model.train_on_batch(x, y)

      ref_y = model.predict(x)
      temp_dir = self.get_temp_dir()
      self.addCleanup(shutil.rmtree, temp_dir)

      temp_saved_model = os.path.join(temp_dir, 'saved_model')
      keras_saved_model.save_model(model, temp_saved_model)
      loaded_model = keras_saved_model.load_model(temp_saved_model)

      y = loaded_model.predict(x)
      self.assertAllClose(ref_y, y, atol=1e-05)

  @test_util.run_in_graph_and_eager_modes
  def test_saving_functional_model_without_compile(self):
    with self.test_session():
      inputs = keras.layers.Input(shape=(3,))
      x = keras.layers.Dense(2)(inputs)
      output = keras.layers.Dense(3)(x)

      model = keras.models.Model(inputs, output)

      x = np.random.random((1, 3))
      y = np.random.random((1, 3))

      ref_y = model.predict(x)
      temp_dir = self.get_temp_dir()
      self.addCleanup(shutil.rmtree, temp_dir)

      temp_saved_model = os.path.join(temp_dir, 'saved_model')
      keras_saved_model.save_model(model, temp_saved_model)
      loaded_model = keras_saved_model.load_model(temp_saved_model)

      y = loaded_model.predict(x)
      self.assertAllClose(ref_y, y, atol=1e-05)

  @test_util.run_in_graph_and_eager_modes
  def test_saving_with_tf_optimizer(self):
    with self.test_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(
          loss='mse',
          optimizer=training_module.RMSPropOptimizer(0.1),
          metrics=['acc'])

      x = np.random.random((1, 3))
      y = np.random.random((1, 3))
      model.train_on_batch(x, y)

      ref_y = model.predict(x)
      temp_dir = self.get_temp_dir()
      self.addCleanup(shutil.rmtree, temp_dir)

      temp_saved_model = os.path.join(temp_dir, 'saved_model')
      keras_saved_model.save_model(model, temp_saved_model)
      loaded_model = keras_saved_model.load_model(temp_saved_model)
      loaded_model.compile(
          loss='mse',
          optimizer=training_module.RMSPropOptimizer(0.1),
          metrics=['acc'])
      y = loaded_model.predict(x)
      self.assertAllClose(ref_y, y, atol=1e-05)

      # test that new updates are the same with both models
      x = np.random.random((1, 3))
      y = np.random.random((1, 3))

      ref_loss = model.train_on_batch(x, y)
      loss = loaded_model.train_on_batch(x, y)
      self.assertAllClose(ref_loss, loss, atol=1e-05)

      ref_y = model.predict(x)
      y = loaded_model.predict(x)
      self.assertAllClose(ref_y, y, atol=1e-05)

      # test saving/loading again
      keras_saved_model.save_model(loaded_model, temp_saved_model)
      loaded_model = keras_saved_model.load_model(temp_saved_model)
      y = loaded_model.predict(x)
      self.assertAllClose(ref_y, y, atol=1e-05)

  def test_saving_subclassed_model_raise_error(self):
    # For now, saving subclassed model should raise an error. It should be
    # avoided later with loading from SavedModel.pb.

    class SubclassedModel(training.Model):

      def __init__(self):
        super(SubclassedModel, self).__init__()
        self.layer1 = keras.layers.Dense(3)
        self.layer2 = keras.layers.Dense(1)

      def call(self, inp):
        return self.layer2(self.layer1(inp))

    model = SubclassedModel()
    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)
    temp_saved_model = os.path.join(temp_dir, 'saved_model')
    with self.assertRaises(NotImplementedError):
      keras_saved_model.save_model(model, temp_saved_model)


if __name__ == '__main__':
  test.main()
