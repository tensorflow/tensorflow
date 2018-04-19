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
"""Tests for normalization layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras._impl import keras
from tensorflow.python.keras._impl.keras import testing_utils
from tensorflow.python.platform import test


class NormalizationLayersTest(test.TestCase):

  def test_basic_batchnorm(self):
    with self.test_session():
      testing_utils.layer_test(
          keras.layers.BatchNormalization,
          kwargs={
              'momentum': 0.9,
              'epsilon': 0.1,
              'gamma_regularizer': keras.regularizers.l2(0.01),
              'beta_regularizer': keras.regularizers.l2(0.01)
          },
          input_shape=(3, 4, 2))
      testing_utils.layer_test(
          keras.layers.BatchNormalization,
          kwargs={
              'gamma_initializer': 'ones',
              'beta_initializer': 'ones',
              'moving_mean_initializer': 'zeros',
              'moving_variance_initializer': 'ones'
          },
          input_shape=(3, 4, 2))
      testing_utils.layer_test(
          keras.layers.BatchNormalization,
          kwargs={'scale': False,
                  'center': False},
          input_shape=(3, 3))

  def test_batchnorm_weights(self):
    with self.test_session():
      layer = keras.layers.BatchNormalization(scale=False, center=False)
      layer.build((None, 3, 4))
      self.assertEqual(len(layer.trainable_weights), 0)
      self.assertEqual(len(layer.weights), 2)

      layer = keras.layers.BatchNormalization()
      layer.build((None, 3, 4))
      self.assertEqual(len(layer.trainable_weights), 2)
      self.assertEqual(len(layer.weights), 4)

  def test_batchnorm_regularization(self):
    with self.test_session():
      layer = keras.layers.BatchNormalization(
          gamma_regularizer='l1', beta_regularizer='l1')
      layer.build((None, 3, 4))
      self.assertEqual(len(layer.losses), 2)
      max_norm = keras.constraints.max_norm
      layer = keras.layers.BatchNormalization(
          gamma_constraint=max_norm, beta_constraint=max_norm)
      layer.build((None, 3, 4))
      self.assertEqual(layer.gamma.constraint, max_norm)
      self.assertEqual(layer.beta.constraint, max_norm)

  def test_batchnorm_correctness(self):
    with self.test_session():
      model = keras.models.Sequential()
      norm = keras.layers.BatchNormalization(input_shape=(10,), momentum=0.8)
      model.add(norm)
      model.compile(loss='mse', optimizer='sgd')

      # centered on 5.0, variance 10.0
      x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10))
      model.fit(x, x, epochs=4, verbose=0)
      out = model.predict(x)
      out -= keras.backend.eval(norm.beta)
      out /= keras.backend.eval(norm.gamma)

      np.testing.assert_allclose(out.mean(), 0.0, atol=1e-1)
      np.testing.assert_allclose(out.std(), 1.0, atol=1e-1)

  def test_batchnorm_convnet(self):
    if test.is_gpu_available(cuda_only=True):
      with self.test_session(use_gpu=True):
        model = keras.models.Sequential()
        norm = keras.layers.BatchNormalization(
            axis=1, input_shape=(3, 4, 4), momentum=0.8)
        model.add(norm)
        model.compile(loss='mse', optimizer='sgd')

        # centered on 5.0, variance 10.0
        x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 3, 4, 4))
        model.fit(x, x, epochs=4, verbose=0)
        out = model.predict(x)
        out -= np.reshape(keras.backend.eval(norm.beta), (1, 3, 1, 1))
        out /= np.reshape(keras.backend.eval(norm.gamma), (1, 3, 1, 1))

        np.testing.assert_allclose(np.mean(out, axis=(0, 2, 3)), 0.0, atol=1e-1)
        np.testing.assert_allclose(np.std(out, axis=(0, 2, 3)), 1.0, atol=1e-1)

  def test_batchnorm_convnet_channel_last(self):
    with self.test_session():
      # keras.backend.set_learning_phase(True)

      model = keras.models.Sequential()
      norm = keras.layers.BatchNormalization(
          axis=-1, input_shape=(4, 4, 3), momentum=0.8)
      model.add(norm)
      model.compile(loss='mse', optimizer='sgd')

      # centered on 5.0, variance 10.0
      x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 4, 4, 3))
      model.fit(x, x, epochs=4, verbose=0)
      out = model.predict(x)
      out -= np.reshape(keras.backend.eval(norm.beta), (1, 1, 1, 3))
      out /= np.reshape(keras.backend.eval(norm.gamma), (1, 1, 1, 3))

      np.testing.assert_allclose(np.mean(out, axis=(0, 1, 2)), 0.0, atol=1e-1)
      np.testing.assert_allclose(np.std(out, axis=(0, 1, 2)), 1.0, atol=1e-1)

  def test_shared_batchnorm(self):
    """Test that a BN layer can be shared across different data streams.
    """
    with self.test_session():
      # Test single layer reuse
      bn = keras.layers.BatchNormalization()
      x1 = keras.layers.Input(shape=(10,))
      _ = bn(x1)

      x2 = keras.layers.Input(shape=(10,))
      y2 = bn(x2)

      x = np.random.normal(loc=5.0, scale=10.0, size=(2, 10))
      model = keras.models.Model(x2, y2)

      model.compile('sgd', 'mse')
      model.train_on_batch(x, x)

      self.assertEqual(len(bn.updates), 4)
      self.assertEqual(len(model.updates), 2)
      self.assertEqual(len(model.get_updates_for(x1)), 0)
      self.assertEqual(len(model.get_updates_for(x2)), 2)

      # Test model-level reuse
      x3 = keras.layers.Input(shape=(10,))
      y3 = model(x3)
      new_model = keras.models.Model(x3, y3, name='new_model')

      self.assertEqual(len(new_model.updates), 2)
      self.assertEqual(len(model.updates), 4)
      self.assertEqual(len(new_model.get_updates_for(x3)), 2)
      new_model.compile('sgd', 'mse')
      new_model.train_on_batch(x, x)


if __name__ == '__main__':
  test.main()
