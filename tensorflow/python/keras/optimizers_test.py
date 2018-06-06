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
"""Tests for Keras optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test
from tensorflow.python.training.adam import AdamOptimizer


def _get_model(input_dim, num_hidden, output_dim):
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(num_hidden,
                               activation='relu',
                               input_shape=(input_dim,)))
  model.add(keras.layers.Dense(output_dim, activation='softmax'))
  return model


def _test_optimizer(optimizer, target=0.75):
  np.random.seed(1337)
  (x_train, y_train), _ = testing_utils.get_test_data(train_samples=1000,
                                                      test_samples=200,
                                                      input_shape=(10,),
                                                      num_classes=2)
  y_train = keras.utils.to_categorical(y_train)
  model = _get_model(x_train.shape[1], 20, y_train.shape[1])
  model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  history = model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)
  assert history.history['acc'][-1] >= target
  config = keras.optimizers.serialize(optimizer)
  optim = keras.optimizers.deserialize(config)
  new_config = keras.optimizers.serialize(optim)
  new_config['class_name'] = new_config['class_name'].lower()
  assert config == new_config

  # Test constraints.
  model = keras.models.Sequential()
  dense = keras.layers.Dense(10,
                             input_shape=(x_train.shape[1],),
                             kernel_constraint=lambda x: 0. * x + 1.,
                             bias_constraint=lambda x: 0. * x + 2.,
                             activation='relu')
  model.add(dense)
  model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
  model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  model.train_on_batch(x_train[:10], y_train[:10])
  kernel, bias = dense.get_weights()
  np.testing.assert_allclose(kernel, 1., atol=1e-3)
  np.testing.assert_allclose(bias, 2., atol=1e-3)


class KerasOptimizersTest(test.TestCase):

  def test_sgd(self):
    with self.test_session():
      _test_optimizer(keras.optimizers.SGD(lr=0.01,
                                           momentum=0.9,
                                           nesterov=True))

  def test_rmsprop(self):
    with self.test_session():
      _test_optimizer(keras.optimizers.RMSprop())
      _test_optimizer(keras.optimizers.RMSprop(decay=1e-3))

  def test_adagrad(self):
    with self.test_session():
      _test_optimizer(keras.optimizers.Adagrad())
      _test_optimizer(keras.optimizers.Adagrad(decay=1e-3))

  def test_adadelta(self):
    with self.test_session():
      _test_optimizer(keras.optimizers.Adadelta(), target=0.6)
      # Accuracy seems dependent on the initialization. Even adding tf.Print
      # nodes in the graph seemed to affect the initialization seed, and hence
      # the accuracy.
      _test_optimizer(keras.optimizers.Adadelta(decay=1e-3), target=0.4)

  def test_adam(self):
    with self.test_session():
      _test_optimizer(keras.optimizers.Adam())
      _test_optimizer(keras.optimizers.Adam(decay=1e-3))
      _test_optimizer(keras.optimizers.Adam(amsgrad=True))

  def test_adamax(self):
    with self.test_session():
      _test_optimizer(keras.optimizers.Adamax())
      _test_optimizer(keras.optimizers.Adamax(decay=1e-3))

  def test_nadam(self):
    with self.test_session():
      _test_optimizer(keras.optimizers.Nadam())

  def test_clipnorm(self):
    with self.test_session():
      _test_optimizer(keras.optimizers.SGD(lr=0.01,
                                           momentum=0.9,
                                           clipnorm=0.5))

  def test_clipvalue(self):
    with self.test_session():
      _test_optimizer(keras.optimizers.SGD(lr=0.01,
                                           momentum=0.9,
                                           clipvalue=0.5))

  def test_tfoptimizer(self):
    optimizer = keras.optimizers.TFOptimizer(AdamOptimizer(0.01))
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
        2, input_shape=(3,), kernel_constraint=keras.constraints.MaxNorm(1)))
    # This is possible
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.fit(np.random.random((5, 3)),
              np.random.random((5, 2)),
              epochs=1,
              batch_size=5,
              verbose=0)
    # not supported
    with self.assertRaises(NotImplementedError):
      _ = optimizer.weights
    with self.assertRaises(NotImplementedError):
      optimizer.get_config()
    with self.assertRaises(NotImplementedError):
      optimizer.from_config(None)


if __name__ == '__main__':
  test.main()
