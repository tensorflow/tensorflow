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
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test
from tensorflow.python.training.rmsprop import RMSPropOptimizer


def custom_generator(mode=2):
  batch_size = 10
  num_samples = 50
  arr_data = np.random.random((num_samples, 2))
  arr_labels = np.random.random((num_samples, 4))
  arr_weights = np.random.random((num_samples,))
  i = 0
  while True:
    batch_index = i * batch_size % num_samples
    i += 1
    start = batch_index
    end = start + batch_size
    x = arr_data[start: end]
    y = arr_labels[start: end]
    w = arr_weights[start: end]
    if mode == 1:
      yield x
    elif mode == 2:
      yield x, y
    else:
      yield x, y, w


@tf_test_util.run_all_in_graph_and_eager_modes
class TestGeneratorMethods(test.TestCase, parameterized.TestCase):

  @unittest.skipIf(
      os.name == 'nt',
      'use_multiprocessing=True does not work on windows properly.')
  @parameterized.parameters('sequential', 'functional')
  def test_fit_generator_method(self, model_type):
    if model_type == 'sequential':
      model = testing_utils.get_small_sequential_mlp(
          num_hidden=3, num_classes=4, input_dim=2)
    else:
      model = testing_utils.get_small_functional_mlp(
          num_hidden=3, num_classes=4, input_dim=2)
    model.compile(
        loss='mse',
        optimizer='sgd',
        metrics=['mae', metrics_module.CategoricalAccuracy()])

    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        workers=4,
                        use_multiprocessing=True)
    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        use_multiprocessing=False)
    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        use_multiprocessing=False,
                        validation_data=custom_generator(),
                        validation_steps=10)
    model.fit_generator(custom_generator(),
                        steps_per_epoch=5,
                        validation_data=custom_generator(),
                        validation_steps=1,
                        workers=0)

  @unittest.skipIf(
      os.name == 'nt',
      'use_multiprocessing=True does not work on windows properly.')
  @parameterized.parameters('sequential', 'functional')
  def test_evaluate_generator_method(self, model_type):
    if model_type == 'sequential':
      model = testing_utils.get_small_sequential_mlp(
          num_hidden=3, num_classes=4, input_dim=2)
    else:
      model = testing_utils.get_small_functional_mlp(
          num_hidden=3, num_classes=4, input_dim=2)
    model.compile(
        loss='mse',
        optimizer='sgd',
        metrics=['mae', metrics_module.CategoricalAccuracy()])
    model.summary()

    model.evaluate_generator(custom_generator(),
                             steps=5,
                             max_queue_size=10,
                             workers=2,
                             verbose=1,
                             use_multiprocessing=True)
    model.evaluate_generator(custom_generator(),
                             steps=5,
                             max_queue_size=10,
                             use_multiprocessing=False)
    model.evaluate_generator(custom_generator(),
                             steps=5,
                             max_queue_size=10,
                             use_multiprocessing=False,
                             workers=0)

  @unittest.skipIf(
      os.name == 'nt',
      'use_multiprocessing=True does not work on windows properly.')
  @parameterized.parameters('sequential', 'functional')
  def test_predict_generator_method(self, model_type):
    if model_type == 'sequential':
      model = testing_utils.get_small_sequential_mlp(
          num_hidden=3, num_classes=4, input_dim=2)
    else:
      model = testing_utils.get_small_functional_mlp(
          num_hidden=3, num_classes=4, input_dim=2)
    model.compile(
        loss='mse',
        optimizer='sgd',
        metrics=['mae', metrics_module.CategoricalAccuracy()])

    model.predict_generator(custom_generator(),
                            steps=5,
                            max_queue_size=10,
                            workers=2,
                            use_multiprocessing=True)
    model.predict_generator(custom_generator(),
                            steps=5,
                            max_queue_size=10,
                            use_multiprocessing=False)
    model.predict_generator(custom_generator(),
                            steps=5,
                            max_queue_size=10,
                            workers=0)
    # Test generator with just inputs (no targets)
    model.predict_generator(custom_generator(mode=1),
                            steps=5,
                            max_queue_size=10,
                            workers=2,
                            use_multiprocessing=True)
    model.predict_generator(custom_generator(mode=1),
                            steps=5,
                            max_queue_size=10,
                            use_multiprocessing=False)
    model.predict_generator(custom_generator(mode=1),
                            steps=5,
                            max_queue_size=10,
                            workers=0)

  def test_generator_methods_with_sample_weights(self):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(4, input_shape=(2,)))
    model.compile(
        loss='mse',
        optimizer='sgd',
        metrics=['mae', metrics_module.CategoricalAccuracy()])

    model.fit_generator(custom_generator(mode=3),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        use_multiprocessing=False)
    model.fit_generator(custom_generator(mode=3),
                        steps_per_epoch=5,
                        epochs=1,
                        verbose=1,
                        max_queue_size=10,
                        use_multiprocessing=False,
                        validation_data=custom_generator(mode=3),
                        validation_steps=10)
    model.predict_generator(custom_generator(mode=3),
                            steps=5,
                            max_queue_size=10,
                            use_multiprocessing=False)
    model.evaluate_generator(custom_generator(mode=3),
                             steps=5,
                             max_queue_size=10,
                             use_multiprocessing=False)

  def test_generator_methods_invalid_use_case(self):

    def invalid_generator():
      while 1:
        yield 0

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(4, input_shape=(2,)))
    model.compile(loss='mse', optimizer='sgd')

    with self.assertRaises(ValueError):
      model.fit_generator(invalid_generator(),
                          steps_per_epoch=5,
                          epochs=1,
                          verbose=1,
                          max_queue_size=10,
                          use_multiprocessing=False)
    with self.assertRaises(ValueError):
      model.fit_generator(custom_generator(),
                          steps_per_epoch=5,
                          epochs=1,
                          verbose=1,
                          max_queue_size=10,
                          use_multiprocessing=False,
                          validation_data=invalid_generator(),
                          validation_steps=10)
    with self.assertRaises(AttributeError):
      model.predict_generator(invalid_generator(),
                              steps=5,
                              max_queue_size=10,
                              use_multiprocessing=False)
    with self.assertRaises(ValueError):
      model.evaluate_generator(invalid_generator(),
                               steps=5,
                               max_queue_size=10,
                               use_multiprocessing=False)

  def test_generator_input_to_fit_eval_predict(self):
    val_data = np.ones([10, 10], np.float32), np.ones([10, 1], np.float32)

    def ones_generator():
      while True:
        yield np.ones([10, 10], np.float32), np.ones([10, 1], np.float32)

    inputs = keras.layers.Input(shape=(10,))
    x = keras.layers.Dense(10, activation='relu')(inputs)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    model.compile(RMSPropOptimizer(0.001), 'binary_crossentropy')
    model.fit(
        ones_generator(),
        steps_per_epoch=2,
        validation_data=val_data,
        epochs=2)
    model.evaluate(ones_generator(), steps=2)
    model.predict(ones_generator(), steps=2)


@tf_test_util.run_all_in_graph_and_eager_modes
class TestGeneratorMethodsWithSequences(test.TestCase):

  def test_training_with_sequences(self):

    class DummySequence(keras.utils.Sequence):

      def __getitem__(self, idx):
        return np.zeros([10, 2]), np.ones([10, 4])

      def __len__(self):
        return 10

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(4, input_shape=(2,)))
    model.compile(loss='mse', optimizer='sgd')

    model.fit_generator(DummySequence(),
                        steps_per_epoch=10,
                        validation_data=custom_generator(),
                        validation_steps=1,
                        max_queue_size=10,
                        workers=0,
                        use_multiprocessing=True)
    model.fit_generator(DummySequence(),
                        steps_per_epoch=10,
                        validation_data=custom_generator(),
                        validation_steps=1,
                        max_queue_size=10,
                        workers=0,
                        use_multiprocessing=False)

  def test_sequence_input_to_fit_eval_predict(self):
    val_data = np.ones([10, 10], np.float32), np.ones([10, 1], np.float32)

    class CustomSequence(keras.utils.Sequence):

      def __getitem__(self, idx):
        return np.ones([10, 10], np.float32), np.ones([10, 1], np.float32)

      def __len__(self):
        return 2

    inputs = keras.layers.Input(shape=(10,))
    x = keras.layers.Dense(10, activation='relu')(inputs)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    model.compile(RMSPropOptimizer(0.001), 'binary_crossentropy')
    model.fit(CustomSequence(), validation_data=val_data, epochs=2)
    model.evaluate(CustomSequence())
    model.predict(CustomSequence())

    with self.assertRaisesRegexp(ValueError, '`y` argument is not supported'):
      model.fit(CustomSequence(), y=np.ones([10, 1]))

    with self.assertRaisesRegexp(ValueError,
                                 '`sample_weight` argument is not supported'):
      model.fit(CustomSequence(), sample_weight=np.ones([10, 1]))


if __name__ == '__main__':
  test.main()
