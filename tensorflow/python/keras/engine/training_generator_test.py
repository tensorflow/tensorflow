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

import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.platform import test
from tensorflow.python.training.rmsprop import RMSPropOptimizer


class TestGeneratorMethods(test.TestCase):

  @unittest.skipIf(
      os.name == 'nt',
      'use_multiprocessing=True does not work on windows properly.')
  def test_generator_methods(self):
    arr_data = np.random.random((50, 2))
    arr_labels = np.random.random((50,))

    def custom_generator():
      batch_size = 10
      num_samples = 50
      while True:
        batch_index = np.random.randint(0, num_samples - batch_size)
        start = batch_index
        end = start + batch_size
        x = arr_data[start: end]
        y = arr_labels[start: end]
        yield x, y

    with self.cached_session():
      x = keras.Input((2,))
      y = keras.layers.Dense(1)(x)
      fn_model = keras.models.Model(x, y)
      fn_model.compile(
          loss='mse',
          optimizer='sgd',
          metrics=['mae', metrics_module.CategoricalAccuracy()])

      seq_model = keras.models.Sequential()
      seq_model.add(keras.layers.Dense(1, input_shape=(2,)))
      seq_model.compile(loss='mse', optimizer='sgd')

      for model in [fn_model, seq_model]:
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

  def test_generator_methods_with_sample_weights(self):
    arr_data = np.random.random((50, 2))
    arr_labels = np.random.random((50,))
    arr_sample_weights = np.random.random((50,))

    def custom_generator():
      batch_size = 10
      num_samples = 50
      while True:
        batch_index = np.random.randint(0, num_samples - batch_size)
        start = batch_index
        end = start + batch_size
        x = arr_data[start: end]
        y = arr_labels[start: end]
        w = arr_sample_weights[start: end]
        yield x, y, w

    with self.cached_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(1, input_shape=(2,)))
      model.compile(
          loss='mse',
          optimizer='sgd',
          metrics=['mae', metrics_module.CategoricalAccuracy()])

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
      model.predict_generator(custom_generator(),
                              steps=5,
                              max_queue_size=10,
                              use_multiprocessing=False)
      model.evaluate_generator(custom_generator(),
                               steps=5,
                               max_queue_size=10,
                               use_multiprocessing=False)

  def test_generator_methods_invalid_use_case(self):

    def custom_generator():
      while 1:
        yield 0

    with self.cached_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(1, input_shape=(2,)))
      model.compile(loss='mse', optimizer='sgd')

      with self.assertRaises(ValueError):
        model.fit_generator(custom_generator(),
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
                            validation_data=custom_generator(),
                            validation_steps=10)
      with self.assertRaises(AttributeError):
        model.predict_generator(custom_generator(),
                                steps=5,
                                max_queue_size=10,
                                use_multiprocessing=False)
      with self.assertRaises(ValueError):
        model.evaluate_generator(custom_generator(),
                                 steps=5,
                                 max_queue_size=10,
                                 use_multiprocessing=False)

  def test_training_with_sequences(self):

    class DummySequence(keras.utils.Sequence):

      def __getitem__(self, idx):
        return np.zeros([10, 2]), np.ones([10])

      def __len__(self):
        return 10

    arr_data = np.random.random((50, 2))
    arr_labels = np.random.random((50,))
    arr_sample_weights = np.random.random((50,))

    def custom_generator():
      batch_size = 10
      num_samples = 50
      while True:
        batch_index = np.random.randint(0, num_samples - batch_size)
        start = batch_index
        end = start + batch_size
        x = arr_data[start: end]
        y = arr_labels[start: end]
        w = arr_sample_weights[start: end]
        yield x, y, w

    with self.cached_session():
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(1, input_shape=(2,)))
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

  @tf_test_util.run_in_graph_and_eager_modes
  def test_generator_input_to_fit_eval_predict(self):
    val_data = np.ones([10, 10], np.float32), np.ones([10, 1], np.float32)

    def custom_generator():
      while True:
        yield np.ones([10, 10], np.float32), np.ones([10, 1], np.float32)

    inputs = keras.layers.Input(shape=(10,))
    x = keras.layers.Dense(10, activation='relu')(inputs)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    model.compile(RMSPropOptimizer(0.001), 'binary_crossentropy')
    model.fit(
        custom_generator(),
        steps_per_epoch=2,
        validation_data=val_data,
        epochs=2)
    model.evaluate(custom_generator(), steps=2)
    model.predict(custom_generator(), steps=2)

  @tf_test_util.run_in_graph_and_eager_modes
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
