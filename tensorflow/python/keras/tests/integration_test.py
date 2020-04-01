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
"""Integration tests for Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import nn_ops as nn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.platform import test


class KerasIntegrationTest(keras_parameterized.TestCase):

  def _save_and_reload_model(self, model):
    self.temp_dir = self.get_temp_dir()
    fpath = os.path.join(self.temp_dir,
                         'test_model_%s' % (random.randint(0, 1e7),))
    if context.executing_eagerly():
      save_format = 'tf'
    else:
      if (not isinstance(model, keras.Sequential) and
          not model._is_graph_network):
        return model  # Not supported
      save_format = 'h5'
    model.save(fpath, save_format=save_format)
    model = keras.models.load_model(fpath)
    return model


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes
class VectorClassificationIntegrationTest(keras_parameterized.TestCase):

  def test_vector_classification(self):
    np.random.seed(1337)
    (x_train, y_train), _ = testing_utils.get_test_data(
        train_samples=100,
        test_samples=0,
        input_shape=(10,),
        num_classes=2)
    y_train = np_utils.to_categorical(y_train)

    model = testing_utils.get_model_from_layers(
        [keras.layers.Dense(16, activation='relu'),
         keras.layers.Dropout(0.1),
         keras.layers.Dense(y_train.shape[-1], activation='softmax')],
        input_shape=x_train.shape[1:])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizer_v2.adam.Adam(0.005),
        metrics=['acc'],
        run_eagerly=testing_utils.should_run_eagerly())
    history = model.fit(x_train, y_train, epochs=10, batch_size=10,
                        validation_data=(x_train, y_train),
                        verbose=2)
    self.assertGreater(history.history['val_acc'][-1], 0.7)
    _, val_acc = model.evaluate(x_train, y_train)
    self.assertAlmostEqual(history.history['val_acc'][-1], val_acc)
    predictions = model.predict(x_train)
    self.assertEqual(predictions.shape, (x_train.shape[0], 2))

  def test_vector_classification_shared_model(self):
    # Test that Sequential models that feature internal updates
    # and internal losses can be shared.
    np.random.seed(1337)
    (x_train, y_train), _ = testing_utils.get_test_data(
        train_samples=100,
        test_samples=0,
        input_shape=(10,),
        num_classes=2)
    y_train = np_utils.to_categorical(y_train)

    base_model = testing_utils.get_model_from_layers(
        [keras.layers.Dense(16,
                            activation='relu',
                            kernel_regularizer=keras.regularizers.l2(1e-5),
                            bias_regularizer=keras.regularizers.l2(1e-5)),
         keras.layers.BatchNormalization()],
        input_shape=x_train.shape[1:])
    x = keras.layers.Input(x_train.shape[1:])
    y = base_model(x)
    y = keras.layers.Dense(y_train.shape[-1], activation='softmax')(y)
    model = keras.models.Model(x, y)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizer_v2.adam.Adam(0.005),
        metrics=['acc'],
        run_eagerly=testing_utils.should_run_eagerly())
    if not testing_utils.should_run_eagerly():
      self.assertEqual(len(model.get_losses_for(None)), 2)
      self.assertEqual(len(model.get_updates_for(x)), 2)
    history = model.fit(x_train, y_train, epochs=10, batch_size=10,
                        validation_data=(x_train, y_train),
                        verbose=2)
    self.assertGreater(history.history['val_acc'][-1], 0.7)
    _, val_acc = model.evaluate(x_train, y_train)
    self.assertAlmostEqual(history.history['val_acc'][-1], val_acc)
    predictions = model.predict(x_train)
    self.assertEqual(predictions.shape, (x_train.shape[0], 2))


@keras_parameterized.run_all_keras_modes
class SequentialIntegrationTest(KerasIntegrationTest):

  def test_sequential_save_and_pop(self):
    # Test the following sequence of actions:
    # - construct a Sequential model and train it
    # - save it
    # - load it
    # - pop its last layer and add a new layer instead
    # - continue training
    np.random.seed(1337)
    (x_train, y_train), _ = testing_utils.get_test_data(
        train_samples=100,
        test_samples=0,
        input_shape=(10,),
        num_classes=2)
    y_train = np_utils.to_categorical(y_train)
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(y_train.shape[-1], activation='softmax')
    ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizer_v2.adam.Adam(0.005),
        metrics=['acc'],
        run_eagerly=testing_utils.should_run_eagerly())
    model.fit(x_train, y_train, epochs=1, batch_size=10,
              validation_data=(x_train, y_train),
              verbose=2)
    model = self._save_and_reload_model(model)

    # TODO(b/134537740): model.pop doesn't update model outputs properly when
    # model.outputs is already defined, so just set to `None` for now.
    model.inputs = None
    model.outputs = None

    model.pop()
    model.add(keras.layers.Dense(y_train.shape[-1], activation='softmax'))

    # TODO(b/134523282): There is an bug with Sequential models, so the model
    # must be marked as compiled=False to ensure the next compile goes through.
    model._is_compiled = False

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizer_v2.adam.Adam(0.005),
        metrics=['acc'],
        run_eagerly=testing_utils.should_run_eagerly())
    history = model.fit(x_train, y_train, epochs=10, batch_size=10,
                        validation_data=(x_train, y_train),
                        verbose=2)
    self.assertGreater(history.history['val_acc'][-1], 0.7)
    model = self._save_and_reload_model(model)
    _, val_acc = model.evaluate(x_train, y_train)
    self.assertAlmostEqual(history.history['val_acc'][-1], val_acc)
    predictions = model.predict(x_train)
    self.assertEqual(predictions.shape, (x_train.shape[0], 2))


# See b/122473407
@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class TimeseriesClassificationIntegrationTest(keras_parameterized.TestCase):

  @keras_parameterized.run_with_all_model_types
  def test_timeseries_classification(self):
    np.random.seed(1337)
    (x_train, y_train), _ = testing_utils.get_test_data(
        train_samples=100,
        test_samples=0,
        input_shape=(4, 10),
        num_classes=2)
    y_train = np_utils.to_categorical(y_train)

    layers = [
        keras.layers.LSTM(5, return_sequences=True),
        keras.layers.GRU(y_train.shape[-1], activation='softmax')
    ]
    model = testing_utils.get_model_from_layers(
        layers, input_shape=x_train.shape[1:])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizer_v2.adam.Adam(0.005),
        metrics=['acc'],
        run_eagerly=testing_utils.should_run_eagerly())
    history = model.fit(x_train, y_train, epochs=15, batch_size=10,
                        validation_data=(x_train, y_train),
                        verbose=2)
    self.assertGreater(history.history['val_acc'][-1], 0.7)
    _, val_acc = model.evaluate(x_train, y_train)
    self.assertAlmostEqual(history.history['val_acc'][-1], val_acc)
    predictions = model.predict(x_train)
    self.assertEqual(predictions.shape, (x_train.shape[0], 2))

  def test_timeseries_classification_sequential_tf_rnn(self):
    np.random.seed(1337)
    (x_train, y_train), _ = testing_utils.get_test_data(
        train_samples=100,
        test_samples=0,
        input_shape=(4, 10),
        num_classes=2)
    y_train = np_utils.to_categorical(y_train)

    with base_layer.keras_style_scope():
      model = keras.models.Sequential()
      model.add(keras.layers.RNN(rnn_cell.LSTMCell(5), return_sequences=True,
                                 input_shape=x_train.shape[1:]))
      model.add(keras.layers.RNN(rnn_cell.GRUCell(y_train.shape[-1],
                                                  activation='softmax',
                                                  dtype=dtypes.float32)))
      model.compile(
          loss='categorical_crossentropy',
          optimizer=keras.optimizer_v2.adam.Adam(0.005),
          metrics=['acc'],
          run_eagerly=testing_utils.should_run_eagerly())

    history = model.fit(x_train, y_train, epochs=15, batch_size=10,
                        validation_data=(x_train, y_train),
                        verbose=2)
    self.assertGreater(history.history['val_acc'][-1], 0.7)
    _, val_acc = model.evaluate(x_train, y_train)
    self.assertAlmostEqual(history.history['val_acc'][-1], val_acc)
    predictions = model.predict(x_train)
    self.assertEqual(predictions.shape, (x_train.shape[0], 2))


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes
class ImageClassificationIntegrationTest(keras_parameterized.TestCase):

  def test_image_classification(self):
    np.random.seed(1337)
    (x_train, y_train), _ = testing_utils.get_test_data(
        train_samples=100,
        test_samples=0,
        input_shape=(10, 10, 3),
        num_classes=2)
    y_train = np_utils.to_categorical(y_train)

    layers = [
        keras.layers.Conv2D(4, 3, padding='same', activation='relu'),
        keras.layers.Conv2D(8, 3, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(8, 3, padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(y_train.shape[-1], activation='softmax')
    ]
    model = testing_utils.get_model_from_layers(
        layers, input_shape=x_train.shape[1:])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizer_v2.adam.Adam(0.005),
        metrics=['acc'],
        run_eagerly=testing_utils.should_run_eagerly())
    history = model.fit(x_train, y_train, epochs=10, batch_size=10,
                        validation_data=(x_train, y_train),
                        verbose=2)
    self.assertGreater(history.history['val_acc'][-1], 0.7)
    _, val_acc = model.evaluate(x_train, y_train)
    self.assertAlmostEqual(history.history['val_acc'][-1], val_acc)
    predictions = model.predict(x_train)
    self.assertEqual(predictions.shape, (x_train.shape[0], 2))


@keras_parameterized.run_all_keras_modes
class ActivationV2IntegrationTest(keras_parameterized.TestCase):
  """Tests activation function V2 in model exporting and loading.

  This test is to verify in TF 2.x, when 'tf.nn.softmax' is used as an
  activation function, its model exporting and loading work as expected.
  Check b/123041942 for details.
  """

  def test_serialization_v2_model(self):
    np.random.seed(1337)
    (x_train, y_train), _ = testing_utils.get_test_data(
        train_samples=100,
        test_samples=0,
        input_shape=(10,),
        num_classes=2)
    y_train = np_utils.to_categorical(y_train)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=x_train.shape[1:]),
        keras.layers.Dense(10, activation=nn.relu),
        # To mimic 'tf.nn.softmax' used in TF 2.x.
        keras.layers.Dense(y_train.shape[-1], activation=nn.softmax_v2),
    ])

    # Check if 'softmax' is in model.get_config().
    last_layer_activation = model.get_layer(index=2).get_config()['activation']
    self.assertEqual(last_layer_activation, 'softmax')

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizer_v2.adam.Adam(0.005),
        metrics=['accuracy'],
        run_eagerly=testing_utils.should_run_eagerly())
    model.fit(x_train, y_train, epochs=2, batch_size=10,
              validation_data=(x_train, y_train),
              verbose=2)

    output_path = os.path.join(self.get_temp_dir(), 'tf_keras_saved_model')
    model.save(output_path, save_format='tf')
    loaded_model = keras.models.load_model(output_path)
    self.assertEqual(model.summary(), loaded_model.summary())

if __name__ == '__main__':
  test.main()
