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
"""Tests for Keras callbacks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import re
import shutil
import tempfile
import threading
import unittest

import numpy as np

from tensorflow.core.framework import summary_pb2
from tensorflow.python import keras
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import adam

try:
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None

try:
  import requests  # pylint:disable=g-import-not-at-top
except ImportError:
  requests = None


TRAIN_SAMPLES = 10
TEST_SAMPLES = 10
NUM_CLASSES = 2
INPUT_DIM = 3
NUM_HIDDEN = 5
BATCH_SIZE = 5


class Counter(keras.callbacks.Callback):
  """Counts the number of times each callback method was run.

  Attributes:
    method_counts: dict. Contains the counts of time  each callback method was
      run.
  """

  def __init__(self):
    self.method_counts = collections.defaultdict(int)
    methods_to_count = [
        'on_batch_begin', 'on_batch_end', 'on_epoch_begin', 'on_epoch_end',
        'on_predict_batch_begin', 'on_predict_batch_end', 'on_predict_begin',
        'on_predict_end', 'on_test_batch_begin', 'on_test_batch_end',
        'on_test_begin', 'on_test_end', 'on_train_batch_begin',
        'on_train_batch_end', 'on_train_begin', 'on_train_end'
    ]
    for method_name in methods_to_count:
      setattr(self, method_name,
              self.wrap_with_counts(method_name, getattr(self, method_name)))

  def wrap_with_counts(self, method_name, method):

    def _call_and_count(*args, **kwargs):
      self.method_counts[method_name] += 1
      return method(*args, **kwargs)

    return _call_and_count


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes
class CallbackCountsTest(keras_parameterized.TestCase):

  def _check_counts(self, counter, expected_counts):
    """Checks that the counts registered by `counter` are those expected."""
    for method_name, expected_count in expected_counts.items():
      self.assertEqual(
          counter.method_counts[method_name],
          expected_count,
          msg='For method {}: expected {}, got: {}'.format(
              method_name, expected_count, counter.method_counts[method_name]))

  def _get_model(self):
    layers = [
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(10,))
    model.compile(
        adam.AdamOptimizer(0.001),
        'binary_crossentropy',
        run_eagerly=testing_utils.should_run_eagerly())
    return model

  def test_callback_hooks_are_called_in_fit(self):
    x, y = np.ones((10, 10)), np.ones((10, 1))
    val_x, val_y = np.ones((4, 10)), np.ones((4, 1))

    model = self._get_model()
    counter = Counter()
    model.fit(
        x,
        y,
        validation_data=(val_x, val_y),
        batch_size=2,
        epochs=5,
        callbacks=[counter])

    self._check_counts(
        counter, {
            'on_batch_begin': 25,
            'on_batch_end': 25,
            'on_epoch_begin': 5,
            'on_epoch_end': 5,
            'on_predict_batch_begin': 0,
            'on_predict_batch_end': 0,
            'on_predict_begin': 0,
            'on_predict_end': 0,
            'on_test_batch_begin': 10,
            'on_test_batch_end': 10,
            'on_test_begin': 5,
            'on_test_end': 5,
            'on_train_batch_begin': 25,
            'on_train_batch_end': 25,
            'on_train_begin': 1,
            'on_train_end': 1
        })

  def test_callback_hooks_are_called_in_evaluate(self):
    x, y = np.ones((10, 10)), np.ones((10, 1))

    model = self._get_model()
    counter = Counter()
    model.evaluate(x, y, batch_size=2, callbacks=[counter])
    self._check_counts(
        counter, {
            'on_test_batch_begin': 5,
            'on_test_batch_end': 5,
            'on_test_begin': 1,
            'on_test_end': 1
        })

  def test_callback_hooks_are_called_in_predict(self):
    x = np.ones((10, 10))

    model = self._get_model()
    counter = Counter()
    model.predict(x, batch_size=2, callbacks=[counter])
    self._check_counts(
        counter, {
            'on_predict_batch_begin': 5,
            'on_predict_batch_end': 5,
            'on_predict_begin': 1,
            'on_predict_end': 1
        })

  def test_callback_list_methods(self):
    counter = Counter()
    callback_list = keras.callbacks.CallbackList([counter])

    batch = 0
    callback_list.on_test_batch_begin(batch)
    callback_list.on_test_batch_end(batch)
    callback_list.on_predict_batch_begin(batch)
    callback_list.on_predict_batch_end(batch)

    self._check_counts(
        counter, {
            'on_test_batch_begin': 1,
            'on_test_batch_end': 1,
            'on_predict_batch_begin': 1,
            'on_predict_batch_end': 1
        })


class KerasCallbacksTest(test.TestCase):

  def test_ModelCheckpoint(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    with self.cached_session():
      np.random.seed(1337)

      temp_dir = self.get_temp_dir()
      self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

      filepath = os.path.join(temp_dir, 'checkpoint.h5')
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = keras.utils.to_categorical(y_test)
      y_train = keras.utils.to_categorical(y_train)
      # case 1
      monitor = 'val_loss'
      save_best_only = False
      mode = 'auto'

      model = keras.models.Sequential()
      model.add(
          keras.layers.Dense(
              NUM_HIDDEN, input_dim=INPUT_DIM, activation='relu'))
      model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
      model.compile(
          loss='categorical_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy'])

      cbks = [
          keras.callbacks.ModelCheckpoint(
              filepath,
              monitor=monitor,
              save_best_only=save_best_only,
              mode=mode)
      ]
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=1,
          verbose=0)
      assert os.path.exists(filepath)
      os.remove(filepath)

      # case 2
      mode = 'min'
      cbks = [
          keras.callbacks.ModelCheckpoint(
              filepath,
              monitor=monitor,
              save_best_only=save_best_only,
              mode=mode)
      ]
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=1,
          verbose=0)
      assert os.path.exists(filepath)
      os.remove(filepath)

      # case 3
      mode = 'max'
      monitor = 'val_acc'
      cbks = [
          keras.callbacks.ModelCheckpoint(
              filepath,
              monitor=monitor,
              save_best_only=save_best_only,
              mode=mode)
      ]
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=1,
          verbose=0)
      assert os.path.exists(filepath)
      os.remove(filepath)

      # case 4
      save_best_only = True
      cbks = [
          keras.callbacks.ModelCheckpoint(
              filepath,
              monitor=monitor,
              save_best_only=save_best_only,
              mode=mode)
      ]
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=1,
          verbose=0)
      assert os.path.exists(filepath)
      os.remove(filepath)

      # Case: metric not available.
      cbks = [
          keras.callbacks.ModelCheckpoint(
              filepath,
              monitor='unknown',
              save_best_only=True)
      ]
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=1,
          verbose=0)
      # File won't be written.
      assert not os.path.exists(filepath)

      # case 5
      save_best_only = False
      period = 2
      mode = 'auto'

      filepath = os.path.join(temp_dir, 'checkpoint.{epoch:02d}.h5')
      cbks = [
          keras.callbacks.ModelCheckpoint(
              filepath,
              monitor=monitor,
              save_best_only=save_best_only,
              mode=mode,
              period=period)
      ]
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=4,
          verbose=1)
      assert os.path.exists(filepath.format(epoch=2))
      assert os.path.exists(filepath.format(epoch=4))
      os.remove(filepath.format(epoch=2))
      os.remove(filepath.format(epoch=4))
      assert not os.path.exists(filepath.format(epoch=1))
      assert not os.path.exists(filepath.format(epoch=3))

      # Invalid use: this will raise a warning but not an Exception.
      keras.callbacks.ModelCheckpoint(
          filepath,
          monitor=monitor,
          save_best_only=save_best_only,
          mode='unknown')

  def test_EarlyStopping(self):
    with self.cached_session():
      np.random.seed(123)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = keras.utils.to_categorical(y_test)
      y_train = keras.utils.to_categorical(y_train)
      model = testing_utils.get_small_sequential_mlp(
          num_hidden=NUM_HIDDEN, num_classes=NUM_CLASSES, input_dim=INPUT_DIM)
      model.compile(
          loss='categorical_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy'])

      cases = [
          ('max', 'val_acc'),
          ('min', 'val_loss'),
          ('auto', 'val_acc'),
          ('auto', 'loss'),
          ('unknown', 'unknown')
      ]
      for mode, monitor in cases:
        patience = 0
        cbks = [
            keras.callbacks.EarlyStopping(
                patience=patience, monitor=monitor, mode=mode)
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=5,
            verbose=0)

  def test_EarlyStopping_reuse(self):
    with self.cached_session():
      np.random.seed(1337)
      patience = 3
      data = np.random.random((100, 1))
      labels = np.where(data > 0.5, 1, 0)
      model = keras.models.Sequential((keras.layers.Dense(
          1, input_dim=1, activation='relu'), keras.layers.Dense(
              1, activation='sigmoid'),))
      model.compile(
          optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
      weights = model.get_weights()

      stopper = keras.callbacks.EarlyStopping(monitor='acc', patience=patience)
      hist = model.fit(data, labels, callbacks=[stopper], verbose=0, epochs=20)
      assert len(hist.epoch) >= patience

      # This should allow training to go for at least `patience` epochs
      model.set_weights(weights)
      hist = model.fit(data, labels, callbacks=[stopper], verbose=0, epochs=20)
      assert len(hist.epoch) >= patience

  def test_EarlyStopping_with_baseline(self):
    with self.cached_session():
      np.random.seed(1337)
      baseline = 0.5
      (data, labels), _ = testing_utils.get_test_data(
          train_samples=100,
          test_samples=50,
          input_shape=(1,),
          num_classes=NUM_CLASSES)
      model = testing_utils.get_small_sequential_mlp(
          num_hidden=1, num_classes=1, input_dim=1)
      model.compile(
          optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

      stopper = keras.callbacks.EarlyStopping(monitor='acc',
                                              baseline=baseline)
      hist = model.fit(data, labels, callbacks=[stopper], verbose=0, epochs=20)
      assert len(hist.epoch) == 1

      patience = 3
      stopper = keras.callbacks.EarlyStopping(monitor='acc',
                                              patience=patience,
                                              baseline=baseline)
      hist = model.fit(data, labels, callbacks=[stopper], verbose=0, epochs=20)
      assert len(hist.epoch) >= patience

  def test_EarlyStopping_final_weights_when_restoring_model_weights(self):

    class DummyModel(object):

      def __init__(self):
        self.stop_training = False
        self.weights = -1

      def get_weights(self):
        return self.weights

      def set_weights(self, weights):
        self.weights = weights

      def set_weight_to_epoch(self, epoch):
        self.weights = epoch

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=2,
                                               restore_best_weights=True)
    early_stop.model = DummyModel()
    losses = [0.2, 0.15, 0.1, 0.11, 0.12]
    # The best configuration is in the epoch 2 (loss = 0.1000).
    epochs_trained = 0
    early_stop.on_train_begin()
    for epoch in range(len(losses)):
      epochs_trained += 1
      early_stop.model.set_weight_to_epoch(epoch=epoch)
      early_stop.on_epoch_end(epoch, logs={'val_loss': losses[epoch]})
      if early_stop.model.stop_training:
        break
    # The best configuration is in epoch 2 (loss = 0.1000),
    # and while patience = 2, we're restoring the best weights,
    # so we end up at the epoch with the best weights, i.e. epoch 2
    self.assertEqual(early_stop.model.get_weights(), 2)

  def test_RemoteMonitor(self):
    if requests is None:
      return

    monitor = keras.callbacks.RemoteMonitor()
    # This will raise a warning since the default address in unreachable:
    monitor.on_epoch_end(0, logs={'loss': 0.})

  def test_LearningRateScheduler(self):
    with self.cached_session():
      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = keras.utils.to_categorical(y_test)
      y_train = keras.utils.to_categorical(y_train)
      model = testing_utils.get_small_sequential_mlp(
          num_hidden=NUM_HIDDEN, num_classes=NUM_CLASSES, input_dim=INPUT_DIM)
      model.compile(
          loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])

      cbks = [keras.callbacks.LearningRateScheduler(lambda x: 1. / (1. + x))]
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=5,
          verbose=0)
      assert (
          float(keras.backend.get_value(
              model.optimizer.lr)) - 0.2) < keras.backend.epsilon()

      cbks = [keras.callbacks.LearningRateScheduler(lambda x, lr: lr / 2)]
      model.compile(
          loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=2,
          verbose=0)
      assert (
          float(keras.backend.get_value(
              model.optimizer.lr)) - 0.01 / 4) < keras.backend.epsilon()

  def test_ReduceLROnPlateau(self):
    with self.cached_session():
      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = keras.utils.to_categorical(y_test)
      y_train = keras.utils.to_categorical(y_train)

      def make_model():
        random_seed.set_random_seed(1234)
        np.random.seed(1337)
        model = testing_utils.get_small_sequential_mlp(
            num_hidden=NUM_HIDDEN, num_classes=NUM_CLASSES, input_dim=INPUT_DIM)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.SGD(lr=0.1))
        return model

      model = make_model()
      # This should reduce the LR after the first epoch (due to high epsilon).
      cbks = [
          keras.callbacks.ReduceLROnPlateau(
              monitor='val_loss',
              factor=0.1,
              min_delta=10,
              patience=1,
              cooldown=5)
      ]
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=5,
          verbose=0)
      self.assertAllClose(
          float(keras.backend.get_value(model.optimizer.lr)),
          0.01,
          atol=1e-4)

      model = make_model()
      cbks = [
          keras.callbacks.ReduceLROnPlateau(
              monitor='val_loss',
              factor=0.1,
              min_delta=0,
              patience=1,
              cooldown=5)
      ]
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=5,
          verbose=2)
      self.assertAllClose(
          float(keras.backend.get_value(model.optimizer.lr)), 0.1, atol=1e-4)

  def test_ReduceLROnPlateau_patience(self):

    class DummyOptimizer(object):

      def __init__(self):
        self.lr = keras.backend.variable(1.0)

    class DummyModel(object):

      def __init__(self):
        self.optimizer = DummyOptimizer()

    reduce_on_plateau = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', patience=2)
    reduce_on_plateau.model = DummyModel()

    losses = [0.0860, 0.1096, 0.1040]
    lrs = []

    for epoch in range(len(losses)):
      reduce_on_plateau.on_epoch_end(epoch, logs={'val_loss': losses[epoch]})
      lrs.append(keras.backend.get_value(reduce_on_plateau.model.optimizer.lr))

    # The learning rates should be 1.0 except the last one
    for lr in lrs[:-1]:
      self.assertEqual(lr, 1.0)
    self.assertLess(lrs[-1], 1.0)

  def test_ReduceLROnPlateau_backwards_compatibility(self):
    with test.mock.patch.object(logging, 'warning') as mock_log:
      reduce_on_plateau = keras.callbacks.ReduceLROnPlateau(epsilon=1e-13)
      self.assertRegexpMatches(
          str(mock_log.call_args), '`epsilon` argument is deprecated')
    self.assertFalse(hasattr(reduce_on_plateau, 'epsilon'))
    self.assertTrue(hasattr(reduce_on_plateau, 'min_delta'))
    self.assertEqual(reduce_on_plateau.min_delta, 1e-13)

  def test_CSVLogger(self):
    with self.cached_session():
      np.random.seed(1337)
      temp_dir = self.get_temp_dir()
      self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
      filepath = os.path.join(temp_dir, 'log.tsv')

      sep = '\t'
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = keras.utils.to_categorical(y_test)
      y_train = keras.utils.to_categorical(y_train)

      def make_model():
        np.random.seed(1337)
        model = testing_utils.get_small_sequential_mlp(
            num_hidden=NUM_HIDDEN, num_classes=NUM_CLASSES, input_dim=INPUT_DIM)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.SGD(lr=0.1),
            metrics=['accuracy'])
        return model

      # case 1, create new file with defined separator
      model = make_model()
      cbks = [keras.callbacks.CSVLogger(filepath, separator=sep)]
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=1,
          verbose=0)

      assert os.path.exists(filepath)
      with open(filepath) as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read())
      assert dialect.delimiter == sep
      del model
      del cbks

      # case 2, append data to existing file, skip header
      model = make_model()
      cbks = [keras.callbacks.CSVLogger(filepath, separator=sep, append=True)]
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=1,
          verbose=0)

      # case 3, reuse of CSVLogger object
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=2,
          verbose=0)

      with open(filepath) as csvfile:
        list_lines = csvfile.readlines()
        for line in list_lines:
          assert line.count(sep) == 4
        assert len(list_lines) == 5
        output = ' '.join(list_lines)
        assert len(re.findall('epoch', output)) == 1

      os.remove(filepath)

  def test_stop_training_csv(self):
    # Test that using the CSVLogger callback with the TerminateOnNaN callback
    # does not result in invalid CSVs.
    np.random.seed(1337)
    tmpdir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)

    with self.cached_session():
      fp = os.path.join(tmpdir, 'test.csv')
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)

      y_test = keras.utils.to_categorical(y_test)
      y_train = keras.utils.to_categorical(y_train)
      cbks = [keras.callbacks.TerminateOnNaN(), keras.callbacks.CSVLogger(fp)]
      model = keras.models.Sequential()
      for _ in range(5):
        model.add(keras.layers.Dense(2, input_dim=INPUT_DIM, activation='relu'))
      model.add(keras.layers.Dense(NUM_CLASSES, activation='linear'))
      model.compile(loss='mean_squared_error',
                    optimizer='rmsprop')

      def data_generator():
        i = 0
        max_batch_index = len(x_train) // BATCH_SIZE
        tot = 0
        while 1:
          if tot > 3 * len(x_train):
            yield (np.ones([BATCH_SIZE, INPUT_DIM]) * np.nan,
                   np.ones([BATCH_SIZE, NUM_CLASSES]) * np.nan)
          else:
            yield (x_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                   y_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE])
          i += 1
          tot += 1
          i %= max_batch_index

      history = model.fit_generator(data_generator(),
                                    len(x_train) // BATCH_SIZE,
                                    validation_data=(x_test, y_test),
                                    callbacks=cbks,
                                    epochs=20)
      loss = history.history['loss']
      assert len(loss) > 1
      assert loss[-1] == np.inf or np.isnan(loss[-1])

      values = []
      with open(fp) as f:
        for x in csv.reader(f):
          # In windows, due to \r\n line ends we may end up reading empty lines
          # after each line. Skip empty lines.
          if x:
            values.append(x)
      assert 'nan' in values[-1], 'The last epoch was not logged.'

  def test_TerminateOnNaN(self):
    with self.cached_session():
      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)

      y_test = keras.utils.to_categorical(y_test)
      y_train = keras.utils.to_categorical(y_train)
      cbks = [keras.callbacks.TerminateOnNaN()]
      model = keras.models.Sequential()
      initializer = keras.initializers.Constant(value=1e5)
      for _ in range(5):
        model.add(
            keras.layers.Dense(
                2,
                input_dim=INPUT_DIM,
                activation='relu',
                kernel_initializer=initializer))
      model.add(keras.layers.Dense(NUM_CLASSES))
      model.compile(loss='mean_squared_error', optimizer='rmsprop')

      history = model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=20)
      loss = history.history['loss']
      self.assertEqual(len(loss), 1)
      self.assertEqual(loss[0], np.inf)

  @test_util.run_deprecated_v1
  def test_TensorBoard(self):
    np.random.seed(1337)

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

    (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(INPUT_DIM,),
        num_classes=NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test)
    y_train = keras.utils.to_categorical(y_train)

    def data_generator(train):
      if train:
        max_batch_index = len(x_train) // BATCH_SIZE
      else:
        max_batch_index = len(x_test) // BATCH_SIZE
      i = 0
      while 1:
        if train:
          yield (x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],
                 y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
        else:
          yield (x_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],
                 y_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
        i += 1
        i %= max_batch_index

    # case: Sequential
    with self.cached_session():
      model = keras.models.Sequential()
      model.add(
          keras.layers.Dense(
              NUM_HIDDEN, input_dim=INPUT_DIM, activation='relu'))
      # non_trainable_weights: moving_variance, moving_mean
      model.add(keras.layers.BatchNormalization())
      model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
      model.compile(
          loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])
      tsb = keras.callbacks.TensorBoard(
          log_dir=temp_dir, histogram_freq=1, write_images=True,
          write_grads=True, batch_size=5)
      cbks = [tsb]

      # fit with validation data
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=3,
          verbose=0)

      # fit with validation data and accuracy
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=2,
          verbose=0)

      # fit generator with validation data
      model.fit_generator(
          data_generator(True),
          len(x_train),
          epochs=2,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          verbose=0)

      # fit generator without validation data
      # histogram_freq must be zero
      tsb.histogram_freq = 0
      model.fit_generator(
          data_generator(True),
          len(x_train),
          epochs=2,
          callbacks=cbks,
          verbose=0)

      # fit generator with validation data and accuracy
      tsb.histogram_freq = 1
      model.fit_generator(
          data_generator(True),
          len(x_train),
          epochs=2,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          verbose=0)

      # fit generator without validation data and accuracy
      tsb.histogram_freq = 0
      model.fit_generator(
          data_generator(True), len(x_train), epochs=2, callbacks=cbks)
      assert os.path.exists(temp_dir)

  @test_util.run_deprecated_v1
  def test_TensorBoard_multi_input_output(self):
    np.random.seed(1337)
    tmpdir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)

    with self.cached_session():
      filepath = os.path.join(tmpdir, 'logs')

      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = keras.utils.to_categorical(y_test)
      y_train = keras.utils.to_categorical(y_train)

      def data_generator(train):
        if train:
          max_batch_index = len(x_train) // BATCH_SIZE
        else:
          max_batch_index = len(x_test) // BATCH_SIZE
        i = 0
        while 1:
          if train:
            # simulate multi-input/output models
            yield ([x_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]] * 2,
                   [y_train[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]] * 2)
          else:
            yield ([x_test[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]] * 2,
                   [y_test[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]] * 2)
          i += 1
          i %= max_batch_index

      inp1 = keras.Input((INPUT_DIM,))
      inp2 = keras.Input((INPUT_DIM,))
      inp = keras.layers.add([inp1, inp2])
      hidden = keras.layers.Dense(2, activation='relu')(inp)
      hidden = keras.layers.Dropout(0.1)(hidden)
      output1 = keras.layers.Dense(NUM_CLASSES, activation='softmax')(hidden)
      output2 = keras.layers.Dense(NUM_CLASSES, activation='softmax')(hidden)
      model = keras.models.Model([inp1, inp2], [output1, output2])
      model.compile(loss='categorical_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy'])

      # we must generate new callbacks for each test, as they aren't stateless
      def callbacks_factory(histogram_freq):
        return [keras.callbacks.TensorBoard(log_dir=filepath,
                                            histogram_freq=histogram_freq,
                                            write_images=True, write_grads=True,
                                            batch_size=5)]

      # fit without validation data
      model.fit([x_train] * 2, [y_train] * 2, batch_size=BATCH_SIZE,
                callbacks=callbacks_factory(histogram_freq=0), epochs=3)

      # fit with validation data and accuracy
      model.fit([x_train] * 2, [y_train] * 2, batch_size=BATCH_SIZE,
                validation_data=([x_test] * 2, [y_test] * 2),
                callbacks=callbacks_factory(histogram_freq=1), epochs=2)

      # fit generator without validation data
      model.fit_generator(data_generator(True), len(x_train), epochs=2,
                          callbacks=callbacks_factory(histogram_freq=0))

      # fit generator with validation data and accuracy
      model.fit_generator(data_generator(True), len(x_train), epochs=2,
                          validation_data=([x_test] * 2, [y_test] * 2),
                          callbacks=callbacks_factory(histogram_freq=1))
      assert os.path.isdir(filepath)

  @test_util.run_deprecated_v1
  def test_Tensorboard_histogram_summaries_in_test_function(self):

    class FileWriterStub(object):

      def __init__(self, logdir, graph=None):
        self.logdir = logdir
        self.graph = graph
        self.steps_seen = []

      def add_summary(self, summary, global_step):
        summary_obj = summary_pb2.Summary()

        # ensure a valid Summary proto is being sent
        if isinstance(summary, bytes):
          summary_obj.ParseFromString(summary)
        else:
          assert isinstance(summary, summary_pb2.Summary)
          summary_obj = summary

        # keep track of steps seen for the merged_summary op,
        # which contains the histogram summaries
        if len(summary_obj.value) > 1:
          self.steps_seen.append(global_step)

      def flush(self):
        pass

      def close(self):
        pass

    def _init_writer(obj):
      obj.writer = FileWriterStub(obj.log_dir)

    np.random.seed(1337)
    tmpdir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
    (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(INPUT_DIM,),
        num_classes=NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test)
    y_train = keras.utils.to_categorical(y_train)

    with self.cached_session():
      model = keras.models.Sequential()
      model.add(
          keras.layers.Dense(
              NUM_HIDDEN, input_dim=INPUT_DIM, activation='relu'))
      # non_trainable_weights: moving_variance, moving_mean
      model.add(keras.layers.BatchNormalization())
      model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
      model.compile(
          loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])
      keras.callbacks.TensorBoard._init_writer = _init_writer
      tsb = keras.callbacks.TensorBoard(
          log_dir=tmpdir,
          histogram_freq=1,
          write_images=True,
          write_grads=True,
          batch_size=5)
      cbks = [tsb]

      # fit with validation data
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=3,
          verbose=0)

      self.assertAllEqual(tsb.writer.steps_seen, [0, 1, 2, 3, 4, 5])

  @test_util.run_deprecated_v1
  def test_Tensorboard_histogram_summaries_with_generator(self):
    np.random.seed(1337)
    tmpdir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)

    def generator():
      x = np.random.randn(10, 100).astype(np.float32)
      y = np.random.randn(10, 10).astype(np.float32)
      while True:
        yield x, y

    with self.cached_session():
      model = testing_utils.get_small_sequential_mlp(
          num_hidden=10, num_classes=10, input_dim=100)
      model.compile(
          loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])
      tsb = keras.callbacks.TensorBoard(
          log_dir=tmpdir,
          histogram_freq=1,
          write_images=True,
          write_grads=True,
          batch_size=5)
      cbks = [tsb]

      # fit with validation generator
      model.fit_generator(
          generator(),
          steps_per_epoch=2,
          epochs=2,
          validation_data=generator(),
          validation_steps=2,
          callbacks=cbks,
          verbose=0)

      with self.assertRaises(ValueError):
        # fit with validation generator but no
        # validation_steps
        model.fit_generator(
            generator(),
            steps_per_epoch=2,
            epochs=2,
            validation_data=generator(),
            callbacks=cbks,
            verbose=0)

      self.assertTrue(os.path.exists(tmpdir))

  @unittest.skipIf(
      os.name == 'nt',
      'use_multiprocessing=True does not work on windows properly.')
  def test_LambdaCallback(self):
    with self.cached_session():
      np.random.seed(1337)
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = keras.utils.to_categorical(y_test)
      y_train = keras.utils.to_categorical(y_train)
      model = keras.models.Sequential()
      model.add(
          keras.layers.Dense(
              NUM_HIDDEN, input_dim=INPUT_DIM, activation='relu'))
      model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
      model.compile(
          loss='categorical_crossentropy',
          optimizer='sgd',
          metrics=['accuracy'])

      # Start an arbitrary process that should run during model
      # training and be terminated after training has completed.
      e = threading.Event()

      def target():
        e.wait()

      t = threading.Thread(target=target)
      t.start()
      cleanup_callback = keras.callbacks.LambdaCallback(
          on_train_end=lambda logs: e.set())

      cbks = [cleanup_callback]
      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=5,
          verbose=0)
      t.join()
      assert not t.is_alive()

  def test_TensorBoard_with_ReduceLROnPlateau(self):
    with self.cached_session():
      temp_dir = self.get_temp_dir()
      self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = keras.utils.to_categorical(y_test)
      y_train = keras.utils.to_categorical(y_train)

      model = testing_utils.get_small_sequential_mlp(
          num_hidden=NUM_HIDDEN, num_classes=NUM_CLASSES, input_dim=INPUT_DIM)
      model.compile(
          loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

      cbks = [
          keras.callbacks.ReduceLROnPlateau(
              monitor='val_loss', factor=0.5, patience=4, verbose=1),
          keras.callbacks.TensorBoard(log_dir=temp_dir)
      ]

      model.fit(
          x_train,
          y_train,
          batch_size=BATCH_SIZE,
          validation_data=(x_test, y_test),
          callbacks=cbks,
          epochs=2,
          verbose=0)

      assert os.path.exists(temp_dir)

  @test_util.run_deprecated_v1
  def test_Tensorboard_batch_logging(self):

    class FileWriterStub(object):

      def __init__(self, logdir, graph=None):
        self.logdir = logdir
        self.graph = graph
        self.batches_logged = []
        self.summary_values = []
        self.summary_tags = []

      def add_summary(self, summary, step):
        self.summary_values.append(summary.value[0].simple_value)
        self.summary_tags.append(summary.value[0].tag)
        self.batches_logged.append(step)

      def flush(self):
        pass

      def close(self):
        pass

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

    tb_cbk = keras.callbacks.TensorBoard(temp_dir, update_freq='batch')
    tb_cbk.writer = FileWriterStub(temp_dir)

    for batch in range(5):
      tb_cbk.on_batch_end(batch, {'acc': batch})
    self.assertEqual(tb_cbk.writer.batches_logged, [0, 1, 2, 3, 4])
    self.assertEqual(tb_cbk.writer.summary_values, [0., 1., 2., 3., 4.])
    self.assertEqual(tb_cbk.writer.summary_tags, ['batch_acc'] * 5)

  @test_util.run_deprecated_v1
  def test_Tensorboard_epoch_and_batch_logging(self):

    class FileWriterStub(object):

      def __init__(self, logdir, graph=None):
        self.logdir = logdir
        self.graph = graph

      def add_summary(self, summary, step):
        if 'batch_' in summary.value[0].tag:
          self.batch_summary = (step, summary)
        elif 'epoch_' in summary.value[0].tag:
          self.epoch_summary = (step, summary)

      def flush(self):
        pass

      def close(self):
        pass

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

    tb_cbk = keras.callbacks.TensorBoard(temp_dir, update_freq='batch')
    tb_cbk.writer = FileWriterStub(temp_dir)

    tb_cbk.on_batch_end(0, {'acc': 5.0})
    batch_step, batch_summary = tb_cbk.writer.batch_summary
    self.assertEqual(batch_step, 0)
    self.assertEqual(batch_summary.value[0].simple_value, 5.0)

    tb_cbk = keras.callbacks.TensorBoard(temp_dir, update_freq='epoch')
    tb_cbk.writer = FileWriterStub(temp_dir)
    tb_cbk.on_epoch_end(0, {'acc': 10.0})
    epoch_step, epoch_summary = tb_cbk.writer.epoch_summary
    self.assertEqual(epoch_step, 0)
    self.assertEqual(epoch_summary.value[0].simple_value, 10.0)

  @test_util.run_in_graph_and_eager_modes
  def test_Tensorboard_eager(self):
    temp_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

    (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
        train_samples=TRAIN_SAMPLES,
        test_samples=TEST_SAMPLES,
        input_shape=(INPUT_DIM,),
        num_classes=NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test)
    y_train = keras.utils.to_categorical(y_train)

    model = testing_utils.get_small_sequential_mlp(
        num_hidden=NUM_HIDDEN, num_classes=NUM_CLASSES, input_dim=INPUT_DIM)
    model.compile(
        loss='binary_crossentropy',
        optimizer=adam.AdamOptimizer(0.01),
        metrics=['accuracy'])

    cbks = [keras.callbacks.TensorBoard(log_dir=temp_dir)]

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=cbks,
        epochs=2,
        verbose=0)

    self.assertTrue(os.path.exists(temp_dir))

  @test_util.run_deprecated_v1
  def test_TensorBoard_update_freq(self):

    class FileWriterStub(object):

      def __init__(self, logdir, graph=None):
        self.logdir = logdir
        self.graph = graph
        self.batch_summaries = []
        self.epoch_summaries = []

      def add_summary(self, summary, step):
        if 'batch_' in summary.value[0].tag:
          self.batch_summaries.append((step, summary))
        elif 'epoch_' in summary.value[0].tag:
          self.epoch_summaries.append((step, summary))

      def flush(self):
        pass

      def close(self):
        pass

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

    # Epoch mode
    tb_cbk = keras.callbacks.TensorBoard(temp_dir, update_freq='epoch')
    tb_cbk.writer = FileWriterStub(temp_dir)

    tb_cbk.on_batch_end(0, {'acc': 5.0, 'size': 1})
    self.assertEqual(tb_cbk.writer.batch_summaries, [])
    tb_cbk.on_epoch_end(0, {'acc': 10.0, 'size': 1})
    self.assertEqual(len(tb_cbk.writer.epoch_summaries), 1)

    # Batch mode
    tb_cbk = keras.callbacks.TensorBoard(temp_dir, update_freq='batch')
    tb_cbk.writer = FileWriterStub(temp_dir)

    tb_cbk.on_batch_end(0, {'acc': 5.0, 'size': 1})
    self.assertEqual(len(tb_cbk.writer.batch_summaries), 1)
    tb_cbk.on_batch_end(0, {'acc': 5.0, 'size': 1})
    self.assertEqual(len(tb_cbk.writer.batch_summaries), 2)
    self.assertFalse(tb_cbk.writer.epoch_summaries)

    # Integer mode
    tb_cbk = keras.callbacks.TensorBoard(temp_dir, update_freq=20)
    tb_cbk.writer = FileWriterStub(temp_dir)

    tb_cbk.on_batch_end(0, {'acc': 5.0, 'size': 10})
    self.assertFalse(tb_cbk.writer.batch_summaries)
    tb_cbk.on_batch_end(0, {'acc': 5.0, 'size': 10})
    self.assertEqual(len(tb_cbk.writer.batch_summaries), 1)
    tb_cbk.on_batch_end(0, {'acc': 5.0, 'size': 10})
    self.assertEqual(len(tb_cbk.writer.batch_summaries), 1)
    tb_cbk.on_batch_end(0, {'acc': 5.0, 'size': 10})
    self.assertEqual(len(tb_cbk.writer.batch_summaries), 2)
    tb_cbk.on_batch_end(0, {'acc': 10.0, 'size': 10})
    self.assertEqual(len(tb_cbk.writer.batch_summaries), 2)
    self.assertFalse(tb_cbk.writer.epoch_summaries)

  def test_RemoteMonitorWithJsonPayload(self):
    if requests is None:
      self.skipTest('`requests` required to run this test')
    with self.cached_session():
      (x_train, y_train), (x_test, y_test) = testing_utils.get_test_data(
          train_samples=TRAIN_SAMPLES,
          test_samples=TEST_SAMPLES,
          input_shape=(INPUT_DIM,),
          num_classes=NUM_CLASSES)
      y_test = keras.utils.np_utils.to_categorical(y_test)
      y_train = keras.utils.np_utils.to_categorical(y_train)
      model = keras.models.Sequential()
      model.add(
          keras.layers.Dense(
              NUM_HIDDEN, input_dim=INPUT_DIM, activation='relu'))
      model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
      model.compile(
          loss='categorical_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy'])
      cbks = [keras.callbacks.RemoteMonitor(send_as_json=True)]

      with test.mock.patch.object(requests, 'post'):
        model.fit(
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, y_test),
            callbacks=cbks,
            epochs=1)


if __name__ == '__main__':
  test.main()
