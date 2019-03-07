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
import sys
import threading
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import random_seed
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import adam
from tensorflow.python.util import tf_contextlib

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


def _get_numpy():
  return np.ones((10, 10)), np.ones((10, 1))


def _get_sequence():

  class MySequence(keras.utils.data_utils.Sequence):

    def __getitem__(self, _):
      return np.ones((2, 10)), np.ones((2, 1))

    def __len__(self):
      return 5

  return MySequence(), None


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

  @parameterized.named_parameters(('with_numpy', _get_numpy()),
                                  ('with_sequence', _get_sequence()))
  def test_callback_hooks_are_called_in_fit(self, data):
    x, y = data
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

  @parameterized.named_parameters(('with_numpy', _get_numpy()),
                                  ('with_sequence', _get_sequence()))
  def test_callback_hooks_are_called_in_evaluate(self, data):
    x, y = data

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

  @parameterized.named_parameters(('with_numpy', _get_numpy()),
                                  ('with_sequence', _get_sequence()))
  def test_callback_hooks_are_called_in_predict(self, data):
    x = data[0]

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


class KerasCallbacksTest(keras_parameterized.TestCase):

  def _get_model(self, input_shape=None):
    layers = [
        keras.layers.Dense(3, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=input_shape)
    model.compile(
        loss='mse',
        optimizer='rmsprop',
        metrics=[keras.metrics.CategoricalAccuracy(name='my_acc')],
        run_eagerly=testing_utils.should_run_eagerly())
    return model

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.run_all_keras_modes
  def test_progbar_logging(self):
    model = self._get_model(input_shape=(3,))

    x = array_ops.ones((50, 3))
    y = array_ops.zeros((50, 2))
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).batch(10)
    expected_log = r'(.*- loss:.*- my_acc:.*)+'

    with self.captureWritesToStream(sys.stdout) as printed:
      model.fit(dataset, epochs=2, steps_per_epoch=10)
      self.assertRegexpMatches(printed.contents(), expected_log)

  @keras_parameterized.run_with_all_model_types(exclude_models='functional')
  @keras_parameterized.run_all_keras_modes
  def test_progbar_logging_deferred_model_build(self):
    model = self._get_model()
    self.assertFalse(model.built)

    x = array_ops.ones((50, 3))
    y = array_ops.zeros((50, 2))
    dataset = dataset_ops.Dataset.from_tensor_slices((x, y)).batch(10)
    expected_log = r'(.*- loss:.*- my_acc:.*)+'

    with self.captureWritesToStream(sys.stdout) as printed:
      model.fit(dataset, epochs=2, steps_per_epoch=10)
      self.assertRegexpMatches(printed.contents(), expected_log)

  @keras_parameterized.run_with_all_model_types
  def test_ModelCheckpoint(self):
    if h5py is None:
      return  # Skip test if models cannot be saved.

    layers = [
        keras.layers.Dense(NUM_HIDDEN, input_dim=INPUT_DIM, activation='relu'),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(10,))
    model.compile(
        loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)

    filepath = os.path.join(temp_dir, 'checkpoint')
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
        loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

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
          loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

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
          optimizer='sgd', loss='binary_crossentropy', metrics=['acc'])

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

      # TODO(psv): Make sure the callback works correctly when min_delta is
      # set as 0. Test fails when the order of this callback and assertion is
      # interchanged.
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
          verbose=0)
      self.assertAllClose(
          float(keras.backend.get_value(model.optimizer.lr)), 0.1, atol=1e-4)

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
          verbose=2)
      self.assertAllClose(
          float(keras.backend.get_value(model.optimizer.lr)), 0.01, atol=1e-4)

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


# A summary that was emitted during a test. Fields:
#   logdir: str. The logdir of the FileWriter to which the summary was
#     written.
#   tag: str. The name of the summary.
_ObservedSummary = collections.namedtuple('_ObservedSummary', ('logdir', 'tag'))


class _MockSummaryFile(object):
  """Record summary tag names and the files to which they're written.

  Fields `scalars`, `images`, and `histograms` are sets containing
  `_ObservedSummary` values.
  """

  def __init__(self):
    self.scalars = set()
    self.images = set()
    self.histograms = set()


@tf_contextlib.contextmanager
def _mock_summary_api():
  summary_file = _MockSummaryFile()

  # Keep track of the logdir associated with each created resource.
  # (There doesn't seem to be an easy way to get this information after
  # the fact.)
  resource_logdirs = {}
  real_create_file_writer = summary_ops_v2.create_file_writer

  def mock_create_file_writer(logdir, *args, **kwargs):
    writer = real_create_file_writer(logdir, *args, **kwargs)
    resource = writer._resource
    assert resource is not None
    assert resource not in resource_logdirs, (resource, resource_logdirs)
    resource_logdirs[resource] = logdir
    return writer

  def make_mock_summary(summary_set):

    def mock_summary(tag, *args, **kwargs):
      del args  # unused
      del kwargs  # unused
      resource = context.context().summary_writer_resource
      logdir = resource_logdirs[resource]
      summary_set.add(_ObservedSummary(logdir=logdir, tag=tag))

    return mock_summary

  with test.mock.patch.object(summary_ops_v2,
                              'create_file_writer',
                              mock_create_file_writer), \
        test.mock.patch.object(summary_ops_v2,
                               'scalar',
                               make_mock_summary(summary_file.scalars)), \
        test.mock.patch.object(summary_ops_v2,
                               'histogram',
                               make_mock_summary(summary_file.histograms)), \
        test.mock.patch.object(summary_ops_v2,
                               'image',
                               make_mock_summary(summary_file.images)):
    yield summary_file


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class TestTensorBoardV2(keras_parameterized.TestCase):

  def setUp(self):
    super(TestTensorBoardV2, self).setUp()
    self.logdir = os.path.join(self.get_temp_dir(), 'tb')
    self.train_dir = os.path.join(self.logdir, 'train')
    self.validation_dir = os.path.join(self.logdir, 'validation')

  def _get_model(self):
    layers = [
        keras.layers.Conv2D(8, (3, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(1)
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(10, 10, 1))
    model.compile('sgd', 'mse', run_eagerly=testing_utils.should_run_eagerly())
    return model

  def test_TensorBoard_basic(self):
    model = self._get_model()
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = keras.callbacks.TensorBoard(self.logdir)

    with _mock_summary_api() as summary_file:
      model.fit(
          x,
          y,
          batch_size=2,
          epochs=2,
          validation_data=(x, y),
          callbacks=[tb_cbk])

    self.assertEqual(
        summary_file.scalars, {
            _ObservedSummary(logdir=self.train_dir, tag='epoch_loss'),
            _ObservedSummary(logdir=self.validation_dir, tag='epoch_loss'),
        })

  def test_TensorBoard_batch_metrics(self):
    model = self._get_model()
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = keras.callbacks.TensorBoard(self.logdir, update_freq=1)

    with _mock_summary_api() as summary_file:
      model.fit(
          x,
          y,
          batch_size=2,
          epochs=2,
          validation_data=(x, y),
          callbacks=[tb_cbk])

    self.assertEqual(
        summary_file.scalars,
        {
            _ObservedSummary(logdir=self.train_dir, tag='batch_loss'),
            _ObservedSummary(logdir=self.train_dir, tag='epoch_loss'),
            _ObservedSummary(logdir=self.validation_dir, tag='epoch_loss'),
        },
    )

  def test_TensorBoard_weight_histograms(self):
    model = self._get_model()
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = keras.callbacks.TensorBoard(self.logdir, histogram_freq=1)

    with _mock_summary_api() as summary_file:
      model.fit(
          x,
          y,
          batch_size=2,
          epochs=2,
          validation_data=(x, y),
          callbacks=[tb_cbk])

    self.assertEqual(
        summary_file.scalars,
        {
            _ObservedSummary(logdir=self.train_dir, tag='epoch_loss'),
            _ObservedSummary(logdir=self.validation_dir, tag='epoch_loss'),
        },
    )
    self.assertEqual(
        self._strip_layer_names(summary_file.histograms),
        {
            _ObservedSummary(logdir=self.train_dir, tag='bias_0'),
            _ObservedSummary(logdir=self.train_dir, tag='kernel_0'),
        },
    )

  def test_TensorBoard_weight_images(self):
    model = self._get_model()
    x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
    tb_cbk = keras.callbacks.TensorBoard(
        self.logdir, histogram_freq=1, write_images=True)

    with _mock_summary_api() as summary_file:
      model.fit(
          x,
          y,
          batch_size=2,
          epochs=2,
          validation_data=(x, y),
          callbacks=[tb_cbk])

    self.assertEqual(
        summary_file.scalars,
        {
            _ObservedSummary(logdir=self.train_dir, tag='epoch_loss'),
            _ObservedSummary(logdir=self.validation_dir, tag='epoch_loss'),
        },
    )
    self.assertEqual(
        self._strip_layer_names(summary_file.histograms),
        {
            _ObservedSummary(logdir=self.train_dir, tag='bias_0'),
            _ObservedSummary(logdir=self.train_dir, tag='kernel_0'),
        },
    )
    self.assertEqual(
        self._strip_layer_names(summary_file.images),
        {
            _ObservedSummary(logdir=self.train_dir, tag='bias_0'),
            _ObservedSummary(logdir=self.train_dir, tag='kernel_0'),
        },
    )

  def _strip_layer_names(self, summaries):
    """Deduplicate summary names modulo layer suffix.

    Args:
      summaries: A `set` of `_ObservedSummary` values.

    Returns:
      A new `set` of `_ObservedSummary` values with layer suffixes
      removed.
    """
    return {s._replace(tag=s.tag[s.tag.rfind('/') + 1:]) for s in summaries}

  def test_TensorBoard_invalid_argument(self):
    with self.assertRaisesRegexp(ValueError, 'Unrecognized arguments'):
      keras.callbacks.TensorBoard(wwrite_images=True)


if __name__ == '__main__':
  test.main()
